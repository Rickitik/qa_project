import numpy as np
from flask import Flask, request, jsonify
import nmslib
import joblib
import os
from typing import List
import redis
import sys
import signal

GEN_NUM = os.getenv("GEN_NUM")
CLUSTER_NUM = os.getenv("CLUSTER_NUM")


def redis_connect() -> redis.client.Redis:
    try:
        client = redis.Redis(
            host="redis",
            port=6379,
            password="rickytik",
            decode_responses=True,
            db=0,
            socket_timeout=5,
        )
        ping = client.ping()
        if ping is True:
            return client
    except redis.AuthenticationError:
        print("AuthenticationError")
        sys.exit(1)


def redis_on_start():
    """
    Если включается первая реплика, то вносим данные по индексу в регистр, если поднимается вторая реплика того же
    поколения, то ничего не делаем. Если видим, что приложение поднято, но поколение не то, заносим в регистр данные
    кластера для последующей подмены
    :return:
    """
    client = redis_connect()
    path_to_centers = f"/var/models/ann_centers/{GEN_NUM}"
    try:
        cluster_centers = joblib.load(os.path.join(path_to_centers, f"clusters_centers_use_dg{GEN_NUM}.pkl"))
    except FileNotFoundError as e:
        print(f"Can't load clusters centers file: {e}")
        raise

    # конвертируем вектор центра кластера в байты
    cluster_center = cluster_centers[CLUSTER_NUM].tobytes()

    # если кластера нет в списке, добавляем актуальные центры кластеров - нормальный запуск
    clusters_list = client.lrange("clusters", 0, -1)
    if CLUSTER_NUM not in clusters_list:
        client.rpush("clusters", CLUSTER_NUM)
        client.hset(f"{CLUSTER_NUM}_actual", "center", cluster_center)
        client.hset(f"{CLUSTER_NUM}_actual", "dg", GEN_NUM)

    elif (CLUSTER_NUM in clusters_list) and (GEN_NUM != client.hget(f"{CLUSTER_NUM}_actual", "dg")):
        # если центр есть в списке поднятых, добавляем обновленный центр в предварительный список
        client.hset(f"{CLUSTER_NUM}_booting", "center", cluster_center)
        client.hset(f"{CLUSTER_NUM}_booting", "dg", GEN_NUM)
        client.expire(f"{CLUSTER_NUM}_booting", 25)

    client.close()


def redis_on_finish(sig, frame):
    """ при выключении проверяем, существует ли список на загрузку (node_<x>_booting), если да, то действующий
    список удаляем, предвариетльный колируем в действующим. Если предварительного списка нет,
    значит выключение штатное и просто удаляем данные из регистра.


    """
    client = redis_connect()

    # проверяем первая ли реплика отключается, если вторая, то убираем информацию из хранилища
    if client.hget(f"{CLUSTER_NUM}_actual_", "1_replica_isdown") is None:
        client.hset(f"{CLUSTER_NUM}_actual_", "1_replica_isdown", "true")
        client.expire(f"{CLUSTER_NUM}_actual_", 20)

    # список на замену существует?
    elif client.hlen(f"{CLUSTER_NUM}_booting"):

        # сохраняем данные в словаре и перезаписываем в действующий список центров.
        actual_dict = {}
        keys = client.hkeys(f"{CLUSTER_NUM}_booting")
        for key in keys:
            actual_dict[key] = client.hget(f"{CLUSTER_NUM}_booting", key)

        client.delete(f"{CLUSTER_NUM}_actual")
        client.delete(f"{CLUSTER_NUM}_booting")
        for item in actual_dict:
            client.hset(f"{CLUSTER_NUM}_actual", item, actual_dict[item])

    else:
        # штатно выключаем приложение
        client.lrem("clusters", 1, CLUSTER_NUM)
        client.delete(f"{CLUSTER_NUM}_actual")

    client.close()
    os.kill(os.getpid(), signal.SIGKILL)


signal.signal(signal.SIGTERM, redis_on_finish)
signal.signal(signal.SIGINT, redis_on_finish)


class KNNServe:
    def __init__(self):
        self.indexes = {}
        self.mappers = {}
        self.load_indexes(int(GEN_NUM))

    def get_data(self, clust_num, q, k):
        """Получаем ИД соседей по кластеру и мапируем в вопрос"""
        ids, _ = self.indexes[clust_num].knnQuery(q, k=k)
        res = list(map(lambda x: self.mappers[clust_num][x], ids))
        return res

    def load_indexes(self, gen_num: int):
        """ Загружает индексы и маперы кластеров в атрибуты в зависимости от заданного поколения"""
        # настройки инициализации индексов
        space_name = 'l2'
        efS = 100
        query_time_params = {'efSearch': efS}

        path_to_model = f"/var/models/ann_index/{gen_num}"
        path_to_mapper = f"/var/models/ann_mapper/{gen_num}"

        children = os.listdir(path_to_model)
        for child in children:
            name = child.split(".")
            if not len(name) > 1:
                index_name = name[0][0]
                self.indexes[index_name] = nmslib.init(method='hnsw', space=space_name, data_type=nmslib.DataType.DENSE_VECTOR)
                self.indexes[index_name].loadIndex(os.path.join(path_to_model, child), load_data=True)
                self.indexes[index_name].setQueryTimeParams(query_time_params)
                self.mappers[index_name] = joblib.load(os.path.join(path_to_mapper, f"ind_str_mapper_{index_name}_cluster.pkl"))


knn_client = KNNServe()
redis_on_start()
app = Flask(__name__)


@app.route('/get_k_near_items', methods=['POST'])
def get_neighbors():
    """
    Ручка API, получает запрос с номером кластера, эмбеддингом вопроса и количеством соседей, которые необходимо вернуть
    :return: словарь с найденными соседями и статусом
    """
    data = request.json
    params = data['params']
    claster_num = params.get("cls_num")
    q = np.array(data['arr'], dtype=np.float32)  # Эмбеддинг вопроса
    k = params.get("k")

    data = knn_client.get_data(claster_num, q, k)

    if data is not None:
        return jsonify({"items": data, "status": "OK"})
    else:
        return jsonify({"items": [], "status": "not found"})


@app.route('/healthcheck')
def check():
    return 'ok'


if __name__ == '__main__':
    app.run()
