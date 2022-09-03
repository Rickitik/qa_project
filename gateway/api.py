import signal
from flask import Flask, request, jsonify
import json
import redis
import sys
import httpx
import os
from datetime import timedelta
import numpy as np

EMB_URL = os.getenv("EMBEDDER_URL", "embedder")
EMB_PORT = os.getenv("EMBEDDER_PORT", "8501")
MODEL_NAME = os.getenv("MODEL_NAME", "my_model")
INDEX_PORT = os.getenv("INDEX_PORT", "5000")


def signal_handler(sig, frame):
    redis_client.close()
    sys.exit(0)


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


def redis_connect() -> redis.client.Redis:
    try:
        client = redis.Redis(
            host="redis",
            port=6379,
            password="rickytik",
            # decode_responses=True,
            db=0,
            socket_timeout=5,
        )
        ping = client.ping()
        if ping is True:
            return client
    except redis.AuthenticationError:
        print("Authentication Error")
        sys.exit(1)


redis_client = redis_connect()


def get_data_from_api(msg: str, max_distance: float = 1.5) -> dict:
    """
        Получает эмбеддинг вопроса и отсылает его в модель поиска соседей
        msg - string: вопрос
        max_distance - float: предел расстояния до центров, после которого идет отказ (L2)
        :return json ответа апишки поиска соседей
    """
    with httpx.Client() as cl:

        # получим эмбеддинг вопроса
        emb_url = f"http://{EMB_URL}:{EMB_PORT}/v1/models/{MODEL_NAME}:predict"
        response = cl.post(emb_url, json={"instances": [msg]})
        res = response.json()
        pred = res['predictions'][0]
        pred = np.array(pred)

        # найдем ближайший центр кластера из зарегистрированных на данный момент в Redis
        nearest_center = get_nearest_center(pred, max_distance)
        if nearest_center is None:
            return {"items": [], "status": "not found"}

        # получим топ 10 ближайших соседей
        ind_url = f"http://cluster_{nearest_center}:{INDEX_PORT}/get_k_near_items"
        params = {'cls_num': nearest_center, "k": 10}
        data = {'params': params, 'arr': pred.tolist()}

        response = cl.post(ind_url, json=data)

        return response.json()


def get_data_from_cache(key: str) -> str:
    """Data from redis."""

    val = redis_client.get(key)
    return val


def set_data_to_cache(key: str, value: str) -> bool:
    """Data to redis."""

    state = redis_client.setex(key, timedelta(seconds=3600), value=value)
    return state


def get_nearest_center(pred: np.ndarray, max_distance: float) -> str:
    """ Получает из редиса список доступных центров, и вычисляет расстояние, если расстояние большое возвращает None"""
    clusters_list = redis_client.lrange("clusters", 0, -1)
    centers = {}
    for cluster in clusters_list:
        cluster = cluster.decode("utf-8")
        encoded_vector = redis_client.hget(f"{cluster}_actual", "center")
        centers[cluster] = np.frombuffer(encoded_vector, dtype=np.float32)

    min_dist = max_distance
    min_center = None
    for center in centers:
        dist_ = distance(pred, centers[center])
        if dist_ <= min_dist:
            min_dist = dist_
            min_center = center

    return min_center


def distance(point_a: np.ndarray, center: np.ndarray) -> float:
    """ Расчет расстояния до центров кластеров"""

    res = np.linalg.norm(point_a - center)
    return res


def find_dublicate(question: str) -> dict:
    """Поиск дубликатов. Вначале в кэше, если там нет, то получает ембединг и делает запрос в api ann сервис
        В кэше содержится key: question:str, value: List[str] - список индексов вопросов
    """

    # Поиск в кеше
    data = get_data_from_cache(key=question)

    # Если найдено в кеше....
    if data is not None:
        data = json.loads(data)
        data["cache"] = True
        return data

    else:
        # Если отсутствует в кеше, получаем эмбеддинг и запрашиваем индексы
        data = get_data_from_api(question)

        # Сохраняем в кеше
        if data.get("status") == "OK":
            data["cache"] = False
            data = json.dumps(data)
            state = set_data_to_cache(key=question, value=data)

            if state is True:
                return json.loads(data)
        return data


app = Flask(__name__)


@app.route('/get_neighbors')
def get_neighbors():
    args = request.args
    msg = args.get("msg", type=str)
    res = find_dublicate(msg)
    return jsonify(res)


@app.route('/healthcheck')
def check():
    return 'ok'


if __name__ == '__main__':
    app.run()
