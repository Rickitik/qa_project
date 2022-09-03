FROM python:3.9

RUN mkdir -p /var/models
WORKDIR /code

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5050

RUN pip install -U pip

RUN pip install nmslib flask joblib redis

EXPOSE 5050

COPY app.py app.py
HEALTHCHECK --interval=20s --retries=3 CMD curl -f 'http://localhost:5050/healthcheck'

CMD ["flask", "run"]
