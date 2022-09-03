FROM python:3.9

RUN mkdir -p /var/models
WORKDIR /code

ENV FLASK_APP=api.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5050

RUN pip install -U pip
RUN pip install pipenv

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock

RUN pipenv install --system --deploy

EXPOSE 5050

COPY api.py api.py

HEALTHCHECK --interval=20s --retries=3 CMD curl -f 'http://localhost:5050/healthcheck'


CMD ["flask", "run"]
