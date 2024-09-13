FROM python:3.7.9-stretch

RUN pip install pipenv
RUN apt-get update && apt-get install -y curl
RUN apt-get install -y tar
RUN pipenv install

WORKDIR /app
COPY settings.py .
COPY server.py .
COPY configure_and_run.py .
ENTRYPOINT ["pipenv", "run", "python", "configure_and_run.py", "&"]
