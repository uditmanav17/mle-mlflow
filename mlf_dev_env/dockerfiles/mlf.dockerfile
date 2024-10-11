FROM python:3.11

RUN pip install mlflow psycopg2 boto3

EXPOSE 5000

CMD [ "mlfow", "server", "--host", "0.0.0.0", "--port", "5000"]