FROM jupyter/scipy-notebook

# don't save caches
ENV PIP_NO_CACHE_DIR=1
# avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# stream outputs to logs
ENV PYTHONUNBUFFERED 1

# jupyter notebook
EXPOSE 8888

# mlflow server
EXPOSE 5000

RUN pip install mlflow scikit-learn

