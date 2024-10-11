FROM continuumio/miniconda3

# don't save caches
ENV PIP_NO_CACHE_DIR=1
# avoid .pyc files
ENV PYTHONDONTWRITEBYTECODE 1
# stream outputs to logs
ENV PYTHONUNBUFFERED 1

RUN pip install \
    requests==2.31.0 \
    mlflow>=1.18.0 \
    numpy \
    scipy \
    pandas \
    scikit-learn \
    cloudpickle \
    pandas_datareader==0.10.0 \
    yfinance


