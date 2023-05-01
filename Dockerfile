# syntax=docker/dockerfile:1.4
FROM rayproject/ray-ml:2.4.0-py39-cpu
#install odbc driver and postgresql client to connect to the database for troubleshooting
RUN sudo apt-get update && \
    apt-get clean &&  \
    apt-get indtall -y unixodbc-dev postgresql-client && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \

RUN $HOME/anaconda3/bin/pip --no-cache-dir install -U pip setuptools mlflow pyodbc sqlalchemy psycopg2-binary
