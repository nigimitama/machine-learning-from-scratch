FROM python:3.9
WORKDIR /workdir
COPY requirements.txt .
RUN pip3 install -r requirements.txt
