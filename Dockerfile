FROM tensorflow/tensorflow:2.11.0-gpu

WORKDIR /app

COPY requirements.txt /app
COPY data /app/data

RUN pip install --upgrade pip
RUN pip install -r /app/requirements.txt

COPY src /app
