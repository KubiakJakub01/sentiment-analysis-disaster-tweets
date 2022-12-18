FROM tensorflow/tensorflow:latest-gpu

WORKDIR /app

COPY conda_requirements.txt /app
COPY data /app/data

RUN pip install --upgrade pip
RUN pip install -r /app/conda_requirements.txt
