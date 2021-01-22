FROM python:3.8-slim

WORKDIR /app
COPY . .
RUN apt-get update && apt-get -y install gcc mono-mcs
RUN pip install --trusted-host pypi.python.org -r requirements.txt 

CMD python app.py
EXPOSE 80