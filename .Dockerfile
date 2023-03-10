FROM python:3.8

RUN mkdir /app

WORKDIR /app/src
COPY ./ /app/src
RUN pip install -r requirements.txt

CMD ["python", "-u", "server.py"]
