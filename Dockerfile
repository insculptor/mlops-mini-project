FROM python:3.11

WORKDIR /app

COPY flask_app/ /app
COPY models/ /app/models
COPY ./requirements.txt /app/requirements.txt
RUN mkdir /app/flask_app
COPY flask_app/* flask_app/


RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install dagshub
RUN python -m nltk.downloader stopwords wordnet
RUN python -m nltk.downloader punkt

EXPOSE 5001

CMD ["gunicorn", "--bind", "0.0.0.0:5001", "--timeout", "120", "app:app"]
