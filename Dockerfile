FROM de.icr.io/basic-package/python:3.8-slim-buster
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip3 install -r requirements.txt
COPY . .
#CMD ["python3","-m","docker imagesflask","run","--host=0.0.0.0"]
CMD ["gunicorn", "--workers=1", "--bind", "0.0.0.0:5000", "--timeout", "500", "app:app"]
#CMD [ "waitress", "-serve", "--listen=*:5000", "wsgi:app"]