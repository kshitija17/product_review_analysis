FROM python:3.11.3-buster
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 3000
# CMD gunicorn --workers=4 --bind 0.0.0.0:$PORT app:app
CMD python -m src.app