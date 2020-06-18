FROM python:3

ENV PYTHONUNBUFFERED 1

RUN mkdir /game_project

WORKDIR /game_project

COPY requirements.txt /game_project/

RUN pip install -r requirements.txt

COPY . /game_project/

# RUN python manage.py migrate

# RUN python manage.py createsuperuser