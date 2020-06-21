FROM python:3

ENV PYTHONUNBUFFERED 1

RUN mkdir /game_project

WORKDIR /game_project

COPY . /game_project/

COPY requirements-phase1.txt /game_project/
COPY requirements-phase2.txt /game_project/

RUN pip install -r requirements-phase1.txt

RUN pip install -r requirements-phase2.txt

# RUN python manage.py migrate

# RUN python manage.py createsuperuser