FROM python:3

ENV PYTHONUNBUFFERED 1

RUN mkdir /game_project

WORKDIR /game_project

COPY requirements-phase1.txt /game_project/
RUN pip install -r requirements-phase1.txt

COPY kanarya_config.json /etc/

COPY . /game_project/

COPY requirements-phase2.txt /game_project/
RUN pip install -r requirements-phase2.txt

RUN python manage.py migrate --noinput

# RUN python manage.py createsuperuser
# Run with docker exec -it ${container_name} python manage.py createsuperuser