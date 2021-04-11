# How to run the web server

First, create a **python3** virtual environment. If you have `virtualenvwrapper` installed in your computer, you can create and activate a virtual environment with the following commands:

```console
$ mkvirtualenv kanarya_game
$ workon kanarya_game
```

Afterwards, clone the repo, install the requirements and run the migrations:

```console
$ git clone https://github.com/derlem/kanarya.git
$ cd kanarya/
$ git checkout --track origin/game_project
$ cd game_project/
$ pip install -r requirements.txt
$ #Get the spellchecker model if you have access to minerva. For example:
$ scp path_to_the_spellchecker_model spellchecker/static/
$ python manage.py migrate
```

Create a configuration file for the application:

```console
sudo touch /etc/kanarya_config.json
```

Afterwards fill in the following three variables inside this file:

```consolve
{
  "SECRET_KEY": "write_down_random_50_characters_here",
  "EMAIL_USER": "username_for_email_service",
  "EMAIL_PASS": "password_for_email_service"
}
```


Then create a super user with the following command:

```console
$ python manage.py createsuperuser
```

This version is in the production mode. If you want to run it in the development mode, set the `DEBUG` setting to `True`
```console
$ vim game_project/settings.py
$ #Set the DEBUG to True
```

Lastly, run the server:

```console
$ python manage.py runserver
```

Go to `http://127.0.0.1:8000/`

# Troubleshooting:

- While deploying the app to a server, some libraries create problems. https://serverfault.com/questions/844761/wsgi-truncated-or-oversized-response-headers-received-from-daemon-process 

