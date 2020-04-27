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
$ python manage.py migrate
```

Then create a super user with the following command:

```console
$ python manage.py createsuperuser
```

Lastly, run the server:

```console
$ python manage.py runserver
```

Go to `http://127.0.0.1:8000/login` and login with the super user.

