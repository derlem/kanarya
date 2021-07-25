from django.urls import path

from . import views

app_name = "spellchecker"
urlpatterns = [

    path('bul/', views.query, name='query'),
    path('geribildirim/', views.feedback, name='feedback'),
    path('bul/<random_id>', views.query_detail, name='query_detail')

]