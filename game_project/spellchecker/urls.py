from django.urls import path

from . import views


urlpatterns = [
	
	path('bul/', views.query, name='spellchecker_query'),

]