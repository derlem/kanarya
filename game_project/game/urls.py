from django.urls import path

from . import views


urlpatterns = [
	
	path('home/', views.home, name='home'),
	path('about/', views.about, name='about'),
	path('question/', views.question, name='question'),
	path('answer/', views.answer, name='answer'),

]