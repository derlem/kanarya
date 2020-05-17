from django.urls import path

from . import views


urlpatterns = [
	
	path('query/', views.query, name='spellchecker_query'),
	path('answer/', views.answer, name='spellchecker_answer')

]