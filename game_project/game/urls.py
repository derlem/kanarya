from django.urls import path

from . import views


urlpatterns = [
	
	path('home/', views.home, name='home'),
	path('about/', views.about, name='about'),
	path('question/', views.question, name='question'),
	path('answer/', views.answer, name='answer'),
	path('test_end/', views.test_end, name='test_end'),
	path('stats/', views.stats, name='stats'),
	#path('onamformu/', views.onamformu, name='onamformu'),
	path('proficiency/', views.proficiency, name='proficiency'),
	path('prof_end/', views.prof_end, name='prof_end'),
	path('welcome/', views.welcome, name='welcome'),

]