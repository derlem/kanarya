from django.urls import path

from . import views


urlpatterns = [
	
	path('anasayfa/', views.home, name='home'),
	path('hakkimizda/', views.about, name='about'),
	path('soru/', views.question, name='question'),
	path('cevap/', views.answer, name='answer'),
	path('test_bitis/', views.test_end, name='test_end'),
	path('stats/', views.stats, name='stats'),
	#path('onamformu/', views.onamformu, name='onamformu'),
	#path('proficiency/', views.proficiency, name='proficiency'),
	#path('prof_end/', views.prof_end, name='prof_end'),
	#path('welcome/', views.welcome, name='welcome'),
	path('isinma_soru/', views.warmup_question, name='warmup_question'),
	path('isinma_cevap/', views.warmup_answer, name='warmup_answer'),
	path('isinma_bitis/', views.warmup_end, name='warmup_end'),
	path('stats/sentencecounts', views.sentence_counts, name='sentence_counts'),
	path('stats/decision/csv/', views.stats_decision_csv, name='stats_decision_csv'),
	path('stats/sentence/csv/', views.stats_sentence_csv, name='stats_sentence_csv'),


]