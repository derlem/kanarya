from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
import csv
import linecache
import random
from .forms import ActivityForm

from .models import Sentence, Question, Activity, Decision, Report

### Apply login required
def question(request):
		
	last_seen_sentence_idx = request.user.profile.last_seen_sentence_idx
	sentence = get_sentence(last_seen_sentence_idx + 1)

	text = sentence.text
	status = sentence.status
	clitic = sentence.clitic
	pos = sentence.pos

	half_text = get_half_text(text, pos, status)

	context = {
		'half_text': half_text,
		'clitic': clitic
	}

	if request.method == 'POST':

		form = ActivityForm(request.POST)

		if form.is_valid():
			
			question = Question(user=request.user,
						sentence=sentence)

			question.save()

			activity = form.save(commit=False)
			activity.question = question
			# date posted ?

			activity.save()

			decision = Decision(question=question,
								name = activity.name)

			decision.save()

			request.session['full_text'] = text

			if decision.name == status:
				request.session['answer'] = True
				request.user.profile.correct_answer_count += 1
			else:
				request.session['answer'] = False

			print("BEFORE: " + str(request.user.profile.last_seen_sentence_idx))
			request.user.profile.last_seen_sentence_idx += 1
			print("AFTER: " + str(request.user.profile.last_seen_sentence_idx))
			request.user.profile.save()

			return redirect('answer')
	else:
		form = ActivityForm()
	return render(request, 'game/question.html',context)
	

def answer(request):
	
	context = {

		'full_text': request.session.get('full_text') ,
		'answer': request.session.get('answer')

	}

	if context['answer']:
		messages.success(request, f'Doğru Cevap')
	else:
		messages.error(request, f'Yanlış Cevap')

	return render(request, 'game/answer.html', context)


	
def get_sentence(sentence_idx):

	sentence = Sentence.objects.all()[sentence_idx]
	return sentence


def get_half_text(full_text, pos, status):

	words = full_text.split()

	half_sentence = ""
	for idx, word in enumerate(words):

		if idx == pos:

			if status == 'ADJACENT':
				half_sentence = half_sentence + " " + word[:-2] # Exclude de/da/te/ta
				break
			else:
				break

		else:
			half_sentence = half_sentence + " " + word

	half_sentence = half_sentence[1:]

	return half_sentence

