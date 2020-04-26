from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
import csv
import linecache
import random
from .forms import ActivityForm, ReportForm

from .models import Sentence, Question, Activity, Decision, Report

go_next = True
current_question = Question()

def home(request):

	return render(request, 'game/home.html')

def about(request):

	return render(request, 'game/about.html')

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
		'clitic': clitic,
		'hints': []
	}

	global go_next 
	global current_question
	
	if request.method == 'POST':

		form = ActivityForm(request.POST)

		if form.is_valid():
			
			# Create the question once for each sentence !
			if go_next:
				question = Question(user=request.user,
						sentence=sentence)

				question.save()
			else:
				#question = request.session['question']
				question = current_question

			activity = form.save(commit=False)
			activity.question = question
			# date posted ?

			activity.save()

			request.session['full_text'] = text
			current_question = question

			if activity.name == "SKIP":

				request.user.profile.last_seen_sentence_idx += 1
				request.user.profile.save()
				go_next = True

				decision = Decision(question=question,
								name = activity.name)

				decision.save()

				return redirect('question')

			elif activity.name == "HINT":

				set_hints(context['hints'], question.hint_count, text, pos)
				question.hint_count += 1

				#request.session['question'] = question
				#request.session['activity'] = activity
				#current_question = question

				go_next = False

				return render(request, 'game/question.html',context)

			elif activity.name == status:

				request.session['answer'] = True
				request.user.profile.correct_answer_count += 1

			else:
				request.session['answer'] = False


			decision = Decision(question=question,
								name = activity.name)

			decision.save()
			go_next = True
			request.user.profile.last_seen_sentence_idx += 1
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

	if request.method == 'POST':

		form = ReportForm(request.POST)

		if form.is_valid():

			report = form.save(commit=False)
			report.question = current_question # ?
			report.save()

			return redirect('question')

	else:

		if context['answer']:
			messages.success(request, f'Doğru Cevap')
		else:
			messages.error(request, f'Yanlış Cevap')

		form = ReportForm()


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

def set_hints(hint_list, hint_count, text, pos):
	
	words = text.split()
	hint_start_idx = pos + 1
	hint_end_idx = hint_start_idx + hint_count + 1

	for word in words[hint_start_idx:hint_end_idx]:
		hint_list.append(word)

