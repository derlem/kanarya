from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import csv
import linecache
import random
from .forms import ActivityForm, ReportForm

from .models import Sentence, Question, Activity, Decision, Report

def home(request):

	request.session['question_num'] = 1
	request.session['go_next_question'] = True

	return render(request, 'game/home.html')

def about(request):

	return render(request, 'game/about.html')

@login_required
def question(request):

	if request.session['question_num'] > 20:
		request.session['question_num'] = 1
		return render(request, 'game/test_end.html')

	last_seen_sentence_idx = request.user.profile.last_seen_sentence_idx
	sentence = get_sentence(last_seen_sentence_idx + 1)

	text = sentence.text
	status = sentence.status
	clitic = sentence.clitic
	pos = sentence.pos

	half_text = get_half_text(text, pos, status)
	deda_separate = get_separate(text, pos, status)
	deda_adjacent = get_adjacent(text, pos, status)
	correct_answer_count = request.user.profile.correct_answer_count
	success_rate = round((correct_answer_count / (last_seen_sentence_idx)) * 100, 1)
	question_num = request.session['question_num']

	context = {
		'half_text': half_text,
		'clitic': clitic,
		'deda_separate': deda_separate,
		'deda_adjacent': deda_adjacent,
		'correct_answer_count': correct_answer_count,
		'last_seen_sentence_idx': last_seen_sentence_idx,
		'success_rate': success_rate,
		'question_num': question_num,
		'hints': []
	}
	
	if request.method == 'POST':

		form = ActivityForm(request.POST)

		if form.is_valid():
			
			# Create the question once for each sentence !
			if request.session['go_next_question']:
				question = Question(user=request.user,
						sentence=sentence)
				
				question.save()

			else:
				question_pk = request.session.get('question_pk')
				question = Question.objects.filter(pk=question_pk)[0]

			activity = form.save(commit=False)
			activity.question = question
			# date posted ?

			activity.save()

			request.session['full_text'] = text
			request.session['question_pk'] = question.pk

			if activity.name == "SKIP":

				request.session['go_next_question'] = True
				request.session['question_num'] += 1

				request.user.profile.last_seen_sentence_idx += 1
				request.user.profile.save()

				decision = Decision(question=question,
									name = activity.name)
				decision.save()

				return redirect('question')

			elif activity.name == "HINT":

				request.session['go_next_question'] = False

				set_hints(context['hints'], question.hint_count, text, pos)
				question.hint_count += 1
				question.save()

				return render(request, 'game/question.html',context)

			else: 

				request.session['question_num'] += 1

				if activity.name == status:

					request.session['answer'] = True
					request.user.profile.correct_answer_count += 1
					request.user.profile.save()

				else:
					request.session['answer'] = False


			request.session['correct_answer_count'] = correct_answer_count
			request.session['last_seen_sentence_idx'] = last_seen_sentence_idx
			request.session['success_rate'] = success_rate

			decision = Decision(question=question,
								name = activity.name)

			decision.save()
			request.session['go_next_question'] = True
			request.user.profile.last_seen_sentence_idx += 1
			request.user.profile.save()
			return redirect('answer')
	else:
		form = ActivityForm()
	return render(request, 'game/question.html',context)
	
	
@login_required
def answer(request):
	
	context = {

		'full_text': request.session.get('full_text') ,
		'answer': request.session.get('answer'),
		'correct_answer_count': request.session.get('correct_answer_count'),
		'last_seen_sentence_idx': request.session.get('last_seen_sentence_idx') ,
		'success_rate': request.session.get('success_rate') ,

	}

	if request.method == 'POST':

		form = ReportForm(request.POST)

		if form.is_valid():

			report = form.save(commit=False)
			question_pk = request.session.get('question_pk')
			question = Question.objects.filter(pk=question_pk)[0]
			report.question = question
			#report.question = current_question # ?
			report.save()

			return redirect('question')

	else:

		if context['answer']:
			messages.success(request, f'Doğru Cevap')
		else:
			messages.error(request, f'Yanlış Cevap')

		form = ReportForm()


	return render(request, 'game/answer.html', context)

def test_end(request):
	pass

def stats(request):

	decision_count = len(Decision.objects.all())

	skip_count = 0

	correct_answer_count = 0

	for decision in Decision.objects.all():

		status = decision.question.sentence.status
		answer = decision.name

		if answer == status:
			correct_answer_count += 1

		if answer == 'SKIP':
			skip_count += 1

	incorrect_answer_count = decision_count - correct_answer_count - skip_count

	success_rate = round((correct_answer_count / decision_count), 2)*100

	print(success_rate)

	context = {

		'decision_count': decision_count,
		'skip_count': skip_count,
		'correct_answer_count': correct_answer_count,
		'incorrect_answer_count': incorrect_answer_count,
		'success_rate': success_rate,

	}

	return render(request, 'game/stats.html', context)


# Returns 20 new sentences
def get_test(sentence_idx):

	test = []

	start_idx = sentence_idx
	end_idx = sentence_idx + 20

	for sentence in Sentence.objects.all()[start_idx, end_idx]:
		test.append(sentence)

	return test

	
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

def get_separate(text, pos, status):

	words = text.split()

	if status == "SEPARATE":
		return words[pos-1] + " " + words[pos]
	else:
		return words[pos][:-2] + " " +  words[pos][-2:]

def get_adjacent(text, pos, status):

	words = text.split()

	if status == "SEPARATE":
		return words[pos-1]  + words[pos]
	else:
		return words[pos]


def set_hints(hint_list, hint_count, text, pos):
	
	words = text.split()
	hint_start_idx = pos + 1
	hint_end_idx = hint_start_idx + hint_count + 1

	for word in words[hint_start_idx:hint_end_idx]:
		hint_list.append(word)

