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

from enum import Enum

QUESTION_PER_TEST = 30

class MODE_THRESHOLD(Enum):
	MODE_1 = 8
	MODE_2 = 16
	MODE_3 = 24
	MODE_4 = QUESTION_PER_TEST

def home(request):

	request.session['question_num'] = 1
	request.session['go_next_question'] = True

	return render(request, 'game/home.html')

def about(request):

	return render(request, 'game/about.html')

@login_required
def question(request):

	last_seen_sentence_idx = request.user.profile.last_seen_sentence_idx
	sentence = get_sentence(last_seen_sentence_idx + 1)

	text = sentence.text
	status = sentence.status
	clitic = sentence.clitic
	pos = sentence.pos

	correct_answer_count = request.user.profile.correct_answer_count
	success_rate = round((correct_answer_count / (last_seen_sentence_idx)) * 100, 1)
	question_num = request.session['question_num']

	context = {
		'correct_answer_count': correct_answer_count,
		'last_seen_sentence_idx': last_seen_sentence_idx,
		'success_rate': success_rate,
		'question_num': question_num,
		'hints': []
	}
	
	if request.session['question_num'] < MODE_THRESHOLD.MODE_1.value:
		
		get_mode_1_context(context, text, pos, status)

	elif request.session['question_num'] < MODE_THRESHOLD.MODE_2.value:
		
		get_mode_2_context(context, text, pos, status)

	elif request.session['question_num'] < MODE_THRESHOLD.MODE_3.value:
		
		get_mode_3_context(context, text, pos, status)

	elif request.session['question_num'] < MODE_THRESHOLD.MODE_4.value:

		get_mode_4_context(context, text, pos, status)

	else:
		request.session['question_num'] = 1
		return render(request, 'game/test_end.html')

	
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

			question.mode = context['mode']
			if context['mode'] == 'MODE_4':
				question.relative_mask_pos = context['relative_mask_pos']
			question.save()

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


def get_first_half_text(full_text, pos, status):

	words = full_text.split()

	first_half_text = ""
	for idx, word in enumerate(words):

		if idx == pos:

			if status == 'ADJACENT':
				#first_half_text = first_half_text + " " + word[:-2] # Exclude de/da/te/ta
				break
			else:
				break

		else:

			# Do not take the preceding word
			if idx == pos - 1 and status == "SEPARATE":
				pass
			else: 
				first_half_text = first_half_text + " " + word

	first_half_text = first_half_text[1:]

	return first_half_text

def get_second_half_text(full_text, pos, status):

	words = full_text.split()

	second_half_text = ""
	for word in words[pos+1:]:
		second_half_text += " " + word

	second_half_text = second_half_text[1:]

	return second_half_text



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

def get_mode_1_context(context, text, pos, status):

	context['mode'] = 'MODE_1'
	context['first_half_text'] = get_first_half_text(text, pos, status)
	context['deda_separate'] = get_separate(text, pos, status)
	context['deda_adjacent'] = get_adjacent(text, pos, status)

def get_mode_2_context(context, text, pos, status):

	context['mode'] = 'MODE_2'
	context['second_half_text'] = get_second_half_text(text, pos, status)
	context['deda_separate'] = get_separate(text, pos, status)
	context['deda_adjacent'] = get_adjacent(text, pos, status)

def get_mode_3_context(context, text, pos, status):

	tokens = text.split()
	clitic = tokens[pos][-2:]

	context['mode'] = 'MODE_3'
	context['second_half_text'] = get_second_half_text(text, pos, status)
	context['clitic'] = clitic

def get_mode_4_context(context, text, pos, status):

	tokens = full_tokenize(text, pos, status)
		
	if status  == "ADJACENT":
		pos += 1

	mask_pos = random.randint(0, len(tokens)-1)
	# Do not mask the clitic
	while(mask_pos == pos):
		mask_pos = random.randint(0, len(tokens)-1)

	relative_mask_pos = mask_pos - pos
	context['relative_mask_pos'] = relative_mask_pos

	if mask_pos < pos:
		
		# first_text ____(de) second_text
		if mask_pos == pos-1:
			
			first_text = ""
			second_text = ""

			for token in tokens[:pos-1]:
				first_text += " " + token
			first_text = first_text[1:]

			for token in tokens[pos+1:]:
				second_text += " " + token
			second_text = second_text[1:]

			clitic = tokens[pos]

			context['type'] = 'TYPE_1'
			context['first_text'] = first_text
			context['second_text'] = second_text
			context['clitic'] = clitic



		# first_text ___ second_text (deda_adjacent and deda_separate) third_text
		else:

			first_text, second_text, third_text = "","",""
			
			for token in tokens[:mask_pos]:
				first_text += " " + token
			first_text = first_text[1:]

			for token in tokens[mask_pos+1:pos-1]:
				second_text += " " + token
			second_text = second_text[1:]

			for token in tokens[pos+1:]:
				third_text += " " + token
			third_text = third_text[1:]

			deda_separate = tokens[pos-1] + " " + tokens[pos]
			deda_adjacent = tokens[pos-1] + tokens[pos]

			context['type'] = 'TYPE_2'
			context['first_text'] = first_text
			context['second_text'] = second_text
			context['third_text'] = third_text
			context['deda_separate'] = deda_separate
			context['deda_adjacent'] = deda_adjacent


	# first_text (deda_adjacent and deda_separate) second_text ____ third_text
	else:
		# Calculate the necessary strings
		first_text, second_text, third_text = "","",""
		deda_separate = tokens[pos-1] + " " + tokens[pos]
		deda_adjacent = tokens[pos-1] + tokens[pos]

		for token in tokens[:pos-1]:
			first_text += " " + token
		first_text = first_text[1:]

		for token in tokens[pos+1:mask_pos]:
			second_text += " " + token
		second_text = second_text[1:]

		for token in tokens[mask_pos+1:]:
			third_text += " " + token
		third_text = third_text[1:]

		context['type'] = 'TYPE_3'
		context['first_text'] = first_text
		context['second_text'] = second_text
		context['third_text'] = third_text
		context['deda_separate'] = deda_separate
		context['deda_adjacent'] = deda_adjacent

	context['mode'] = 'MODE_4'
	return context

	


def full_tokenize(text, pos, status):

	words = text.split()

	if status == "SEPARATE":
		return words
	else:
		prev_word = words[pos][:-2]
		clitic = words[pos][-2:]

		words[pos] = prev_word

		words.insert(pos+1, clitic)

		return words
