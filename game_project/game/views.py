from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib import messages
import csv
import linecache
import random
import subprocess
from .forms import DecisionForm

from enum import Enum
class Status(Enum):
	ADJACENT = 0
	SEPARATE = 1

# Get rid of global variables, Find better ways
is_posted = False
is_first = True
current_sentence_info = {}

### Apply login required
def question(request):
	
	global is_posted

	context = get_sentence()
	#print('sentence_idx: ' + str(context['sentence_idx']) )
	if request.method == 'POST':

		is_posted = True

		form = DecisionForm(request.POST)

		if form.is_valid():
			decision = form.save(commit=False)
			decision.user = request.user
			decision.sentence_idx = context['sentence_idx']
			decision.save()

			status = context['status']

			request.session['full_sentence'] = context['full_sentence']

			# if the user answer is correct
			if decision.decision == status.name:
				request.session['answer'] = True
			else:
				request.session['answer'] = False

			return redirect('answer')
	else:
		form = DecisionForm()
	return render(request, 'game/question.html',context)

def answer(request):
	
	context = {

		'full_sentence': request.session.get('full_sentence') ,
		'answer': request.session.get('answer')

	}

	if context['answer']:
		messages.success(request, f'Doğru Cevap')
	else:
		messages.error(request, f'Yanlış Cevap')

	return render(request, 'game/answer.html', context)



# Return a random sentence from the dataset	
def get_sentence():

	global current_sentence_info
	global is_first
	global is_posted

	if not is_first and not is_posted:

		return current_sentence_info

	# ! Find a better way
	path_to_data = "/home/tony/Desktop/491/kanarya/game_project/game/static/game/sentences.csv"
	# Hard-coded: Find a generic way
	num_of_sentences = 38026
	line_num = random.randint(1,38026)
	line = linecache.getline(path_to_data, line_num)
	sentence_idx = list(csv.reader([line]))[0][0]
	sentence = list(csv.reader([line]))[0][1]
	pos = int(list(csv.reader([line]))[0][2])	



	sentence_info = {
		'sentence_idx': sentence_idx,
		'full_sentence': sentence,
		'pos': pos
	}

	status = get_status(sentence_info)

	sentence_info['status'] = status

	half_sentence = get_half_sentence(sentence_info)

	sentence_info['half_sentence'] = half_sentence

	clitic = get_clitic(sentence_info)

	sentence_info['clitic'] = clitic


	# Find a better way
	current_sentence_info = sentence_info
	is_first = False
	is_posted = False

	return sentence_info

def get_clitic(sentence_info):

	full_sentence = sentence_info['full_sentence']
	pos = sentence_info['pos']

	words = full_sentence.split()

	word_deda = words[pos]

	return word_deda[-2:]

def get_half_sentence(sentence_info):

	full_sentence = sentence_info['full_sentence']
	pos = sentence_info['pos']
	status = sentence_info['status']

	words = full_sentence.split()

	half_sentence = ""
	for idx, word in enumerate(words):

		if idx == pos:

			if status == Status.ADJACENT:
				half_sentence = half_sentence + " " + word[:-2] # Exclude de/da/te/ta
				break
			else:
				break

		else:
			half_sentence = half_sentence + " " + word

	half_sentence = half_sentence[1:]

	return half_sentence



		



# Returns if the deda is separate or adjacent
def get_status(sentence_info):

	full_sentence = sentence_info['full_sentence']
	pos = sentence_info['pos']
	
	words = full_sentence.split()

	# if separate
	if len(words[pos]) == 2:
		return Status.SEPARATE
	else:
		return Status.ADJACENT