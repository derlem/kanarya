from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User
#from .models import Decision
import csv
import linecache
import random
import subprocess
from .forms import DecisionForm

### Apply login required

def home(request):
	
	context = get_sentence()

	if request.method == 'POST':
		form = DecisionForm(request.POST)

		if form.is_valid():
			decision = form.save(commit=False)
			decision.user = request.user
			decision.sentence_idx = context['sentence_idx']
			decision.save()

			print(decision.user)
			print(decision.sentence_idx)
			print(decision.decision)
			#val = form.cleaned_data.get("btn")
			#decision = form.cleaned_data.get("decision")
			#context['decision'] = decision

			# Redirect
	else:
		form = DecisionForm()
	return render(request, 'game/home.html',context)

# Return a random sentence from the dataset	
def get_sentence():
	# ! Find a better way
	path_to_data = "/home/tony/Desktop/491/kanarya/game_project/game/static/game/sentences.csv"
	# Hard-coded: Find a generic way
	num_of_sentences = 38026
	line_num = random.randint(1,38026)
	line = linecache.getline(path_to_data, line_num)
	sentence_idx = list(csv.reader([line]))[0][0]
	sentence = list(csv.reader([line]))[0][1]

	sentence_info = {
		'sentence_idx': sentence_idx,
		'sentence': sentence
	}

	return sentence_info
	