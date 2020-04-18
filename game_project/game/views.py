from django.shortcuts import render
from django.http import HttpResponse
from django.contrib.auth.models import User
from .models import Decision
import csv
import linecache
import random
import subprocess

def home(request):

	sentence = get_sentence()
	context = {

		'sentence': sentence
	}

	return render(request, 'game/home.html', context)

# Return a random sentence from the dataset	
def get_sentence():
	# ! Find a better way
	path_to_data = "/home/tony/Desktop/491/kanarya/game_project/game/static/game/sentences.csv"
	# Hard-coded: Find a generic way
	num_of_sentences = 38026
	line_num = random.randint(1,38026)
	line = linecache.getline(path_to_data, line_num)
	sentence = list(csv.reader([line]))[0][1]

	return sentence
	