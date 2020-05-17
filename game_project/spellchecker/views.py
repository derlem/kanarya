# Memory Monitor
import os
import psutil
process = psutil.Process(os.getpid())

print("Beginning: " + str(process.memory_info().rss/(1024*1024)) + " MB")

from django.shortcuts import render, redirect
from .forms import QueryForm
from .models import Query

import flair, torch
from flair.models import SequenceTagger
from flair.data import Sentence
from django.contrib.staticfiles import finders


print("After Imports: " + str(process.memory_info().rss/(1024*1024)) + " MB")

url = finders.find('best-model.pt')
flair.device = torch.device('cpu')
classifier = SequenceTagger.load_from_file(url)

print("After Loading Spellchecker: " + str(process.memory_info().rss/(1024*1024)) + " MB")

def query(request):

    if request.method == 'POST':

        form = QueryForm(request.POST)

        if form.is_valid():


            sentence = form.cleaned_data['sentence']

            request.session['sentence'] = sentence

            
            return redirect('spellchecker_answer')


    else:

        form = QueryForm()


    return render(request, 'spellchecker/spellchecker_query.html')

def answer(request):
	
	sentence = request.session.get('sentence')

	labeled_words = spellchecker(sentence)

	
	print("\nAfter Using Spellchecker: " + str(process.memory_info().rss/(1024*1024)) + " MB\n")

	#query = Query()


	context = {
	'labeled_words': labeled_words
	}

	return render(request, 'spellchecker/spellchecker_answer.html', context)
	



def spellchecker(sentence):

	sentence = Sentence(sentence)

	classifier.predict(sentence)

	tagged_string = sentence.to_tagged_string()
	
	labeled_words = {}

	words, labels = [],[] 

	tokens = tagged_string.split()

	print(tagged_string)


	idx = 0
	for token in tokens:

		print(str(idx) + ": " + token)

		if token == '<B-ERR>':
			labels[idx-1] = True


			# if de/da is seperate, highlight the word before the de/da as well 
			if len(words[idx-1]) == 2:

				labels[idx - 2] = True
				words[idx - 2] += " " + words[idx-1] 

				del words[idx-1]
				del labels[idx-1]
				idx -= 1

				

			idx -= 1
			
		else:
			words.append(token)
			labels.append(False)

		idx += 1

	labeled_words = [{'word':words[i],'label':labels[i]} for i in range(len(words))]

	return labeled_words



	