from django.http import HttpResponseRedirect, HttpResponseNotAllowed
from django.shortcuts import render, redirect
from django.urls import reverse

from .forms import QueryForm, QueryFeedbackForm

import flair, torch
from flair.models import SequenceTagger
from flair.data import Sentence
from django.contrib.staticfiles import finders

from .models import Query, Feedback

from game.forms import ReportForm
from game.models import Report

from users.models import User

from game.views import str2bool

url = finders.find('best-model.pt')
flair.device = torch.device('cpu')
classifier = SequenceTagger.load_from_file(url)


def query(request):
    if request.method == 'GET':
        form = QueryForm()
    elif request.method == 'POST':
        form = QueryForm(request.POST)

        if form.is_valid():

            sentence = form.cleaned_data['sentence']

            labeled_words, is_error_found, tagged_sentence = invoke_spellchecker_engine(sentence)

            query = Query(sentence=sentence,
                          tagged_sentence=tagged_sentence)
            if request.user.is_authenticated:
                query.user = request.user
            else:
                query.user = None
            query.save()

            request.session['query_pk'] = query.pk

            context = {
                'labeled_words': labeled_words,
                'is_error_found': is_error_found,
                'form': form
            }

            # return redirect('spellchecker_answer')
            return render(request, 'spellchecker/spellchecker_query.html', context)
    else:
        form = QueryForm()

    context = {'form': form}
    return render(request, 'spellchecker/spellchecker_query.html', context)


def feedback(request):

    if request.method == 'POST':

        form = QueryFeedbackForm(request.POST)

        if form.is_valid():
            report = form.cleaned_data['report']
            query_pk = request.session.get('query_pk')
            query = Query.objects.get(pk=query_pk)
            feedback = Feedback(report=report,
                                query=query)
            feedback.save()

            return HttpResponseRedirect(reverse('spellchecker:query'))
        else:
            return HttpResponseRedirect(reverse('welcome'))
    else:
        return HttpResponseNotAllowed(permitted_methods=["POST"])


def invoke_spellchecker_engine(sentence_str):
    sentence = Sentence(sentence_str)

    classifier.predict(sentence)

    tagged_string = sentence.to_tagged_string()

    is_error_found, labeled_words = parse_tagged_string(tagged_string)

    return labeled_words, is_error_found, tagged_string


def parse_tagged_string(tagged_string):
    words, labels = [], []
    tokens = tagged_string.split()
    print(tagged_string)
    is_error_found = False
    idx = 0
    for token in tokens:

        print(str(idx) + ": " + token)

        if token == '<B-ERR>':

            is_error_found = True

            labels[idx - 1] = True

            # if de/da is seperate, highlight the word before the de/da as well
            if len(words[idx - 1]) == 2:
                labels[idx - 2] = True
                words[idx - 2] += " " + words[idx - 1]

                del words[idx - 1]
                del labels[idx - 1]
                idx -= 1

            idx -= 1

        else:
            words.append(token)
            labels.append(False)

        idx += 1
    labeled_words = [{'word': words[i], 'label': labels[i]} for i in range(len(words))]
    return is_error_found, labeled_words
