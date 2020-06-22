from django.shortcuts import render, redirect
from .forms import QueryForm, QueryFeedbackForm
from .models import Query

import flair, torch
from flair.models import SequenceTagger
from flair.data import Sentence
from django.contrib.staticfiles import finders

from .models import Query

from game.views import str2bool

url = finders.find('best-model.pt')
flair.device = torch.device('cpu')
classifier = SequenceTagger.load_from_file(url)


def query(request):

    if request.method == 'POST':
        form = QueryForm(request.POST)
    elif request.method == "GET":
        form = QueryForm(request.GET)
    else:
        form = QueryForm()

    if form.is_valid():
        sentence = form.cleaned_data['sentence']

        # request.session['sentence'] = sentence

        query = Query(sentence=sentence)

        if request.user.is_authenticated:
            query.user = request.user

        query.save()

        labeled_words = spellchecker(sentence)

        context = {
            'labeled_words': labeled_words,
            'sentence': sentence
        }

        request.session['query_pk'] = query.pk

        return render(request, 'spellchecker/spellchecker_query.html', context)
    else:
        return render(request, 'spellchecker/spellchecker_query.html')


def spellchecker(sentence):
    sentence = Sentence(sentence)

    classifier.predict(sentence)

    tagged_string = sentence.to_tagged_string()

    labeled_words = {}

    words, labels = [], []

    tokens = tagged_string.split()

    print(tagged_string)

    idx = 0
    for token in tokens:

        print(str(idx) + ": " + token)

        if token == '<B-ERR>':
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

    return labeled_words
