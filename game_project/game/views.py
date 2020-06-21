from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth.models import User
from django.contrib.auth.decorators import login_required
from django.contrib import messages
import csv
import linecache
import random
#import distutils
import os
from .forms import ActivityForm, ReportForm#, ProfForm

from .models import Sentence, Question, Activity, Decision, Report

from enum import Enum

NUM_OF_PROF_QUESTIONS = 5

SENTENCE_SEEN_LIMIT = 2

### Warmup parameters

WARMUP_QUESTION_NUM_MODE_0 = 1
WARMUP_QUESTION_NUM_MODE_1 = 1
WARMUP_QUESTION_NUM_MODE_2 = 1
WARMUP_QUESTION_NUM_MODE_3 = 1
WARMUP_QUESTION_NUM_MODE_4 = 1
WARMUP_QUESTION_NUM_MODE_5 = 1
WARMUP_QUESTION_NUM_MODE_6 = 1

WARMUP_QUESTION_PER_TEST =(WARMUP_QUESTION_NUM_MODE_0 +
                    WARMUP_QUESTION_NUM_MODE_1 +
                    WARMUP_QUESTION_NUM_MODE_2 + 
                    WARMUP_QUESTION_NUM_MODE_3 +
                    WARMUP_QUESTION_NUM_MODE_4 + 
                    WARMUP_QUESTION_NUM_MODE_5 +
                    WARMUP_QUESTION_NUM_MODE_6)

class WARMUP_MODE_THRESHOLD(Enum):
    MODE_0 = WARMUP_QUESTION_NUM_MODE_0
    MODE_1 = WARMUP_QUESTION_NUM_MODE_0 + WARMUP_QUESTION_NUM_MODE_1
    MODE_2 = WARMUP_QUESTION_NUM_MODE_0 + WARMUP_QUESTION_NUM_MODE_1 + WARMUP_QUESTION_NUM_MODE_2
    MODE_3 = WARMUP_QUESTION_NUM_MODE_0 + WARMUP_QUESTION_NUM_MODE_1 + WARMUP_QUESTION_NUM_MODE_2 + WARMUP_QUESTION_NUM_MODE_3
    MODE_4 = WARMUP_QUESTION_NUM_MODE_0 + WARMUP_QUESTION_NUM_MODE_1 + WARMUP_QUESTION_NUM_MODE_2 + WARMUP_QUESTION_NUM_MODE_3 + WARMUP_QUESTION_NUM_MODE_4
    MODE_5 = WARMUP_QUESTION_NUM_MODE_0 + WARMUP_QUESTION_NUM_MODE_1 + WARMUP_QUESTION_NUM_MODE_2 + WARMUP_QUESTION_NUM_MODE_3 + WARMUP_QUESTION_NUM_MODE_4 + WARMUP_QUESTION_NUM_MODE_5
    MODE_6 = WARMUP_QUESTION_NUM_MODE_0 + WARMUP_QUESTION_NUM_MODE_1 + WARMUP_QUESTION_NUM_MODE_2 + WARMUP_QUESTION_NUM_MODE_3 + WARMUP_QUESTION_NUM_MODE_4 + WARMUP_QUESTION_NUM_MODE_5 + WARMUP_QUESTION_NUM_MODE_6


### Main experiment parameters

QUESTION_NUM_MODE_0 = 5
QUESTION_NUM_MODE_1 = 5
QUESTION_NUM_MODE_2 = 5
QUESTION_NUM_MODE_3 = 5
QUESTION_NUM_MODE_4 = 5
QUESTION_NUM_MODE_5 = 5
QUESTION_NUM_MODE_6 = 5

QUESTION_PER_TEST =(QUESTION_NUM_MODE_0 +
                    QUESTION_NUM_MODE_1 +
                    QUESTION_NUM_MODE_2 + 
                    QUESTION_NUM_MODE_3 +
                    QUESTION_NUM_MODE_4 + 
                    QUESTION_NUM_MODE_5 +
                    QUESTION_NUM_MODE_6)

mode_labels = {
    'MODE_0': 'SEVİYE 0',
    'MODE_1': 'SEVİYE 1',
    'MODE_2': 'SEVİYE 2',
    'MODE_3': 'SEVİYE 3',
    'MODE_4': 'SEVİYE 4',
    'MODE_5': 'SEVİYE 5',
    'MODE_6': 'SEVİYE 6',
}

mode_descriptions = {
    'MODE_0': 'Bu cümlede "de/da" eki ya da bağlacı bitişik mi yoksa ayrı mı yazılmalıdır?',
    'MODE_1': 'Bir kelimesi "_____" işaretiyle gizlenmiş olan bu cümlede "de/da" eki ya da bağlacı bitişik mi yoksa ayrı mı yazılmalıdır?',
    'MODE_2': '"De/da" eki ya da bağlacından sonraki tüm kelimelerin * * * işareti ile silindiği bu cümlede "de/da" eki ya da bağlacı bitişik mi yoksa ayrı mı yazılmalıdır?',
    'MODE_3': '"De/da" ek ya da bağlacı ile ilgili kelimeden önceki tüm kelimelerin * * * işareti ile silindiği bu cümlede "de/da" eki ya da bağlacı bitişik mi yoksa ayrı mı yazılmalıdır?',
    'MODE_4': '"De/da" ek ya da bağlacından önceki tüm kelimelerin * * * işareti ile silindiği bu cümlede "de/da" eki ya da bağlacı bitişik mi yoksa ayrı mı yazılmalıdır?',
    'MODE_5': '"De/da" ek ya da bağlacından bir önceki kelimeye ek olarak sonraki tüm kelimelerin * * * işareti ile silindiği bu cümlede "de/da" eki ya da bağlacı bitişik mi yoksa ayrı mı yazılmalıdır?',
    'MODE_6': '"De/da" ek ya da bağlacının dışındaki kelimelerden sadece biri hariç tümünün "_____" işareti ile silindiği bu cümlede "de/da" eki ya da bağlacı bitişik mi yoksa ayrı mı yazılmalıdır?',
}

class MODE_THRESHOLD(Enum):
    MODE_0 = QUESTION_NUM_MODE_0
    MODE_1 = QUESTION_NUM_MODE_0 + QUESTION_NUM_MODE_1
    MODE_2 = QUESTION_NUM_MODE_0 + QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2
    MODE_3 = QUESTION_NUM_MODE_0 + QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3
    MODE_4 = QUESTION_NUM_MODE_0 + QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3 + QUESTION_NUM_MODE_4
    MODE_5 = QUESTION_NUM_MODE_0 + QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3 + QUESTION_NUM_MODE_4 + QUESTION_NUM_MODE_5
    MODE_6 = QUESTION_NUM_MODE_0 + QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3 + QUESTION_NUM_MODE_4 + QUESTION_NUM_MODE_5 + QUESTION_NUM_MODE_6

def welcome(request):

    return render(request, 'game/welcome.html')

@login_required
def home(request):

    # Set new test settings
    request.session['question_idx'] = 1
    request.session['correct_answer_count_for_current_test'] = 0
    request.session['solved_question_num'] = 0
    request.session['success_rate'] = 0

    request.session['go_next_question'] = True

    context = {

        "is_warmup_done": request.user.profile.is_warmup_done

    }

    return render(request, 'game/home.html', context)

def about(request):

    return render(request, 'game/about.html')

@login_required
def question(request):

    # Current question index according the user progress
    last_seen_sentence_idx = request.user.profile.last_seen_sentence_idx

    
    # Get the current question 
    #sentence = get_sentence(last_seen_sentence_idx + 1)

    

    sentence = ""
    if request.session['question_idx'] <= MODE_THRESHOLD.MODE_0.value:
        next_sentence_idx = get_next_sentence_idx(last_seen_sentence_idx, 'MODE_0')
        sentence = get_sentence(next_sentence_idx, 'MODE_0')
    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_1.value:
        next_sentence_idx = get_next_sentence_idx(last_seen_sentence_idx, 'MODE_1')
        sentence = get_sentence(next_sentence_idx, 'MODE_1')
    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_2.value:
        next_sentence_idx = get_next_sentence_idx(last_seen_sentence_idx, 'MODE_2')
        sentence = get_sentence(next_sentence_idx, 'MODE_2')
    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_3.value:
        next_sentence_idx = get_next_sentence_idx(last_seen_sentence_idx, 'MODE_3')
        sentence = get_sentence(next_sentence_idx, 'MODE_3')
    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_4.value:
        next_sentence_idx = get_next_sentence_idx(last_seen_sentence_idx, 'MODE_4')
        sentence = get_sentence(next_sentence_idx, 'MODE_4')
    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_5.value:
        next_sentence_idx = get_next_sentence_idx(last_seen_sentence_idx, 'MODE_5')
        sentence = get_sentence(next_sentence_idx, 'MODE_5')
    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_6.value:
        next_sentence_idx = get_next_sentence_idx(last_seen_sentence_idx, 'MODE_6')
        sentence = get_sentence(next_sentence_idx, 'MODE_6')
    else:
        # Set test start settings
        request.session['question_idx'] = 1
        request.session['correct_answer_count_for_current_test'] = 0
        request.session['solved_question_num'] = 0
        request.session['success_rate'] = 0

        return redirect('test_end')

    full_text = sentence.text
    status = sentence.status
    clitic = sentence.clitic
    pos = sentence.pos

    # Get Progress Info
    question_idx = request.session['question_idx']
    correct_answer_count_for_current_test = request.session.get('correct_answer_count_for_current_test')
    solved_question_num = request.session.get('solved_question_num')
    success_rate = request.session.get('success_rate')
    

    # Set initial context
    context = {
        'question_idx': question_idx,
        'correct_answer_count_for_current_test': request.session.get('correct_answer_count_for_current_test'),
        'solved_question_num': solved_question_num,
        'success_rate': success_rate,
        'QUESTION_PER_TEST': QUESTION_PER_TEST
    }
    
    # Set modal context


    if request.session['question_idx'] <= MODE_THRESHOLD.MODE_0.value:
        
        get_mode_0_context(context, full_text, pos, status)

    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_1.value:
        
        get_mode_1_context(context, full_text, pos, status)

    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_2.value:
        
        get_mode_2_context(context, full_text, pos, status)

    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_3.value:
        
        get_mode_3_context(context, full_text, pos, status)

    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_4.value:

        get_mode_4_context(context, full_text, pos, status)

    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_5.value:

        get_mode_5_context(context, full_text, pos, status)

    elif request.session['question_idx'] <= MODE_THRESHOLD.MODE_6.value:

        get_mode_6_context(context, full_text, pos, status)

    else:
        # Set test start settings
        request.session['question_idx'] = 1
        request.session['correct_answer_count_for_current_test'] = 0
        request.session['solved_question_num'] = 0
        request.session['success_rate'] = 0

        return redirect('test_end')

    context['mode_label'] = mode_labels[context['mode']]
    context['mode_description'] = mode_descriptions[context['mode']]

    # Process the POST request
    if request.method == 'POST':

        form = ActivityForm(request.POST)

        if form.is_valid():
            
            # Create the question once for each sentence !
            if request.session['go_next_question']:
                question = Question(user=request.user,
                        sentence=sentence)
                
                question.save()


            # This else statement seems redundant. Check!
            else:
                question_pk = request.session.get('question_pk')
                question = Question.objects.filter(pk=question_pk)[0]


            question.mode = context['mode']
            if context['mode'] == 'MODE_1':
                question.relative_mask_pos = request.session.get('relative_mask_pos')

            if context['mode'] == 'MODE_6':
                question.relative_unmask_pos = request.session.get('relative_unmask_pos')
            question.save()

            activity = form.save(commit=False)
            activity.question = question
            # date posted ?

            activity.save()

            # Pass the answer to the context
            request.session['selected_button'] = activity.name
            request.session['full_text'] = full_text
            request.session['question_pk'] = question.pk
            request.session['solved_question_num'] += 1
            request.session['activity'] = activity.name
            request.session['sentence_pk'] = sentence.pk
            


            if activity.name == "SKIP":

                request.session['go_next_question'] = True # necessary?
                request.session['question_idx'] += 1
                request.session['success_rate'] = round((request.session.get('correct_answer_count_for_current_test') / request.session.get('solved_question_num')) * 100, 1)

                request.user.profile.last_seen_sentence_idx += 1
                request.user.profile.save()

                decision = Decision(question=question,
                                    name = activity.name)
                decision.save()

                return redirect('question')

            elif activity.name == "INDECISIVE":

                
                request.session['go_next_question'] = True
                request.session['success_rate'] = round((request.session.get('correct_answer_count_for_current_test') / request.session.get('solved_question_num')) * 100, 1)
                request.session['question_idx'] += 1
                request.session['success_rate'] = round((request.session.get('correct_answer_count_for_current_test') / request.session.get('solved_question_num')) * 100, 1)
                
                # Increase the seen count of the answered sentences
                print("Decision Count: " + str(sentence.decision_count) + " + " + str(1) + " = " + str(sentence.decision_count + 1))
                sentence.decision_count += 1
                sentence.save()

            else: 

                # Increase the seen count of the answered sentences
                print("Decision Count: " + str(sentence.decision_count) + " + " + str(1) + " = " + str(sentence.decision_count + 1))
                sentence.decision_count += 1
                sentence.save()

                request.session['question_idx'] += 1

                if activity.name == status:

                    request.session['answer'] = True

                    request.session['correct_answer_count_for_current_test'] += 1
                    request.session['success_rate'] = round((request.session.get('correct_answer_count_for_current_test') / request.session.get('solved_question_num')) * 100, 1)

                    request.user.profile.correct_answer_count += 1
                    request.user.profile.save()

                else:
                    request.session['answer'] = False
                    request.session['success_rate'] = round((request.session.get('correct_answer_count_for_current_test') / request.session.get('solved_question_num')) * 100, 1)


            
            decision = Decision(question=question,
                                name = activity.name)

            decision.save()
            request.session['go_next_question'] = True
            request.user.profile.last_seen_sentence_idx += 1
            request.user.profile.save()
            return redirect('answer')
    else:

        # Pass the context to answer page
        request.session['context'] = context
        #request.session['last_seen_sentence_idx'] = last_seen_sentence_idx

        if context['mode'] == 'MODE_1':
            request.session['relative_mask_pos'] = context['relative_mask_pos']

        if context['mode'] == 'MODE_6':
            request.session['relative_unmask_pos'] = context['relative_unmask_pos']

        form = ActivityForm()

    return render(request, 'game/question.html',context)
    
    
@login_required
def answer(request):
    
    # Get the question context
    context = request.session.get('context')

    #last_seen_sentence_idx = request.session.get('last_seen_sentence_idx')
    #sentence = get_sentence(last_seen_sentence_idx + 1)

    #sentence = get_sentence(context['mode'])
    
    sentence_pk = request.session.get('sentence_pk')
    sentence = Sentence.objects.filter(pk=sentence_pk)[0]


    full_text = sentence.text
    status = sentence.status
    clitic = sentence.clitic
    pos = sentence.pos

    first_answer_text, highlighted_answer_text, second_answer_text = get_answer_text(full_text, pos, status)

    context['first_answer_text'] = first_answer_text
    context['highlighted_answer_text'] = highlighted_answer_text
    context['second_answer_text'] = second_answer_text
    context['answer'] = request.session.get('answer')
    context['correct_answer_count_for_current_test'] = request.session.get('correct_answer_count_for_current_test') # necessary?
    context['solved_question_num'] = request.session.get('solved_question_num')
    context['success_rate'] = request.session.get('success_rate')
    context['selected_button'] = request.session.get('selected_button')


    if request.method == 'POST':

        form = ReportForm(request.POST)

        if form.is_valid():

            report = form.save(commit=False)
            question_pk = request.session.get('question_pk')
            question = Question.objects.filter(pk=question_pk)[0]
            report.question = question
            report.save()

            return redirect('question')

    else:

        form = ReportForm()


    return render(request, 'game/answer.html', context)

@login_required
def test_end(request):
    
    messages.success(request, f'Test bitti, çok teşekkürler!')

    return render(request, 'game/test_end.html')

@login_required
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

    context = {

        'decision_count': decision_count,
        'skip_count': skip_count,
        'correct_answer_count': correct_answer_count,
        'incorrect_answer_count': incorrect_answer_count,
        'success_rate': success_rate,

    }

    return render(request, 'game/stats.html', context)

@login_required
def sentence_counts(request):

    sentences = Sentence.objects.all().order_by('-decision_count')

    context = {
        'sentences': sentences
    }

    return render(request, 'game/sentence_counts.html', context)


def get_warmup_question_context(warmup_question_index):

    path_to_data = os.path.dirname(__file__) + '/static/game/WarmupSentences.csv'

    f = open(path_to_data, 'r')

    all_lines = f.readlines()

    current_row = all_lines[warmup_question_index -1]

    index = list(csv.reader([current_row]))[0][0]
    text = list(csv.reader([current_row]))[0][1]
    status = list(csv.reader([current_row]))[0][2]
    clitic = list(csv.reader([current_row]))[0][3]
    pos = int(list(csv.reader([current_row]))[0][4])
    

    return text, status, clitic, pos

def warmup_question(request):    

    warmup_question_index = request.user.profile.last_seen_warmup_idx

    full_text, status, clitic, pos = get_warmup_question_context(warmup_question_index)

    # Set initial context
    context = {
        'question_idx': warmup_question_index,
        'QUESTION_PER_TEST': WARMUP_QUESTION_PER_TEST,
    }
    
    # Set modal context

    if warmup_question_index <= WARMUP_MODE_THRESHOLD.MODE_0.value:
        
        get_mode_0_context(context, full_text, pos, status)

    elif warmup_question_index <= WARMUP_MODE_THRESHOLD.MODE_1.value:
        
        get_mode_1_context(context, full_text, pos, status)

    elif warmup_question_index <= WARMUP_MODE_THRESHOLD.MODE_2.value:
        
        get_mode_2_context(context, full_text, pos, status)

    elif warmup_question_index <= WARMUP_MODE_THRESHOLD.MODE_3.value:
        
        get_mode_3_context(context, full_text, pos, status)

    elif warmup_question_index <= WARMUP_MODE_THRESHOLD.MODE_4.value:

        get_mode_4_context(context, full_text, pos, status)

    elif warmup_question_index <= WARMUP_MODE_THRESHOLD.MODE_5.value:

        get_mode_5_context(context, full_text, pos, status)

    elif warmup_question_index <= WARMUP_MODE_THRESHOLD.MODE_6.value:

        get_mode_6_context(context, full_text, pos, status)

    else:

        request.user.profile.is_warmup_done = True
        request.user.profile.save()

        return redirect('warmup_end')

    context['mode_label'] = mode_labels[context['mode']]
    context['mode_description'] = mode_descriptions[context['mode']]

    # Process the POST request
    if request.method == 'POST':

        form = ActivityForm(request.POST)

        if form.is_valid():
            

            activity = form.save(commit=False)
            #activity.save()

            request.session['selected_button'] = activity.name          

            if activity.name == "SKIP":

                request.user.profile.last_seen_warmup_idx += 1
                request.user.profile.save()

                return redirect('warmup_question')

            elif activity.name == "INDECISIVE":
                pass

            elif activity.name == "ADJACENT" or activity.name == "SEPARATE": 

                if activity.name == status:

                    request.session['answer'] = True


                else:
                    request.session['answer'] = False

            request.user.profile.last_seen_warmup_idx += 1
            request.user.profile.save()
            return redirect('warmup_answer')
    else:

        # Pass the context to answer page
        request.session['context'] = context

        form = ActivityForm()

    return render(request, 'game/warmup_question.html',context)
    


def warmup_answer(request):

    # Get the question context
    context = request.session.get('context')

    warmup_question_index = request.user.profile.last_seen_warmup_idx - 1

    full_text, status, clitic, pos = get_warmup_question_context(warmup_question_index)

    first_answer_text, highlighted_answer_text, second_answer_text = get_answer_text(full_text, pos, status)

    context['first_answer_text'] = first_answer_text
    context['highlighted_answer_text'] = highlighted_answer_text
    context['second_answer_text'] = second_answer_text
    context['answer'] = request.session.get('answer') 
    context['selected_button'] = request.session.get('selected_button') 


    if request.method == 'POST':

        form = ReportForm(request.POST)

        if form.is_valid():

            return redirect('warmup_question')

    else:

        form = ReportForm()


    return render(request, 'game/warmup_answer.html', context)

@login_required
def warmup_end(request):
    messages.success(request, f'Isınma turumuz sona erdi, çok teşekkürler!')

    return render(request, 'game/warmup_end.html')
    
def get_sentence(next_sentence_idx, mode):

    sentence = Sentence.objects.all().filter(mode=mode, decision_count__lt=SENTENCE_SEEN_LIMIT)[next_sentence_idx]
    #sentence = Sentence.objects.all()[sentence_idx]
    print("Sentence " + str(sentence.index) + " : Decision Count: "  + str(sentence.decision_count))
    
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



def get_separate(full_text, pos, status):

    words = full_text.split()

    if status == "SEPARATE":
        return words[pos-1] + " " + words[pos]
    else:
        return words[pos][:-2] + " " +  words[pos][-2:]

def get_adjacent(full_text, pos, status):

    words = full_text.split()

    if status == "SEPARATE":
        return words[pos-1]  + words[pos]
    else:
        return words[pos]


def get_mode_0_context(context, full_text, pos, status):

    tokens = full_tokenize(full_text, pos, status)

    if status  == "ADJACENT":
        pos += 1

    first_text, second_text = "",""

    for token in tokens[:pos-1]:
        first_text += " " + token
    first_text = first_text[1:]

    for token in tokens[pos+1:]:
        second_text += " " + token
    second_text = second_text[1:]

    deda_separate = tokens[pos-1] + " " + tokens[pos]
    deda_adjacent = tokens[pos-1] + tokens[pos]

    context['first_text'] = first_text
    context['second_text'] = second_text
    context['deda_separate'] = deda_separate
    context['deda_adjacent'] = deda_adjacent


    context['mode'] = 'MODE_0'
    return context


def get_mode_1_context(context, full_text, pos, status):

    tokens = full_tokenize(full_text, pos, status)
        
    if status  == "ADJACENT":
        pos += 1

    mask_pos = random.randint(0, len(tokens)-1)
    # Do not mask the clitic
    while(mask_pos == pos):
        mask_pos = random.randint(0, len(tokens)-1)

    relative_mask_pos = mask_pos - pos
    context['relative_mask_pos'] = relative_mask_pos
    #print("mask_pos: " + str(mask_pos))
    #print("pos: " + str(pos))
    #print("relative_mask_pos: " + str(relative_mask_pos))

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

    context['mode'] = 'MODE_1'
    return context

def get_mode_2_context(context, full_text, pos, status):

    context['mode'] = 'MODE_2'
    context['first_half_text'] = get_first_half_text(full_text, pos, status)
    context['deda_separate'] = get_separate(full_text, pos, status)
    context['deda_adjacent'] = get_adjacent(full_text, pos, status)

def get_mode_3_context(context, full_text, pos, status):

    context['mode'] = 'MODE_3'
    context['second_half_text'] = get_second_half_text(full_text, pos, status)
    context['deda_separate'] = get_separate(full_text, pos, status)
    context['deda_adjacent'] = get_adjacent(full_text, pos, status)

def get_mode_4_context(context, full_text, pos, status):

    tokens = full_text.split()
    clitic = tokens[pos][-2:]

    context['mode'] = 'MODE_4'
    context['second_half_text'] = get_second_half_text(full_text, pos, status)
    context['clitic'] = clitic


def get_mode_5_context(context, full_text, pos, status):

    tokens = full_tokenize(full_text, pos, status)

    if status  == "ADJACENT":
        pos += 1

    clitic = tokens[pos][-2:]

    first_text = ""

    for token in tokens[:pos-1]:
        first_text += " " + token
    first_text = first_text[1:]

    context['mode'] = 'MODE_5'
    context['clitic'] = clitic
    context['first_text'] = first_text
    
def get_mode_6_context(context, full_text, pos, status):

    tokens = full_tokenize(full_text, pos, status)

    if status  == "ADJACENT":
        pos += 1

    unmask_pos = random.randint(0, len(tokens)-1)
    # Do not unmask the clitic, it will be visible in any case
    while(unmask_pos == pos):
        unmask_pos = random.randint(0, len(tokens)-1)

    relative_unmask_pos = unmask_pos - pos
    context['relative_unmask_pos'] = relative_unmask_pos
    context['mode'] = 'MODE_6'

    if unmask_pos < pos:
        
        if unmask_pos == pos-1:

            context['type'] = 'TYPE_1'
            context['first_masked_num'] = unmask_pos
            context['deda_adjacent'] = tokens[pos - 1] + tokens[pos]
            context['deda_separate'] = tokens[pos - 1] + " " + tokens[pos]
            context['second_masked_num'] = len(tokens) - pos - 1

            #print(context['type'])
            #print("first_masked_num: " + str(context['first_masked_num']))
            #print("deda_adjacent: " + context['deda_adjacent'])
            #print("deda_separate: " + context['deda_separate'])
            #print("second_masked_num: " + str(context['second_masked_num']))

        else:

            context['type'] = 'TYPE_2'
            context['first_masked_num'] = unmask_pos 
            context['unmasked_word'] = tokens[unmask_pos]
            context['second_masked_num'] = pos - unmask_pos - 2
            context['clitic'] = tokens[pos]
            context['third_masked_num'] = len(tokens) - pos - 1

            #print(context['type'])
            #print("first_masked_num: " + str(context['first_masked_num']))
            #print("unmasked_word: " + context['unmasked_word'])
            #print("second_masked_num: " + str(context['second_masked_num']))
            #print("clitic: " + context['clitic'])
            #print("third_masked_num: " + str(context['third_masked_num']))

    else:

        context['type'] = 'TYPE_3'
        context['first_masked_num'] = pos - 1
        context['clitic'] = tokens[pos]
        context['second_masked_num'] = unmask_pos - pos - 1
        context['unmasked_word'] = tokens[unmask_pos]
        context['third_masked_num'] = len(tokens) - unmask_pos - 1

        #print(context['type'])
        #print("first_masked_num: " + str(context['first_masked_num']))
        #print("clitic: " + context['clitic'])
        #print("second_masked_num: " + str(context['second_masked_num']))
        #print("unmasked_word: " + context['unmasked_word'])        
        #print("third_masked_num: " + str(context['third_masked_num']))

    return context




# Treat the clitic as a separate token even if status is adjacent
def full_tokenize(full_text, pos, status):

    words = full_text.split()

    if status == "SEPARATE":
        return words
    else:
        prev_word = words[pos][:-2]
        clitic = words[pos][-2:]

        words[pos] = prev_word

        words.insert(pos+1, clitic)

        return words



def get_answer_text(full_text, pos, status):

    tokens = full_tokenize(full_text, pos, status)

    # Update the pos
    if status  == "ADJACENT":
        pos += 1

    # Construct highlighted text
    if status == "ADJACENT":
        highlighted_text = tokens[pos - 1] + tokens[pos]
    else:
        highlighted_text = tokens[pos - 1] + " " + tokens[pos]

    # Construct first and second text
    first_text, second_text = "", ""
    for idx, token in enumerate(tokens):

        if idx < (pos - 1):

            first_text += " " + token

        if idx > (pos):

            second_text += " " + token

    first_text = first_text[1:]
    second_text = second_text[1:]

    return first_text, highlighted_text, second_text


def str2bool(s):

    return True if s == "true" or s == "True" else False

# Returns the sentence idx for the current mode.
# If the user solved QUESTION_PER_TEST + 
def get_next_sentence_idx(last_seen_sentence_idx, mode):

    current_question_idx_for_curr_test = last_seen_sentence_idx % QUESTION_PER_TEST
    current_mode_idx = current_question_idx_for_curr_test % 5
    num_of_tests_solved = last_seen_sentence_idx // QUESTION_PER_TEST
    num_of_questions_solved_for_curr_mode = 5 * num_of_tests_solved + current_mode_idx

    num_of_overlimit_sentences_for_curr_mode = len(Sentence.objects.all().filter(mode=mode, decision_count__gte=SENTENCE_SEEN_LIMIT))
    #next_question_idx = num_of_questions_solved_for_curr_mode - num_of_overlimit_sentences_for_curr_mode

    if num_of_questions_solved_for_curr_mode >= num_of_overlimit_sentences_for_curr_mode:
        next_question_idx = num_of_questions_solved_for_curr_mode - num_of_overlimit_sentences_for_curr_mode
    else:
        next_question_idx = num_of_questions_solved_for_curr_mode

    print("---")
    print("Mode: " + mode)
    #print("current_question_idx_for_curr_test : " + str(current_question_idx_for_curr_test))
    print("current_mode_idx : " + str(current_mode_idx))
    print("num_of_tests_solved : " + str(num_of_tests_solved))
    print("num_of_questions_solved_for_curr_mode : " + str(num_of_questions_solved_for_curr_mode))
    print("num_of_overlimit_sentences_for_curr_mode : " + str(num_of_overlimit_sentences_for_curr_mode))
    print("next_question_idx : " + str(next_question_idx))
    print("---")

    return next_question_idx