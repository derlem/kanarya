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

QUESTION_NUM_MODE_1 = 5
QUESTION_NUM_MODE_2 = 5
QUESTION_NUM_MODE_3 = 5
QUESTION_NUM_MODE_4 = 5
QUESTION_NUM_MODE_5 = 5
QUESTION_NUM_MODE_6 = 5

QUESTION_PER_TEST =(QUESTION_NUM_MODE_1 +
                    QUESTION_NUM_MODE_2 + 
                    QUESTION_NUM_MODE_3 +
                    QUESTION_NUM_MODE_4 + 
                    QUESTION_NUM_MODE_5 +
                    QUESTION_NUM_MODE_6)

mode_labels = {
    'MODE_1': 'SEVİYE 1',
    'MODE_2': 'SEVİYE 2',
    'MODE_3': 'SEVİYE 3',
    'MODE_4': 'SEVİYE 4',
    'MODE_5': 'SEVİYE 5',
    'MODE_6': 'SEVİYE 6',
}

mode_descriptions = {
    'MODE_1': 'Bu seviyede cümledeki bir kelime rastgele silinir. Sizin elinizdeki kelimelerle "de/da" ek ya da bağlacının doğru yazımını tahmin etmeniz gerekir.',
    'MODE_2': 'Bu seviyede cümledeki "de/da" ek ya da bağlacından sonraki tüm kelimeler silinir. Sizin elinizdeki kelimelerle "de/da" ek ya da bağlacının doğru yazımını tahmin etmeniz gerekir.',
    'MODE_3': 'Bu seviyede cümledeki "de/da" ek ya da bağlacından önceki tüm kelimeler (bir önceki kelime dışında) silinir. Sizin elinizdeki kelimelerle "de/da" ek ya da bağlacının doğru yazımını tahmin etmeniz gerekir.',
    'MODE_4': 'Bu seviyede cümledeki "de/da" ek ya da bağlacından önceki tüm kelimeler silinir. Sizin elinizdeki kelimelerle "de/da" ek ya da bağlacının doğru yazımını tahmin etmeniz gerekir.',
    'MODE_5': 'Bu seviyede cümledeki "de/da" ek ya da bağlacından bir önceki kelime ve sonraki tüm kelimeler silinir. Sizin elinizdeki kelimelerle "de/da" ek ya da bağlacının doğru yazımını tahmin etmeniz gerekir.',
    'MODE_6': 'Bu seviyede cümledeki bir kelime dışındaki tüm sözcükler silinir. Sizin elinizdeki kelime ile "de/da" ek ya da bağlacının doğru yazımını tahmin etmeniz gerekir.',
}

class MODE_THRESHOLD(Enum):
    MODE_1 = QUESTION_NUM_MODE_1
    MODE_2 = QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2
    MODE_3 = QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3
    MODE_4 = QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3 + QUESTION_NUM_MODE_4
    MODE_5 = QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3 + QUESTION_NUM_MODE_4 + QUESTION_NUM_MODE_5
    MODE_6 = QUESTION_NUM_MODE_1 + QUESTION_NUM_MODE_2 + QUESTION_NUM_MODE_3 + QUESTION_NUM_MODE_4 + QUESTION_NUM_MODE_5 + QUESTION_NUM_MODE_6

def home(request):

    # Set new test settings
    request.session['question_idx'] = 1
    request.session['correct_answer_count_for_current_test'] = 0
    request.session['solved_question_num'] = 0
    request.session['success_rate'] = 0

    request.session['go_next_question'] = True

    return render(request, 'game/home.html')

def about(request):

    return render(request, 'game/about.html')

@login_required
def question(request):

    # Current question index according the user progress
    last_seen_sentence_idx = request.user.profile.last_seen_sentence_idx

    # Get the current question 
    sentence = get_sentence(last_seen_sentence_idx + 1)
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
        'hints': [],
        'correct_answer_count_for_current_test': request.session.get('correct_answer_count_for_current_test'),
        'solved_question_num': solved_question_num,
        'success_rate': success_rate,
        'QUESTION_PER_TEST': QUESTION_PER_TEST
    }
    
    # Set modal context

    if request.session['question_idx'] <= MODE_THRESHOLD.MODE_1.value:
        
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

        return render(request, 'game/test_end.html')

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

            else:
                question_pk = request.session.get('question_pk')
                question = Question.objects.filter(pk=question_pk)[0]

            question.mode = context['mode']
            if context['mode'] == 'MODE_1':
                question.relative_mask_pos = context['relative_mask_pos']

            if context['mode'] == 'MODE_6':
                question.relative_unmask_pos = context['relative_unmask_pos']
            question.save()



            activity = form.save(commit=False)
            activity.question = question
            # date posted ?

            activity.save()

            request.session['full_text'] = full_text
            request.session['question_pk'] = question.pk

            request.session['solved_question_num'] += 1
            
            

            if activity.name == "SKIP":

                request.session['go_next_question'] = True
                request.session['question_idx'] += 1
                request.session['success_rate'] = round((request.session.get('correct_answer_count_for_current_test') / request.session.get('solved_question_num')) * 100, 1)

                request.user.profile.last_seen_sentence_idx += 1
                request.user.profile.save()

                decision = Decision(question=question,
                                    name = activity.name)
                decision.save()

                return redirect('question')

            elif activity.name == "HINT":

                request.session['go_next_question'] = False
                request.session['success_rate'] = round((request.session.get('correct_answer_count_for_current_test') / request.session.get('solved_question_num')) * 100, 1)

                set_hints(context['hints'], question.hint_count, full_text, pos)
                question.hint_count += 1
                question.save()

                return render(request, 'game/question.html',context)

            else: 

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
        form = ActivityForm()
    return render(request, 'game/question.html',context)
    
    
@login_required
def answer(request):
    
    last_seen_sentence_idx = request.user.profile.last_seen_sentence_idx
    sentence = get_sentence(last_seen_sentence_idx)

    full_text = sentence.text
    status = sentence.status
    clitic = sentence.clitic
    pos = sentence.pos

    first_text, highlighted_text, second_text = get_answer_text(full_text, pos, status)
    
    context = {

        'first_text' : first_text ,
        'highlighted_text' : highlighted_text ,
        'second_text' : second_text ,
        'answer': request.session.get('answer'),
        'correct_answer_count_for_current_test': request.session.get('correct_answer_count_for_current_test'),
        'solved_question_num': request.session.get('solved_question_num'),
        'success_rate': request.session.get('success_rate'),
    }

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


def set_hints(hint_list, hint_count, full_text, pos):
    
    words = full_text.split()
    hint_start_idx = pos + 1
    hint_end_idx = hint_start_idx + hint_count + 1

    for word in words[hint_start_idx:hint_end_idx]:
        hint_list.append(word)

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

    # Do not forget to the record the relative_unmask_pos

    print(full_text)

    if unmask_pos < pos:
        
        if unmask_pos == pos-1:

            context['type'] = 'TYPE_1'
            context['first_masked_num'] = unmask_pos
            context['deda_adjacent'] = tokens[pos - 1] + tokens[pos]
            context['deda_separate'] = tokens[pos - 1] + " " + tokens[pos]
            context['second_masked_num'] = len(tokens) - pos - 1

            print(context['type'])
            print("first_masked_num: " + str(context['first_masked_num']))
            print("deda_adjacent: " + context['deda_adjacent'])
            print("deda_separate: " + context['deda_separate'])
            print("second_masked_num: " + str(context['second_masked_num']))

        else:

            context['type'] = 'TYPE_2'
            context['first_masked_num'] = unmask_pos 
            context['unmasked_word'] = tokens[unmask_pos]
            context['second_masked_num'] = pos - unmask_pos - 2
            context['clitic'] = tokens[pos]
            context['third_masked_num'] = len(tokens) - pos - 1

            print(context['type'])
            print("first_masked_num: " + str(context['first_masked_num']))
            print("unmasked_word: " + context['unmasked_word'])
            print("second_masked_num: " + str(context['second_masked_num']))
            print("clitic: " + context['clitic'])
            print("third_masked_num: " + str(context['third_masked_num']))

    else:

        context['type'] = 'TYPE_3'
        context['first_masked_num'] = pos - 1
        context['clitic'] = tokens[pos]
        context['second_masked_num'] = unmask_pos - pos - 1
        context['unmasked_word'] = tokens[unmask_pos]
        context['third_masked_num'] = len(tokens) - unmask_pos - 1

        print(context['type'])
        print("first_masked_num: " + str(context['first_masked_num']))
        print("clitic: " + context['clitic'])
        print("second_masked_num: " + str(context['second_masked_num']))
        print("unmasked_word: " + context['unmasked_word'])        
        print("third_masked_num: " + str(context['third_masked_num']))

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





