files = ['de-da-te-ta.10E-4percent.conll.dev', 'de-da-te-ta.10E-4percent.conll.train', 'de-da-te-ta.10E-4percent.conll.test', 'de-da-te-ta.10E-4percent.conll.84max.dev', 'de-da-te-ta.10E-4percent.conll.84max.train', 'de-da-te-ta.10E-4percent.conll.84max.test']
corpus = ''
for file in files:
	with open(file, 'r') as f:
		corpus += '\n' + f.read()

tp = 0
tn = 0
fp = 0
fn = 0
corpus = corpus.split('\n\n')
for sentence in corpus:
	pairs = sentence.split('\n')
	for pair in pairs:
		if pair == '':
			continue
		word = pair.split()[0]
		tag = pair.split()[1]
		if word != 'de' and word != 'da' and (word[-2:] == 'de' or word[-2:] == 'da' or word[-2:] == 'te' or word[-2:] == 'ta'):
			if tag == 'B-ERR':
				tp += 1
			elif tag == 'O':
				fp += 1
		else:
			if tag == 'B-ERR':
				fn += 1 
			elif tag == 'O':
				tn += 1

print('true positive ' + str(tp))
print('true negative ' + str(tn))
print('false positive ' + str(fp))
print('false negative ' + str(fn))