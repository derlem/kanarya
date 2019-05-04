import sys

old_file = sys.argv[1]
new_file = sys.argv[2]

new_vocab = ''
with open(old_file, 'r') as f:
    for line in f:
        line = line.split()[0]
        line = line.replace(chr(9601), '##')
        line = line + '\n'
        new_vocab = new_vocab + line

f2 = open(new_file, 'w')
f2.write(new_vocab)
f2.close()
