import os

files = os.listdir('.')
files.sort()
counter = 1
for file in files:
	os.rename(os.path.basename(file), "chunk_" + str(counter) + ".txt")
	counter = counter + 1