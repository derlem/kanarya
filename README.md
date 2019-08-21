# Detecting Clitics Related Orthographic Errors in Turkish 


For the spell correction task,  vocabulary
based  methods  have  been  replaced  with
methods that take morphological and gram-
mar  rules  into  account.   However,  such
tools are fairly immature, and, worse, non-
existent for many low resource languages.
Checking only if a word is well-formed
with respect to the morphological rules of a
language may produce false negatives due
to the ambiguity resulting from the pres-
ence of numerous homophonic words. In
this work, we propose an approach to de-
tect  and  correct  the  “de/da”  clitic  errors
in Turkish text.  Our model is a neural se-
quence tagger trained with a synthetically
constructed dataset consisting of positive
and negative samples. The model’s perfor-
mance with this dataset is presented accord-
ing to different word embedding configu-
rations.  The model achieved an F
1
score
of 86.67% on a synthetically constructed
dataset. We also compared the model’s per-
formance on a manually curated dataset of
challenging samples that proved superior
to other spelling correctors with 71% accu-
racy compared to the second best (Google
Docs) with 34% accuracy.

Please cite the following if you make use of this work:

Ugurcan Arikan, Onur Gungor, and Suzan Uskudarli. "Detecting Clitics Related Orthographic Errors in Turkish". Recent Advances in Natural Language Processing 2019, Sept. 2019, Varna Bulgaria
