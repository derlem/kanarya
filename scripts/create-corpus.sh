#!/usr/bin/env bash

corpus_filepath=${1:-../../data/turkish-texts-tokenized.txt.gz}
# output_filepath=${2:-window-5.txt}
dataset_label=${2:-window-5}
deda=${3:-de}

echo extracting ${deda} word windows from ${corpus_filepath}
zless $corpus_filepath | awk ' { where = 1; str = $0; while (where != 0) { where = match(str, /(\S+ \S+ (\S+)) (de) (\S+ \S+ \S+)/, ary); if (where != 0) { print ary[1], ary[3], ary[4]; print ary[1] ary[3], ary[4]; print ary[2], ary[3]; concatenated_word = ary[2] ary[3]; print concatenated_word; print ""; }; str = substr(str, RSTART+RLENGTH); } }' > ${dataset_label}-correct-${deda}-word-02.txt

echo concatenating
cat ${dataset_label}-correct-${deda}-word-02.txt | awk ' NR % 5 == 4 { print $1; print ""; }' > ${dataset_label}-correct-${deda}-word-02-concatenated.txt

echo generating morphological analyses
less ${dataset_label}-correct-${deda}-word-02-concatenated.txt | awk ' !/^$/ { print } ' | while read word; do  output=`./scripts/morph-analyze.sh $word | wc -l`; if (( $output == 2 )); then echo 1 $word; else echo 0 $word; fi ; echo ; done > ${dataset_label}-correct-${deda}-word-02-concatenated-output-from-morph-analyzer.txt

echo combining all
awk ' BEGIN { count = 0 } !/^$/ { if (ARGIND == 1) { analyses[count] = $0; count += 1; } else { print } } BEGINFILE { if (ARGIND == 2) { count = 0 } } /^$/ { if (ARGIND == 2) { print analyses[count]; print ""; count += 1; } } ' ${dataset_label}-correct-${deda}-word-02-concatenated-output-from-morph-analyzer.txt ${dataset_label}-correct-${deda}-word-02.txt > ${dataset_label}-correct-${deda}-word-02-combined.txt

echo ratio of samples which cannot be disambiguated by using morphological analyzer (confirmed some with Google Docs spellchecker)
less ${dataset_label}-correct-${deda}-word-02-combined.txt | awk ' NR % 6 == 5 { if ($1 == 0) { count += 1 } } /^$/ { record_count += 1 } END { print count/record_count, count, record_count }'

echo generating CoNLL format for positive and negative samples
less ${dataset_label}-correct-${deda}-word-02-combined.txt | awk ' NR % 6 == 1 { for (i = 1; i <= NF; i++) { print $i, "O"; } ; print ""; } NR % 6 == 2 { for (i = 1; i <= NF; i++) { if (i == 3) { label = "B-ERR"; } else { label = "O" }; print $i, label; } print ""; }' > ${dataset_label}-correct-${deda}-word-02-conll.txt

