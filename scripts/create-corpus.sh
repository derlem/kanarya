#!/usr/bin/env bash

corpus_filepath=${1:-../../data/turkish-texts-tokenized.txt.gz}
# output_filepath=${2:-window-5.txt}
dataset_label=${2:-window-5}
#deda=${3:-de}

perform_morph_analysis=0

for deda in de da te ta ; do

	echo extracting ${deda} word windows from ${corpus_filepath}
	# zless $corpus_filepath | pv -l -s 22M -p | python scripts/sentence_tokenizer.py | head -10000 | awk ' { str = $0; where = match(str, /^(.*\S+ (\S+)) ('${deda}') (.+)$/, ary); if (where != 0) { print ary[1], ary[3], ary[4]; print ary[1] ary[3], ary[4]; print ary[2], ary[3]; concatenated_word = ary[2] ary[3]; print concatenated_word; print ""; }; }' > ${dataset_label}-correct-${deda}-word-02.txt
	# zless $corpus_filepath | pv -l -s 22M -p | python scripts/sentence_tokenizer.py | head -10000 | awk ' /^(.*\S+ (\S+)) ('${deda}') (.+)$/ { left_part = ""; for (i = 1; i <=NF; i++ ) { left_part = left_part $i; if ($i == "'${deda}'") { }  } print ary[1], ary[3], ary[4]; print ary[1] ary[3], ary[4]; print ary[2], ary[3]; concatenated_word = ary[2] ary[3]; print concatenated_word; print ""; }; }' > ${dataset_label}-correct-${deda}-word-02.txt
	zless $corpus_filepath | pv -l -s 22M -p | python scripts/sentence_tokenizer.py | python scripts/conll_sample_creator.py ${deda} word > ${dataset_label}-${deda}-word-03-conll.txt

	echo extracting ${deda} suffix windows from ${corpus_filepath}
	zless $corpus_filepath | pv -l -s 22M -p | python scripts/sentence_tokenizer.py | python scripts/conll_sample_creator.py ${deda} suffix > ${dataset_label}-${deda}-suffix-03-conll.txt

	if (( $perform_morph_analysis == 1 )) ; then
		echo concatenating
		cat ${dataset_label}-correct-${deda}-word-02.txt | awk ' NR % 5 == 4 { print $1; print ""; }' > ${dataset_label}-correct-${deda}-word-02-concatenated.txt

		echo generating morphological analyses
		less ${dataset_label}-correct-${deda}-word-02-concatenated.txt | awk ' !/^$/ { print } ' | while read word; do  output=`./scripts/morph-analyze.sh $word | wc -l`; if (( $output == 2 )); then echo 1 $word; else echo 0 $word; fi ; echo ; done > ${dataset_label}-correct-${deda}-word-02-concatenated-output-from-morph-analyzer.txt

		echo combining all
		awk ' BEGIN { count = 0 } !/^$/ { if (ARGIND == 1) { analyses[count] = $0; count += 1; } else { print } } BEGINFILE { if (ARGIND == 2) { count = 0 } } /^$/ { if (ARGIND == 2) { print analyses[count]; print ""; count += 1; } } ' ${dataset_label}-correct-${deda}-word-02-concatenated-output-from-morph-analyzer.txt ${dataset_label}-correct-${deda}-word-02.txt > ${dataset_label}-correct-${deda}-word-02-combined.txt

		echo ratio of samples which cannot be disambiguated by using morphological analyzer \(confirmed some with Google Docs spellchecker\)
		less ${dataset_label}-correct-${deda}-word-02-combined.txt | awk ' NR % 6 == 5 { if ($1 == 0) { count += 1 } } /^$/ { record_count += 1 } END { print count/record_count, count, record_count }' ;
	fi

	for type in word-03 suffix-03 ; do
#		if [ "${type}" == "suffix-02" ] ; then
#			nelements=5
#			position=4
#		else
#			if (( $perform_morph_analysis == 1 )) ; then
#				nelements=6
#				position=3
#			else
#				nelements=5
#				position=3
#				# type=word-02
#			fi
#		fi
#		echo generating CoNLL format for positive and negative ${type} samples
#		less ${dataset_label}-correct-${deda}-${type}.txt | pv -l -s `wc -l ${dataset_label}-correct-${deda}-${type}.txt` -p | awk ' NR % '${nelements}' == 1 { for (i = 1; i <= NF; i++) { print $i, "O"; } ; print ""; } NR % '${nelements}' == 2 { for (i = 1; i <= NF; i++) { if (i == '${position}') { label = "B-ERR"; } else { label = "O" }; print $i, label; } print ""; }' > ${dataset_label}-${deda}-${type}-conll.txt
		echo seperating the dataset into train and test parts \(${type}\)
		less ${dataset_label}-${deda}-${type}-conll.txt | pv -l -s `wc -l ${dataset_label}-${deda}-${type}-conll.txt` -p | awk 'BEGIN { srand(); sample = ""; second = 0; } !/^$/ { sample = sample "\n" $0; } /^$/ { if (second == 0) { sample = sample "\n"; second = 1 } else if (second == 1) { if (rand() < 0.1 ) { print sample > "'${dataset_label}'-correct-'${deda}'-'${type}'-conll.txt.test" } else if (rand() < 0.2 ) { print sample > "'${dataset_label}'-correct-'${deda}'-'${type}'-conll.txt.dev" } else { print sample > "'${dataset_label}'-correct-'${deda}'-'${type}'-conll.txt.train" }; sample = ""; second = 0; } }'
	done

done

# TODO: get rid of sed at the end
for dataset in train dev test ; do
	echo removing 1st line of ${dataset} dataset ;
	cat ${dataset_label}-*-word-03-*.${dataset} ${dataset_label}-*-suffix-03-*.${dataset} > de-da-te-ta.conll.${dataset} ;
	sed -i '1d' de-da-te-ta.conll.${dataset} ;
done

for dataset in train dev test ; do
	echo ${dataset};
	less de-da-te-ta.conll.${dataset} | awk 'BEGIN { srand(); sample = ""; second = 0; } !/^$/ { sample = sample "\n" $0; } /^$/ { if (second == 0) { sample = sample "\n"; second = 1 } else if (second == 1) { if (rand() < 0.0001 ) { print sample > "de-da-te-ta.10E-4percent.conll.'${dataset}'" } sample = ""; second = 0; } }' ;
done
