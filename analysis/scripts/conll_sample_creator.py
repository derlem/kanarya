import sys


def check_for_target(token, target_type):
    assert target_type in ["word", "suffix"]
    if target_type == "word":
        return token == target_deda
    elif target_type == "suffix":
        return token[-2:] == target_deda

def create_conll_line_pair(previous_token, token, target_type, target_deda):
    assert target_type in ["word", "suffix"]

    if target_type == "word":
        return [[[token, "O"]], [[previous_token+token, "B-ERR"]]]
    elif target_type == "suffix":
        if token == target_deda:
            return [[[token, "O"]], [[token[-2:], "B-ERR"]]]
        else:
            return [[[token, "O"]], [[token[:-2], "O"], [token[-2:], "B-ERR"]]]

def print_conll_record(conll_record):
    for conll_line in conll_record:
        print " ".join(conll_line)
    print ""

sys.stderr.write("extracting ${deda} windows\n".format(deda=sys.argv[1:]))

for line in sys.stdin:

    for target_deda in [sys.argv[1]]:
        for target_type in [sys.argv[2]]:
            tokens = line.rstrip().split(" ")
            conll_records = [[], []]
            matched = False
            for token_idx, token in enumerate(tokens):
                # if we find our spot, generate two possible pairs
                if check_for_target(token, target_type):
                    conll_line_pairs = create_conll_line_pair(tokens[token_idx-1] if token_idx > 0 else "", token, target_type, target_deda)
                    # new_conll_records = []
                    # for conll_record in conll_records:
                    #     new_conll_records.append(conll_record + conll_line_pairs[0])
                    #     new_conll_records.append(conll_record + conll_line_pairs[1])
                    # we generate only two samples from each sentence for easier processing. omit the above code.
                    conll_records[0] += conll_line_pairs[0]
                    if target_type == "word":
                        conll_records[1] = conll_records[1][:-1]
                    conll_records[1] += conll_line_pairs[1]
                    matched = True
                else:
                    conll_records[0] += [[token, "O"]]
                    conll_records[1] += [[token, "O"]]

            if matched:
                for conll_record in conll_records:
                    print_conll_record(conll_record)
