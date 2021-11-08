from fairseq.models.roberta import RobertaModel
import os
from collections import defaultdict
from colorama import Fore, Back, Style
import torch
import numpy as np
import sys
import getopt


def int2hex(s):
    return s
    # return hex(int(s))[2:]


def print_line():
    print("---------------------------------------------------------------------")


def value_to_char(val):
    if (val == 0):
        return '-'
    elif (val == 2):
        return 'S'
    elif (val == 1):
        return 'E'
    else:
        sys.exit("Error in value_to_char")


def SE2FR(i):
    if (i == '-'):
        return '-'
    elif (i == 'S'):
        return 'F'
    elif (i == 'E'):
        return 'R'
    else:
        sys.exit("Error in SE2FR()")

def pause():
    hint = 'Press <Enter> to continue; Press q to quit\n'
    get_key = input(hint)
    while get_key != '' and get_key != 'q':
        get_key = input(hint)
    if (get_key == 'q'):
        sys.exit("Exit: Press q")
    else:
        return


def abstract_label_from_line(line):
    line_split = line.strip().split()
    return line_split[1]


def ida_f1(dirname, filename, start_idx=0, end_idx=-1):
    TP = FP = FN = TN = 0
    f_ida = open(f'{dirname}/ida_labeled_code/{filename}', 'r')
    f_truth = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
    idas = [abstract_label_from_line(line_ida) for line_ida in f_ida]
    truths = [abstract_label_from_line(line_truth) for line_truth in f_truth]
    if (end_idx != -1):
        idas = idas[start_idx:end_idx]
        truths = truths[start_idx:end_idx]
    for (ida, truth) in zip(idas, truths):
        if ida == truth and (truth == 'F' or truth == 'R'):
            TP += 1
        elif ida == truth == '-':
            TN += 1
        elif ida in ['F', 'R'] and truth == '-':
            FP += 1
        elif ida == '-' and truth in ['F', 'R']:
            FN += 1
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    total_tokens = TP + TN + FP + FN
    print(f'IDA --- {filename} (total {total_tokens}): TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, precision: {precision}, recall: {recall}, F1: {F1}')

    f_ida.close()
    f_truth.close()


def xda_f1(dirname, filename, labels):
    TP = FP = FN = TN = 0
    f_truth = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
    truths = [abstract_label_from_line(line_truth) for line_truth in f_truth]
    xdas = [SE2FR(label) for label in labels]
    for xda, truth in zip(xdas, truths):
        if xda == truth and (truth == 'F' or truth == 'R'):
            TP += 1
        elif xda == truth == '-':
            TN += 1
        elif xda in ['F', 'R'] and  truth == '-':
            FP += 1
        elif xda == '-' and truth in ['F', 'R']:
            FN += 1
        # else:
        #     sys.exit("Error happens in xda_f1")
    precision = (1.0 * TP) / (TP + FP)
    recall = (1.0 * TP) / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    total_tokens = TP + TN + FP + FN
    print(f'XDA --- {filename} (total {total_tokens}): TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, precision: {precision}, recall: {recall}, F1: {F1}')

    f_truth.close()


def xda_prediction(model, tokens, start_idx=0, end_idx=-1):
    if (end_idx == -1):
        end_idx = len(tokens) + start_idx
    end_idx = min(len(tokens), end_idx)
    token_chunks = [tokens[i:min(i+512, end_idx)] for i in range(start_idx, end_idx, 512)]
    labels = []
    for chunk_tokens in token_chunks:
        # print (f'Processing chunk of size {len(chunk_tokens)}')
        encoded_tokens = model.encode(' '.join(chunk_tokens))
        sub_logprobs = model.predict('funcbound', encoded_tokens)
        sub_labels = sub_logprobs.argmax(dim=2).view(-1).data
        labels = np.concatenate((labels, sub_labels))
    labels = [value_to_char(l) for l in labels]
    return labels



def get_token_list(dirname, filename, start_idx=0, end_idx=-1):
    tokens = []
    f_truth = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
    for i, line_truth in enumerate(f_truth):
        if i < start_idx:
            continue

        line_truth_split = line_truth.strip().split()
        # Prepare the tokens for prediction by XDA
        tokens.append(int2hex(line_truth_split[0]).lower())

        if end_idx != -1 and i > end_idx:
            break

    f_truth.close()
    return tokens
   

def get_truth_labels(dirname, filename, start_idx=0, end_idx=-1):
    labels = []
    f_truth = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
    for i, line_truth in enumerate(f_truth):
        if i < start_idx:
            continue

        line_truth_split = line_truth.strip().split()
        label = line_truth_split[1]
        if label == '-':
            labels.append('-')
        elif label == 'F':
            labels.append('S')
        elif label == 'R':
            labels.append('E')
        else:
            sys.exit("Error happens in get_truth_labels")

        if end_idx != -1 and i > end_idx:
            break

    f_truth.close()
    return labels

def get_ida_labels(dirname, filename, start_idx=0, end_idx=-1):
    labels = []
    f_ida = open(f'{dirname}/ida_labeled_code/{filename}', 'r')
    for i, line_ida in enumerate(f_ida):
        if i < start_idx:
            continue

        line_ida_split = line_ida.strip().split()
        label = line_ida_split[1]
        if label == '-':
            labels.append('-')
        elif label == 'F':
            labels.append('S')
        elif label == 'R':
            labels.append('E')
        else:
            sys.exit('Error in get_ida_labels')

        if end_idx != -1 and i > end_idx:
            break

    f_ida.close()
    return labels


def print_labels(labels):
    for i, label in enumerate(labels):
        if label == '-':
            print('-', end=' ')
        elif label == 'S':
            print(f'{Fore.RED}{"S"}{Fore.RESET}', end=' ')
        elif label == 'E':
            print(f'{Fore.GREEN}{"E"}{Fore.RESET}', end=' ')
        else:
            sys.exit('Error in print_labels')
    print()

def print_tokens(tokens, labels=None):
    if labels:
        for (token, label) in zip(tokoens, labels):
            if label == '-':
                print(f'{token}', end=' ')
            elif label == 'S':
                print(f'{Fore.RED}{token}{Fore.RESET}', end=' ')
            elif label == 'E':
                print(f'{Fore.GREEN}{token}{Fore.RESET}', end=' ')
            else:
                sys.exit("Error in print_tokens")
    else:
        for token in tokens:
            print(f'{token}', end=' ')
  

def main():
    max_index = -1
    debug_mode = False
    print_labels = False

    try:
        opts, args = getopt.getopt(sys.argv[1:], '', ['debug-mode', 'print-labels', 'max-index='])
    except:
        print("No args")

    for opt, arg in opts:
        if opt in ['--max-index']:
            max_index = int(arg)
        elif opt in ['--debug-mode']:
            debug_mode = True
        elif opt in ['--print-labels']:
            print_labels = True
 
    if debug_mode:
        print("DEBUG MODE ON")
    else:
        print("DEBUG MODE OFF")

    if print_labels:
        print("PRINT LABEL ON")
    else:
        print("PRINT LABEL OFF")

    if max_index != -1:
        print(f'max_index is set to {max_index}')
    else:
        print('max_index is not specified')

   
    # Load our model
    roberta = \
    RobertaModel.from_pretrained('checkpoints/finetune_msvs_funcbound_64', \
                                 'checkpoint_best.pt', \
                                 'data-bin/funcbound_msvs_64', \
                                 bpe=None, \
                                 user_dir='finetune_tasks')
    roberta.eval()
    
    dirname = 'data-raw/msvs_funcbound_64_bap_test'
    
    files_path = 'data-raw/msvs_funcbound_64_bap_test/truth_labeled_code'
    files = [f for f in os.listdir(files_path)]
    
    for filename in files:
        print('Playing with file: {}'.format(filename))
        tokens = get_token_list(dirname, filename, 0, max_index)
        print(f'The number of tokens is {len(tokens)}')
        xda_prediction_labels = xda_prediction(roberta, tokens, 0, max_index)
        truth_labels = get_truth_labels(dirname, filename, 0, max_index)
        IDA_labels = get_ida_labels(dirname, filename, 0, max_index)

        if (print_labels):
            print("Truth Labels:")
            print_labels(truth_labels)
            print("XDA Labels:")
            print_labels(xda_prediction_labels)
            print("IDA Labels:")
            print_labels(IDA_labels)

        ida_f1(dirname, filename, 0, max_index)
        xda_f1(dirname, filename, xda_prediction_labels)

        if debug_mode:
            pause()

    print('---END---')


if __name__ == "__main__":
    main()
