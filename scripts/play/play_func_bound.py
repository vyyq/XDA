from fairseq.models.roberta import RobertaModel
import os
from collections import defaultdict
from colorama import Fore, Back, Style
import torch


def int2hex(s):
    return s
    # return hex(int(s))[2:]


def print_line():
    print("---------------------------------------------------------------------")


def value_to_char(val):
    if (val == 0):
        return '-'
    elif (val == 2):
        return 'F'
    elif (val == 1):
        return 'R'
    else:
        exit()

def pause():
    hint = 'Press <Enter> to continue; Press q to quit\n'
    get_key = input(hint)
    while get_key != '' and get_key != 'q':
        get_key = input(hint)
    if (get_key == 'q'):
        exit()
    else:
        return

def print_labels(labels):
    def identity(s):
        return s
    for i, label in enumerate(labels):
        if label == '-':
            print('-', end=" ")
        elif label == 'F':
            print(f'{Fore.RED}{"F"}{Fore.RESET}', end=" ")
        elif label == 'R':
            print(f'{Fore.GREEN}{"R"}{Fore.RESET}', end=" ")
    print()


def ida_f1(dirname):
    for filename in os.listdir(f'{dirname}/ida_labeled_code'):
        TP = FP = FN = TN = 0
        f_ida = open(f'{dirname}/ida_labeled_code/{filename}', 'r')
        f_truth = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
        for line_ida, line_truth in zip(f_ida, f_truth):
            line_ida_split = line_ida.strip().split()
            line_truth_split = line_truth.strip().split()
            if line_ida_split[1] == line_truth_split[1] and (line_truth_split[1] == 'F' or line_truth_split[1] == 'R'):
                TP += 1
            elif line_ida_split[1] == line_truth_split[1] == '-':
                TN += 1
            elif line_ida_split[1] in ['F', 'R'] and line_truth_split[1] == '-':
                FP += 1
            elif line_ida_split[1] == '-' and line_truth_split[1] in ['F', 'R']:
                FN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = 2 * precision * recall / (precision + recall)
        print(f'{filename}: TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, precision: {precision}, recall: {recall}, F1: {F1}')

        f_ida.close()
        f_truth.close()


def xda_f1(dirname, filename, model, tokens, end_idx=510):
    encoded_tokens = model.encode(' '.join(tokens))
    logprobs = model.predict('funcbound', encoded_tokens)
    labels = logprobs.argmax(dim=2).view(-1).data
    def abstract_label_from_line(line_truth):
        line_truth_split = line_truth.strip().split()
        return line_truth_split[1]
    TP = FP = FN = TN = 0
    f_xda = [value_to_char(v) for v in labels]
    f_truth_f = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
    f_truth = [abstract_label_from_line(line_truth) for line_truth in f_truth_f]
    del f_truth[510:]
    print(f'{Fore.BLUE}{"XDA prediction:"}{Fore.RESET}')
    print_labels(f_xda)
    print(f'{Fore.BLUE}{"Ground truth:"}{Fore.RESET}')
    print_labels(f_truth)
    for xda, truth in zip(f_xda, f_truth):
        if xda == truth and (truth == 'F' or truth == 'R'):
            TP += 1
        elif xda == truth == '-':
            TN += 1
        elif xda in ['F', 'R'] and  truth == '-':
            FP += 1
        elif xda == '-' and truth in ['F', 'R']:
            FN += 1
    precision = (1.0 * TP) / (TP + FP)
    recall = (1.0 * TP) / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    print(f'{filename}: TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}, precision: {precision}, recall: {recall}, F1: {F1}')

    f_truth_f.close()


def predict_color(dirname, filename, model, start_idx=0, end_idx=0):
    f_truth = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
    f_ida = open(f'{dirname}/ida_labeled_code/{filename}', 'r')

    tokens = []
    print('\nGround Truth:')
    for i, line_truth in enumerate(f_truth):
        if i < start_idx:
            continue
        line_truth_split = line_truth.strip().split()
        if line_truth_split[1] == '-':
            print(f'{int2hex(line_truth_split[0]).lower()}', end=" ")
        elif line_truth_split[1] == 'F':
            print(f'{Fore.RED}{int2hex(line_truth_split[0]).lower()}{Fore.RESET}', end=" ")
        elif line_truth_split[1] == 'R':
            print(f'{Fore.GREEN}{int2hex(line_truth_split[0]).lower()}{Fore.RESET}', end=" ")

        if i > end_idx:
            print(Style.RESET_ALL + '\n')
            break
    f_truth.close()

    f_truth = open(f'{dirname}/truth_labeled_code/{filename}', 'r')
    # for i, line_truth in enumerate(f_truth):
    #     if i < start_idx:
    #         continue

    #     line_truth_split = line_truth.strip().split()
    #     print(f'{line_truth_split[1]}', end=" ")

    #     if i > end_idx:
    #         print(Style.RESET_ALL + '\n')
    #         break
    f_truth.close()

    print('IDA-PRO:')
    for i, line_ida in enumerate(f_ida):
        if i < start_idx:
            continue

        line_ida_split = line_ida.strip().split()
        if line_ida_split[1] == '-':
            print(f'{int2hex(line_ida_split[0]).lower()}', end=" ")
        elif line_ida_split[1] == 'F':
            print(f'{Fore.RED}{int2hex(line_ida_split[0]).lower()}{Fore.RESET}', end=" ")
        elif line_ida_split[1] == 'R':
            print(f'{Fore.GREEN}{int2hex(line_ida_split[0]).lower()}{Fore.RESET}', end=" ")

        # Prepare the tokens for prediction by XDA
        tokens.append(int2hex(line_ida_split[0]).lower())

        if i > end_idx:
            print(Style.RESET_ALL + '\n')
            break
    f_ida.close()

    encoded_tokens = model.encode(' '.join(tokens))
    # print(encoded_tokens)
    logprobs = model.predict('funcbound', encoded_tokens)
    labels = logprobs.argmax(dim=2).view(-1).data

    print('XDA:')
    func_start = []
    func_end = []
    for i, (token, label) in enumerate(zip(tokens, labels)):
        if label == 0:
            print(f'{token}', end=" ")
        elif label == 2:
            print(f'{Fore.RED}{token}{Fore.RESET}', end=" ")
            func_start.append((i, token))
        elif label == 1:
            print(f'{Fore.GREEN}{token}{Fore.RESET}', end=" ")
            func_end.append((i, token))

    print(Style.RESET_ALL + '\n')
    return tokens, func_start, func_end


# Load our model
roberta = RobertaModel.from_pretrained('checkpoints/finetune_msvs_funcbound_64', 'checkpoint_best.pt',
                                       'data-bin/funcbound_msvs_64', bpe=None, user_dir='finetune_tasks')
roberta.eval()

dirname = 'data-raw/msvs_funcbound_64_bap_test'
end_idx = 510

print_line()
print("f1 value of IDA:")
ida_f1(dirname)
print_line()
pause()

# tokens, func_start, func_end = predict_color('msvs_64_O2_vim', roberta, start_idx=0, end_idx=510)
files_path = 'data-raw/msvs_funcbound_64_bap_test/truth_labeled_code'
files = [f for f in os.listdir(files_path)]

for filename in files:
    print('Playing with file: {}'.format(filename))
    tokens, func_start, func_end = predict_color(dirname, filename, roberta, start_idx=0, end_idx=end_idx)
    xda_f1(dirname, filename, roberta, tokens, end_idx)
    pause()

print('---END---')
