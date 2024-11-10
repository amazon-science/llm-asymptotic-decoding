import pandas as pd

from torch.utils.data import Dataset, DataLoader

import sys
HaDes_folder_path = "/mnt/efs/Haw-Shiuan/true_entropy/src/process_hallucination_dataset/Hades"
sys.path.append(HaDes_folder_path)

from data_loader import DataProcessor
from utils import remove_marked_sen

from mosestokenizer import *
detokenizer = MosesDetokenizer('en')

import json

def load_state_file(file_name, input_mode):
    df = pd.read_csv(file_name)
    output_text = df['statement'].tolist()
    output_label = df['label'].tolist()
    output_cat = df['category'].tolist()
    if input_mode == 'humor':
        output_label_reg = df['label_reg'].tolist()
        return output_text, output_label, output_cat, output_label_reg
    elif input_mode == 'state':
        return output_text, output_label, output_cat

class state_dataset(Dataset):
    def __init__(self, input_text, input_label, tokenizer):
        #left_text_arr = []
        #self.label_arr = []
        self.len_arr = []
        for i in range(len(input_text)):
            left_tok = tokenizer.tokenize(input_text[i])
            left_len = len(left_tok)
            self.len_arr.append(left_len)
        #for i in range(len(input_text)):
        #    left_text_arr.append(input_text[i])
        #    self.label_arr.append(input_label[i])
        #self.left_text_tensor = tokenizer(left_text_arr, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)
        #print(self.len_arr)

        self.label_arr = input_label
        self.left_text_tensor = tokenizer(input_text, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)
        #self.org_left_word_tensor = {k: v.to(device) for k, v in self.org_left_word_tensor.items()}

    def __len__(self):
        return self.left_text_tensor["input_ids"].size(0)

    def __getitem__(self, idx):
        #return self.org_left_len_arr[idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.org_left_tensor["length"][idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.left_text_tensor["length"][idx] - 2, self.left_text_tensor["input_ids"][idx,:], self.label_arr[idx]
        return self.len_arr[idx] - 2, self.left_text_tensor["input_ids"][idx,:], self.label_arr[idx]


def load_Halu_file(f_in):
    output_list = []
    for line in f_in:
        input_dict = json.loads(line)
        org_left = input_dict['context']
        pos_left_text = input_dict['context'] + input_dict['text_pos'].strip()
        neg_left_text = input_dict['context'] + input_dict['text_neg'].strip()

        output_list.append( [org_left, pos_left_text, neg_left_text] )
    return output_list


class Halu_dataset(Dataset):
    def __init__(self, examples, tokenizer):
        self.org_left_len_arr = []
        self.pos_left_len_arr = []
        self.neg_left_len_arr = []
        pos_left_text_arr = []
        neg_left_text_arr = []
        for example in examples:
            org_left, pos_left_text, neg_left_text = example
            org_left_tok = tokenizer.tokenize(org_left)
            pos_left_tok = tokenizer.tokenize(pos_left_text)
            neg_left_tok = tokenizer.tokenize(neg_left_text)
            #org_left_text_tok = tokenizer.tokenize(org_left_text)
            left_len = len(org_left_tok)
            self.org_left_len_arr.append(left_len)
            pos_left_len = len(pos_left_tok)
            self.pos_left_len_arr.append(pos_left_len)
            neg_left_len = len(neg_left_tok)
            self.neg_left_len_arr.append(neg_left_len)

            pos_left_text_arr.append(pos_left_text)
            neg_left_text_arr.append(neg_left_text)
        pos_left_text_tok = tokenizer.tokenize(pos_left_text_arr[0])
        print(pos_left_text_tok[:self.org_left_len_arr[0]])
        print(pos_left_text_tok[self.org_left_len_arr[0]:])
        #self.org_left_tensor = tokenizer(org_left_arr, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True)
        #self.org_left_tensor = {k: v.to(device) for k, v in self.org_left_tensor.items()}
        self.pos_left_text_tensor = tokenizer(pos_left_text_arr, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)
        self.neg_left_text_tensor = tokenizer(neg_left_text_arr, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)

        #self.org_left_word_tensor = {k: v.to(device) for k, v in self.org_left_word_tensor.items()}

    def __len__(self):
        return self.pos_left_text_tensor["input_ids"].size(0)

    def __getitem__(self, idx):
        #return self.org_left_len_arr[idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.org_left_tensor["length"][idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.org_left_len_arr[idx] - 2, self.pos_left_text_tensor["length"][idx] - 2, self.pos_left_text_tensor["input_ids"][idx,:], self.neg_left_text_tensor["length"][idx] - 2, self.neg_left_text_tensor["input_ids"][idx,:]
        return self.org_left_len_arr[idx] - 2, self.pos_left_len_arr[idx] - 2, self.pos_left_text_tensor["input_ids"][idx,:], self.neg_left_len_arr[idx] - 2, self.neg_left_text_tensor["input_ids"][idx,:]


def load_HaDes_example(data_path):
    dp = DataProcessor()
    examples = dp.get_examples(data_path)
    print(dp.get_label_dist())
    return examples

def example2text(example):
    rep_start_id, rep_end_id = example.idxs
    rep_tokens = remove_marked_sen(example.sen, rep_start_id, rep_end_id)
    left_context = rep_tokens[:rep_start_id]
    left_context_word = rep_tokens[:rep_end_id+1]
    org_left = detokenizer(left_context).strip()
    org_left_word = detokenizer(left_context_word).strip()
    org_left = " " + org_left if org_left[0] != ' ' else org_left
    org_left_word = " " + org_left_word if org_left_word[0] != ' ' else org_left_word

    return org_left, org_left_word

class HADESDataset(Dataset):
    def __init__(self, examples, tokenizer):
        #self.org_left_len_arr = []
        #org_left_arr = []
        org_left_word_arr = []
        self.len_left_word_arr = []
        self.len_left_arr = []
        for example in examples:
            org_left, org_left_word = example2text(example)
            org_left_tok = tokenizer.tokenize(org_left)
            org_left_word_tok = tokenizer.tokenize(org_left_word)
            left_len = len(org_left_tok)
            len_left_word = len(org_left_word_tok)
            if len_left_word <= left_len:
                left_len = len_left_word - 1
            assert len_left_word > left_len, print(org_left_word_tok, "<-- + word, no word -->", org_left_tok)
            self.len_left_arr.append(left_len)
            self.len_left_word_arr.append(len_left_word)
            #self.org_left_len_arr.append(left_len)
            #assert left_len + 1 < tokenizer.model_max_length
            org_left_word_trun = tokenizer.convert_tokens_to_string(org_left_word_tok[:len_left_word])

            #org_left_arr.append(org_left)
            org_left_word_arr.append(org_left_word_trun)
        print(org_left_word_arr[0])
        #self.org_left_tensor = tokenizer(org_left_arr, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True)
        #self.org_left_tensor = {k: v.to(device) for k, v in self.org_left_tensor.items()}
        self.org_left_word_tensor = tokenizer(org_left_word_arr, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)

        #self.org_left_word_tensor = {k: v.to(device) for k, v in self.org_left_word_tensor.items()}

    def __len__(self):
        return self.org_left_word_tensor["input_ids"].size(0)

    def __getitem__(self, idx):
        #return self.org_left_len_arr[idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.org_left_tensor["length"][idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.org_left_word_tensor["length"][idx] - 2, self.org_left_word_tensor["input_ids"][idx,:]
        return self.len_left_arr[idx] - 2, self.len_left_word_arr[idx] - 2, self.org_left_word_tensor["input_ids"][idx,:]


def combine_left_text(context_list, text_list):
    prefix = ' '
    output_list = []
    for context, text in zip(context_list, text_list):
        output_list.append(prefix + context + prefix + text.strip())
    return output_list

def load_factor_file(file_name):
    df = pd.read_csv(file_name)
    context = [x.strip() for x in df['turncated_prefixes'].tolist()]
    pos_text = combine_left_text(context, df['completion'].tolist())
    neg_text_1 = combine_left_text(context, df['contradiction_0'].tolist())
    neg_text_2 = combine_left_text(context, df['contradiction_1'].tolist())
    neg_text_3 = combine_left_text(context, df['contradiction_2'].tolist())
    return context, pos_text, neg_text_1, neg_text_2, neg_text_3


class factor_dataset(Dataset):
    def __init__(self, context, pos_text, neg_text_1, neg_text_2, neg_text_3, tokenizer):
        self.org_left_len_arr = []
        self.pos_len_arr = []
        self.neg_1_len_arr = []
        self.neg_2_len_arr = []
        self.neg_3_len_arr = []
        for i in range(len(context)):
            context_i = context[i]
            pos_text_i = pos_text[i]
            neg_1_text_i = neg_text_1[i]
            neg_2_text_i = neg_text_2[i]
            neg_3_text_i = neg_text_3[i]

            org_left_tok = tokenizer.tokenize(context_i)
            #org_left_text_tok = tokenizer.tokenize(org_left_text)
            left_len = len(org_left_tok)
            self.org_left_len_arr.append(left_len)
            self.pos_len_arr.append(len(tokenizer.tokenize(pos_text_i)))
            self.neg_1_len_arr.append(len(tokenizer.tokenize(neg_1_text_i)))
            self.neg_2_len_arr.append(len(tokenizer.tokenize(neg_2_text_i)))
            self.neg_3_len_arr.append(len(tokenizer.tokenize(neg_3_text_i)))

        pos_left_text_tok = tokenizer.tokenize(pos_text[0])
        print(pos_left_text_tok[:self.org_left_len_arr[0]])
        print(pos_left_text_tok[self.org_left_len_arr[0]:])
        #self.org_left_tensor = tokenizer(org_left_arr, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True)
        #self.org_left_tensor = {k: v.to(device) for k, v in self.org_left_tensor.items()}
        self.pos_left_text_tensor = tokenizer(pos_text, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)
        self.neg_1_left_text_tensor = tokenizer(neg_text_1, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)
        self.neg_2_left_text_tensor = tokenizer(neg_text_2, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)
        self.neg_3_left_text_tensor = tokenizer(neg_text_3, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)

        #self.org_left_word_tensor = {k: v.to(device) for k, v in self.org_left_word_tensor.items()}

    def __len__(self):
        return self.pos_left_text_tensor["input_ids"].size(0)

    def __getitem__(self, idx):
        #return self.org_left_len_arr[idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.org_left_tensor["length"][idx], self.org_left_tensor["input_ids"][idx,:], self.org_left_word_tensor["input_ids"][idx,:]
        #return self.org_left_len_arr[idx] - 2, self.pos_left_text_tensor["length"][idx] - 2, self.pos_left_text_tensor["input_ids"][idx,:], self.neg_1_left_text_tensor["length"][idx] - 2, self.neg_1_left_text_tensor["input_ids"][idx,:], self.neg_2_left_text_tensor["length"][idx] - 2, self.neg_2_left_text_tensor["input_ids"][idx,:], self.neg_3_left_text_tensor["length"][idx] - 2, self.neg_3_left_text_tensor["input_ids"][idx,:]
        return self.org_left_len_arr[idx] - 2, self.pos_len_arr[idx] - 2, self.pos_left_text_tensor["input_ids"][idx,:], self.neg_1_len_arr[idx] - 2, self.neg_1_left_text_tensor["input_ids"][idx,:], self.neg_2_len_arr[idx] - 2, self.neg_2_left_text_tensor["input_ids"][idx,:], self.neg_3_len_arr[idx] - 2, self.neg_3_left_text_tensor["input_ids"][idx,:]
