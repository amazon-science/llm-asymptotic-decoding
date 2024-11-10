from transformers import AutoTokenizer
import torch
import random
import sys
import os
import argparse

import logging
logging.getLogger('transformers.tokenization_utils').setLevel(logging.ERROR)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default = 'EleutherAI/pythia-70m-deduped')
    parser.add_argument("--input_file", type=str, required=True, default = 'data/raw/OWT_wiki_1e7')
    parser.add_argument("--output_dir", type=str, default = './data/processed/OWT_wiki_1e7_Pythia/tensors_all/')
    parser.add_argument("--training_ratio", type=float, default=0.96)
    parser.add_argument("--val_ratio", type=float, default=0.02)

    args = parser.parse_args()
    return args

args = parse_args()

input_file = args.input_file
output_dir = args.output_dir
model_name = args.model_name
training_ratio = args.training_ratio
val_ratio = args.val_ratio

output_train_file = output_dir + "train.pt"
output_val_file = output_dir + "val_org.pt"
output_test_file = output_dir + "test_org.pt"

if not os.path.exists(output_dir):
   os.makedirs(output_dir)

max_line_num = 100000000000000
#max_line_num = 100000
#max_line_num = 10000000
#max_line_num = 20000000
#max_line_num = 2000000

#max_sent_len = 256

output_arr = []

tokenizer = AutoTokenizer.from_pretrained(model_name)

i=0
with open(input_file, encoding='latin-1') as f_in:
    for line in f_in:
        raw_text = line
        i+=1
        #indexed_tokens = tokenizer.encode(raw_text, add_prefix_space=True)
        indexed_tokens = tokenizer.encode(raw_text)
        output_arr.append(indexed_tokens)
        if i % 100000 == 0:
            print(i)
            sys.stdout.flush()
        if i > max_line_num:
            break

#idx_shuffled = list(range(len(output_arr)))
#random.shuffle(idx_shuffled)
training_size = int(len(output_arr)*training_ratio)
val_size = int(len(output_arr)*val_ratio)

def save_to_tensor(output_arr, output_file_name):
    data_size = len(output_arr)
    len_sum = 0
    for sent in output_arr:
        sent_len = len(sent)
        len_sum += sent_len
    #output_tensor = torch.zeros((len_sum),dtype = torch.uint16)
    output_tensor = torch.zeros((len_sum),dtype = torch.int32)

    current_start = 0
    for i in range(data_size):
        sent = output_arr[i]
        #output_tensor[current_start:current_start+len(sent)] = torch.tensor(sent,dtype = torch.uint16)
        output_tensor[current_start:current_start+len(sent)] = torch.tensor(sent,dtype = torch.int32)
        current_start += len(sent)

    torch.save(output_tensor, output_file_name)

save_to_tensor(output_arr[:training_size], output_train_file)
save_to_tensor(output_arr[training_size:training_size+val_size], output_val_file)
save_to_tensor(output_arr[training_size+val_size:], output_test_file)
