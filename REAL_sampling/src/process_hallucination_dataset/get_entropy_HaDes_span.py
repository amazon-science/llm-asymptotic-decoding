import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
#HaDes_folder_path = "/mnt/efs/Haw-Shiuan/HaDes/baselines"
#sys.path.append(HaDes_folder_path)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from model import GPTNeoXForEntropyClassification
#import matplotlib.pyplot as plt
#from train_entropy_prediction_model import compose_dataset

from torch.utils.data import DataLoader

from compute_ent_features import word_ent_per, compute_model_feature
from data_classes import HADESDataset, load_HaDes_example

import json

def get_all_ent_features(dataloader, tokenizer,  model_ent, model_per, model_large_lm, model_small_lm):
    all_features_list = []

    for idx, batch in enumerate(dataloader):
        print(idx / len(dataloader))
        #org_left_len, org_left_word_tensor = batch
        org_left_len, org_left_word_len, org_left_word_tensor = batch
        #input_len = torch.argmin()
        entropy_tensor_small = word_ent_per( model_small_lm, org_left_word_tensor, org_left_len, org_left_word_len, 'ent')
        perplexity_tensor_small = word_ent_per( model_small_lm, org_left_word_tensor, org_left_len, org_left_word_len, 'per')
        entropy_tensor_large = word_ent_per( model_large_lm, org_left_word_tensor, org_left_len, org_left_word_len, 'ent')
        perplexity_tensor_large = word_ent_per( model_large_lm, org_left_word_tensor, org_left_len, org_left_word_len, 'per')

        c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2 = compute_model_feature(model_ent, org_left_word_tensor, org_left_len, org_left_word_len)
        c_per, pred_last_per, curve_last_per, per_score1, per_score2 = compute_model_feature(model_per, org_left_word_tensor, org_left_len, org_left_word_len)

        ent_score3 = torch.pow(entropy_tensor_large * torch.maximum(torch.tensor(0), entropy_tensor_small - entropy_tensor_large),0.5 )
        per_score3 = torch.pow(perplexity_tensor_large * torch.maximum(torch.tensor(0), perplexity_tensor_small - perplexity_tensor_large),0.5 )
        all_features_i = [entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2]
        #1-3 real entropy, 4-6 real perplexity, 7-11 predicted entropy, 12-16 predicted perplexity
        all_features_i = torch.stack(all_features_i, dim=1)
        all_features_list.append(all_features_i)

    all_ent_features = torch.concat(all_features_list) #the last batch size might be different, so cannot use stack
    return all_ent_features

#device_ent = 'cpu'
#device_per = 'cpu'
#device_pythia = 'cpu'

device_ent = 'cuda:0'
device_per = 'cuda:1'
device_pythia = 'cuda:2'

small_model_name = 'EleutherAI/pythia-70m-deduped'
large_model_name = 'EleutherAI/pythia-6.9b-deduped'

#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/wiki_1e6_1.4b_bsz_32_exp_pred_last"
#per_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_410m_bsz_32_exp_pred_last"

ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1.4b_bsz_32_exp_pred_last"
per_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/per_OWT_1e6_1.4b_bsz_32_exp_pred_last"
log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]

#data_path = '/mnt/efs/Haw-Shiuan/HaDes/data_collections/Wiki-Hades/train.txt'
data_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/all_train.txt'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/features/all_span_train_wiki_smallOWT.json'
output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/features/all_span_train_OWT_perOWT.json'

#data_path = '/mnt/efs/Haw-Shiuan/HaDes/data_collections/Wiki-Hades/valid.txt'
#data_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/all_val.txt'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/features/all_span_val_wiki_smallOWT.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/HaDes/features/all_span_val_OWT_perOWT.json'

print('loading datasets')
tokenizer = AutoTokenizer.from_pretrained(small_model_name, truncation_side='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

examples = load_HaDes_example(data_path)

print(len(examples))
print(examples[0].sen)
print(examples[0].idxs)
print(examples[0].guid)

dataset = HADESDataset(examples, tokenizer)
dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

print('loading all models')
model_ent = GPTNeoXForEntropyClassification.from_pretrained(ent_model_path, log_model_size=log_model_size)
model_ent = model_ent.to(device_ent)
model_ent.eval()

model_per = GPTNeoXForEntropyClassification.from_pretrained(per_model_path, log_model_size=log_model_size)
model_per = model_per.to(device_per)
model_per.eval()

model_large_lm = AutoModelForCausalLM.from_pretrained(large_model_name)
model_large_lm = model_large_lm.to(device_pythia)
model_large_lm.eval()

model_small_lm = AutoModelForCausalLM.from_pretrained(small_model_name)
model_small_lm = model_small_lm.to(device_pythia)
model_small_lm.eval()


all_ent_features = get_all_ent_features(dataloader, tokenizer, model_ent, model_per, model_large_lm, model_small_lm)

assert len(examples) == all_ent_features.size(0)

output_dict = {}

for i in range(len(examples)):
    output_dict[ examples[i].sen ] = all_ent_features[i,:].squeeze().tolist()

with open(output_path, 'w') as f_out:
    json.dump(output_dict, f_out,indent=4)
