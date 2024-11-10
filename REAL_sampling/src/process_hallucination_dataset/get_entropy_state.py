import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from compute_ent_features import word_ent_per, compute_model_feature
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from model import GPTNeoXForEntropyClassification

from torch.utils.data import Dataset, DataLoader

from data_classes import load_state_file, state_dataset

small_model_name = 'EleutherAI/pythia-70m-deduped'
large_model_name = 'EleutherAI/pythia-6.9b-deduped'

#device_ent = 'cpu'
#device_per = 'cpu'
#device_pythia = 'cpu'
device_ent = 'cuda:0'
device_per = 'cuda:1'
device_pythia = 'cuda:2'

ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1.4b_bsz_32_exp_pred_last"
per_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/per_OWT_1e6_1.4b_bsz_32_exp_pred_last"

#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/wiki_1e6_1.4b_bsz_32_exp_pred_last"
#per_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_410m_bsz_32_exp_pred_last"

log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]

#input_mode = 'state'
input_mode = 'humor'

if input_mode == 'state':
    data_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/state/'
    dataset_name = 'all'
elif input_mode == 'humor':
    data_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/humor/'
    #dataset_name = 'all_1024'
    dataset_name = 'all_128'
    #dataset_name = 'all_128_very_small'

#use_deepspeed = True
use_deepspeed = False

#dataset_name = 'animals_true_false'

#data_split_name = 'train'
data_split_name = 'val'
ent_model_name = '_OWT_perOWT'
#ent_model_name = '_wiki_smallOWT'

data_path = data_folder + dataset_name + '_' + data_split_name + '.csv'
output_path = data_folder + 'features/' + dataset_name + '_'+ data_split_name + ent_model_name + '.json'

#data_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_data_train.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_feature_train.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_feature_train_wiki_small.json'

def collect_features(org_left_text_len, org_left_text_tensor, model_small_lm, model_large_lm, model_ent, model_per):
    org_left_len = 0
    entropy_tensor_small = word_ent_per( model_small_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'ent', use_deepspeed)
    perplexity_tensor_small = word_ent_per( model_small_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'per', use_deepspeed)
    entropy_tensor_large = word_ent_per( model_large_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'ent', use_deepspeed)
    perplexity_tensor_large = word_ent_per( model_large_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'per', use_deepspeed)

    c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2 = compute_model_feature(model_ent, org_left_text_tensor, org_left_len, org_left_text_len)
    c_per, pred_last_per, curve_last_per, per_score1, per_score2 = compute_model_feature(model_per, org_left_text_tensor, org_left_len, org_left_text_len)

    ent_score3 = torch.pow(entropy_tensor_large * torch.maximum(torch.tensor(0), entropy_tensor_small - entropy_tensor_large), 0.5 )
    per_score3 = torch.pow(perplexity_tensor_large * torch.maximum(torch.tensor(0), perplexity_tensor_small - perplexity_tensor_large), 0.5 )
    #1-3 real entropy, 4-6 real perplexity, 7-11 predicted entropy, 12-16 predicted perplexity
    all_features_i = [entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2]
    #print(torch.stack(all_features_i, dim=1).size())
    all_features_i = torch.stack(all_features_i, dim=1).squeeze(dim=2).tolist()
    #all_features_i = torch.stack(all_features_i, dim=1).tolist()
    return all_features_i

def get_all_text_ent_features(dataloader, tokenizer,  model_ent, model_per, model_large_lm, model_small_lm):
    all_features_list = []
    all_label_list = []

    for idx, batch in enumerate(dataloader):
        print(idx / len(dataloader))
        left_text_len, left_text_tensor, labels = batch
        features_i = collect_features(left_text_len, left_text_tensor, model_small_lm, model_large_lm, model_ent, model_per)

        #input_len = torch.argmin()
        all_features_list.append(features_i)
        all_label_list.append(labels.tolist())

    return all_features_list, all_label_list

print('loading datasets')
tokenizer = AutoTokenizer.from_pretrained(small_model_name, truncation_side='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

with open(data_path) as f_in:
    if input_mode == 'state':
        output_text, output_label, output_cat = load_state_file(f_in, input_mode)
    elif input_mode == 'humor':
        output_text, output_label, output_cat, output_label_reg = load_state_file(f_in, input_mode)

#print(output_text[0])

dataset = state_dataset(output_text, output_label, tokenizer)
#dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
#dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

print('loading all models')
model_ent = GPTNeoXForEntropyClassification.from_pretrained(ent_model_path, log_model_size=log_model_size)
model_ent = model_ent.to(device_ent)

model_per = GPTNeoXForEntropyClassification.from_pretrained(per_model_path, log_model_size=log_model_size)
model_per = model_per.to(device_per)

model_large_lm = AutoModelForCausalLM.from_pretrained(large_model_name)
model_large_lm = model_large_lm.to(device_pythia)

model_small_lm = AutoModelForCausalLM.from_pretrained(small_model_name)
model_small_lm = model_small_lm.to(device_pythia)

if use_deepspeed:
    import deepspeed
    model_ent = deepspeed.init_inference(model_ent,mp_size=1,dtype=torch.half,replace_with_kernel_inject=True).module
    model_per = deepspeed.init_inference(model_per,mp_size=1,dtype=torch.half,replace_with_kernel_inject=True).module
    model_large_lm = deepspeed.init_inference(model_large_lm,mp_size=1,dtype=torch.half,replace_with_kernel_inject=True).module

model_ent.eval()
model_per.eval()
model_large_lm.eval()
model_small_lm.eval()

all_features, all_labels = get_all_text_ent_features(dataloader, tokenizer, model_ent, model_per, model_large_lm, model_small_lm)

with open(output_path, 'w') as f_out:
    if input_mode == 'state':
        json.dump([all_features, all_labels, output_cat], f_out,indent=4)
    elif input_mode == 'humor':
        json.dump([all_features, all_labels, output_cat, output_label_reg], f_out,indent=4)
