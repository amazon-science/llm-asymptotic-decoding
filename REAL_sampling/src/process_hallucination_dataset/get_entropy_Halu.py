import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from compute_ent_features import collect_features_Halu
import json

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
from model import GPTNeoXForEntropyClassification

from torch.utils.data import Dataset, DataLoader

from data_classes import load_Halu_file, Halu_dataset

small_model_name = 'EleutherAI/pythia-70m-deduped'
large_model_name = 'EleutherAI/pythia-6.9b-deduped'

device_ent = 'cuda:0'
device_per = 'cuda:1'
device_pythia = 'cuda:2'

#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1.4b_bsz_32_exp_pred_last"
#per_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/per_OWT_1e6_1.4b_bsz_32_exp_pred_last"

ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/wiki_1e6_1.4b_bsz_32_exp_pred_last"
per_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_410m_bsz_32_exp_pred_last"

log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]

data_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/'

#use_deepspeed = True
use_deepspeed = False

datasets = ['dialogue_data', 'qa_data', 'dialogue_data_knowledge', 'summarization_data_1024']
#ent_model_names = ['OWT_perOWT', 'wiki_smallOWT']
#ent_model_names = ['OWT_perOWT']
#ent_model_names = ['wiki_smallOWT']
data_splits = ['train', 'val']


#dataset_name = 'dialogue_data'
#dataset_name = 'dialogue_data_knowledge'
#dataset_name = 'qa_data'
#dataset_name = 'qa_data_knowledge'
#dataset_name = 'summarization_data_2048'
#dataset_name = 'summarization_data_1024'

#data_split_name = 'train'
#data_split_name = 'val'
#ent_model_name = '_OWT_perOWT'
ent_model_name = '_wiki_smallOWT'


#data_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_data_train.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_feature_train.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_feature_train_wiki_small.json'


def get_all_text_ent_features(dataloader, tokenizer,  model_ent, model_per, model_large_lm, model_small_lm):
    all_pos_features_list = []
    all_neg_features_list = []

    for idx, batch in enumerate(dataloader):
        print(idx / len(dataloader))
        org_left_len, pos_left_text_len, pos_left_text_tensor, neg_left_text_len, neg_left_text_tensor = batch
        pos_features_i = collect_features_Halu(org_left_len, pos_left_text_len, pos_left_text_tensor, model_small_lm, model_large_lm, model_ent, model_per, use_deepspeed)
        neg_features_i = collect_features_Halu(org_left_len, neg_left_text_len, neg_left_text_tensor, model_small_lm, model_large_lm, model_ent, model_per, use_deepspeed)

        #input_len = torch.argmin()
        all_pos_features_list.append(pos_features_i)
        all_neg_features_list.append(neg_features_i)

    #all_pos_features = torch.concat(all_pos_features_list) #the last batch size might be different, so cannot use stack
    #all_neg_features = torch.concat(all_neg_features_list) #the last batch size might be different, so cannot use stack
    return all_pos_features_list, all_neg_features_list

def run_main(data_path, output_path):

    print('loading datasets')
    with open(data_path) as f_in:
        examples = load_Halu_file(f_in)

    dataset = Halu_dataset(examples, tokenizer)
    if 'summarization' in data_path:
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    else:
        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)
    #dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

    all_pos_features, all_neg_features = get_all_text_ent_features(dataloader, tokenizer, model_ent, model_per, model_large_lm, model_small_lm)

    with open(output_path, 'w') as f_out:
        json.dump([all_pos_features, all_neg_features], f_out,indent=4)


tokenizer = AutoTokenizer.from_pretrained(small_model_name, truncation_side='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

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
    #model_ent = deepspeed.init_inference(model_ent,mp_size=1,dtype=torch.float16,replace_with_kernel_inject=True).module
    #model_per = deepspeed.init_inference(model_per,mp_size=1,dtype=torch.float16,replace_with_kernel_inject=True).module
    #model_large_lm = deepspeed.init_inference(model_large_lm,mp_size=1,dtype=torch.float16,replace_with_kernel_inject=True).module
    model_ent = deepspeed.init_inference(model_ent,mp_size=1,dtype=torch.half,replace_with_kernel_inject=True).module
    model_per = deepspeed.init_inference(model_per,mp_size=1,dtype=torch.half,replace_with_kernel_inject=True).module
    model_large_lm = deepspeed.init_inference(model_large_lm,mp_size=1,dtype=torch.half,replace_with_kernel_inject=True).module

model_ent.eval()
model_per.eval()
model_large_lm.eval()
model_small_lm.eval()

for dataset_name in datasets:
    for data_split_name in data_splits:
        print(dataset_name, ent_model_name, data_split_name)
        data_path = data_folder + dataset_name + '_' + data_split_name + '.json'
        output_path = data_folder + 'features/' + dataset_name + '_'+ data_split_name + ent_model_name + '.json'
        run_main(data_path, output_path)

