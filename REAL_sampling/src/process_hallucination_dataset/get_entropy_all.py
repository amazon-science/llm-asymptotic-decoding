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
from compute_ent_features import compute_model_feature
from data_classes import load_state_file, state_dataset, load_Halu_file, Halu_dataset, load_HaDes_example, HADESDataset, load_factor_file, factor_dataset

small_model_name = 'EleutherAI/pythia-70m-deduped'
large_model_name = 'EleutherAI/pythia-6.9b-deduped'

device_ent = 'cuda:0'

#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_32_exp_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_32_exp_pred_last_d08_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_8_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_32_exp_pred_last_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a6_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a4_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_32_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1024_70M_bsz_32_exp_pred_last_a4_e10"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_8_exp_pred_last_a4_e1"
ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/per_OWT_1e6_70M_bsz_128_exp_pred_last_a10_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1024_70M_bsz_64_exp_pred_last_e10"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1024_410M_bsz_32_exp_pred_last_a4_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1024_70M_bsz_64_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/wiki_1e6_70M_bsz_64_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/wiki_1e6_410M_bsz_8_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/wiki_1e6_70M_bsz_8_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1024_1e6_410M_bsz_8_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1.4b_bsz_16_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_1b_bsz_64_exp_pred_last"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_70M_bsz_32_exp_pred_last_d08_e3"
#ent_model_path = "/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_1e6_410M_bsz_16_exp_pred_last_d08_e3"

#ent_model_name = '_OWTwiki_1e7_70M_b32_e3_no_last'
#ent_model_name = '_OWTwiki_1e7_70M_b32_d08_e3'
#ent_model_name = '_OWTwiki_1e7_70M_b32_a6_e3'
#ent_model_name = '_OWTwiki_1e7_70M_b32_a4_e3'
#ent_model_name = '_OWTwiki_1e7_70M_b8_a4_e1'
ent_model_name = '_OWTwiki_1e7_70M_b128_a10_e3'
#ent_model_name = '_per_OWT_1e6_70M_b128_a10_e3'
#ent_model_name = '_OWTwiki_1e7_70M_b8_e1'
#ent_model_name = '_OWTwiki_1e7_70M_b32_e3'
#ent_model_name = '_OWTwiki_1e7_70M_b32_e1'
#ent_model_name = '_OWT_70M_b32_a4_e10'
#ent_model_name = '_OWT_70M_b32_d08_e3'
#ent_model_name = '_OWT_410M_b16_d08_e3'
#ent_model_name = '_OWT_410M_b32_a4_e3'
#ent_model_name = '_OWT_70M_b64_e10'
#ent_model_name = '_OWT_70M_b64'
#ent_model_name = '_wiki_70M_b64'
#ent_model_name = '_wiki_410M'
#ent_model_name = '_wiki_70M'
#ent_model_name = '_wiki_70M_a4_e3'
#ent_model_name = '_OWT_b16'
#ent_model_name = '_OWT_1b_b64'

log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]

dataset_cat_d2_names = {'factor': ['expert_factor', 'news_factor', 'wiki_factor'], 
                     'HaDes': ['all'], 
                     #'Halu': ['dialogue_data', 'qa_data', 'dialogue_data_knowledge', 'summarization_data_1024'], 
                     #'Halu': ['dialogue_data', 'qa_data', 'summarization_data_1024'], 
                     #'humor': ['all_128'], 
                     'state': ['all'] } 

data_splits = ['train', 'val']


#use_deepspeed = True
use_deepspeed = False

def get_feature_from_model(org_left_len, org_left_word_len, org_left_word_tensor, model_ent):
    c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2 = compute_model_feature(model_ent, org_left_word_tensor, org_left_len, org_left_word_len)
    all_features_i = [c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2]
    all_features_i = torch.stack(all_features_i, dim=1).squeeze(dim=2).tolist()
    return all_features_i

def get_all_factor_features(dataloader, tokenizer,  model_ent):
    all_pos_features_list = []
    all_neg_1_features_list = []
    all_neg_2_features_list = []
    all_neg_3_features_list = []

    for idx, batch in enumerate(dataloader):
        print(idx / len(dataloader))
        org_left_len, pos_left_text_len, pos_left_text_tensor, neg_1_left_text_len, neg_1_left_text_tensor, neg_2_left_text_len, neg_2_left_text_tensor, neg_3_left_text_len, neg_3_left_text_tensor = batch
        pos_features_i = get_feature_from_model(org_left_len, pos_left_text_len, pos_left_text_tensor, model_ent)
        neg_1_features_i = get_feature_from_model(org_left_len, neg_1_left_text_len, neg_1_left_text_tensor, model_ent)
        neg_2_features_i = get_feature_from_model(org_left_len, neg_2_left_text_len, neg_2_left_text_tensor, model_ent)
        neg_3_features_i = get_feature_from_model(org_left_len, neg_3_left_text_len, neg_3_left_text_tensor, model_ent)

        #input_len = torch.argmin()
        all_pos_features_list.append(pos_features_i)
        all_neg_1_features_list.append(neg_1_features_i)
        all_neg_2_features_list.append(neg_2_features_i)
        all_neg_3_features_list.append(neg_3_features_i)

    #all_pos_features = torch.concat(all_pos_features_list) #the last batch size might be different, so cannot use stack
    #all_neg_features = torch.concat(all_neg_features_list) #the last batch size might be different, so cannot use stack
    return all_pos_features_list, all_neg_1_features_list, all_neg_2_features_list, all_neg_3_features_list


def get_all_Halu_features(dataloader, tokenizer,  model_ent):
    all_pos_features_list = []
    all_neg_features_list = []

    for idx, batch in enumerate(dataloader):
        print(idx / len(dataloader))
        org_left_len, pos_left_text_len, pos_left_text_tensor, neg_left_text_len, neg_left_text_tensor = batch
        pos_features_i = get_feature_from_model(org_left_len, pos_left_text_len, pos_left_text_tensor, model_ent)
        neg_features_i = get_feature_from_model(org_left_len, neg_left_text_len, neg_left_text_tensor, model_ent)

        #input_len = torch.argmin()
        all_pos_features_list.append(pos_features_i)
        all_neg_features_list.append(neg_features_i)

    #all_pos_features = torch.concat(all_pos_features_list) #the last batch size might be different, so cannot use stack
    #all_neg_features = torch.concat(all_neg_features_list) #the last batch size might be different, so cannot use stack
    return all_pos_features_list, all_neg_features_list

def get_all_HaDes_features(dataloader, tokenizer,  model_ent):
    all_features_list = []
    for idx, batch in enumerate(dataloader):
        print(idx / len(dataloader))
        org_left_len, org_left_word_len, org_left_word_tensor = batch
        all_features_i = get_feature_from_model(org_left_len, org_left_word_len, org_left_word_tensor, model_ent)
        all_features_list.append(torch.tensor(all_features_i))

    all_ent_features = torch.concat(all_features_list)
    return all_ent_features

def get_all_state_features(dataloader, tokenizer,  model_ent):
    all_features_list = []
    all_label_list = []

    for idx, batch in enumerate(dataloader):
        print(idx / len(dataloader))
        left_text_len, left_text_tensor, labels = batch
        org_left_len = 0
        features_i = get_feature_from_model(org_left_len, left_text_len, left_text_tensor, model_ent)

        #input_len = torch.argmin()
        all_features_list.append(features_i)
        all_label_list.append(labels.tolist())

    return all_features_list, all_label_list

def run_main(dataset_cat, data_path, output_path, model_ent, tokenizer):

    print('loading datasets')
    if dataset_cat == 'HaDes':
        examples = load_HaDes_example(data_path)
        dataset = HADESDataset(examples, tokenizer)
        bsz = 2
    else:
        with open(data_path) as f_in:
            if dataset_cat == 'Halu':
                examples = load_Halu_file(f_in)
                dataset = Halu_dataset(examples, tokenizer)
                if 'summarization' in data_path:
                    bsz = 1
                else:
                    bsz = 2
            elif dataset_cat == 'factor':
                context, pos_text, neg_text_1, neg_text_2, neg_text_3 = load_factor_file(f_in)
                dataset = factor_dataset(context, pos_text, neg_text_1, neg_text_2, neg_text_3, tokenizer)
                bsz = 1
            elif dataset_cat == 'state':
                output_text, output_label, output_cat = load_state_file(f_in, dataset_cat)
                dataset = state_dataset(output_text, output_label, tokenizer)
                bsz = 4
            elif dataset_cat == 'humor':
                output_text, output_label, output_cat, output_label_reg = load_state_file(f_in, dataset_cat)
                dataset = state_dataset(output_text, output_label, tokenizer)
                bsz = 4

    dataloader = DataLoader(dataset, batch_size=bsz, shuffle=False)

    with open(output_path, 'w') as f_out:
        if dataset_cat == 'HaDes':
            all_ent_features = get_all_HaDes_features(dataloader, tokenizer,  model_ent)
            output_dict = {}
            for i in range(len(examples)):
                output_dict[ examples[i].sen ] = all_ent_features[i,:].squeeze().tolist()
            json.dump(output_dict, f_out,indent=4)
        elif dataset_cat == 'Halu':
            all_pos_features, all_neg_features = get_all_Halu_features(dataloader, tokenizer, model_ent)
            json.dump([all_pos_features, all_neg_features], f_out,indent=4)
        elif dataset_cat == 'factor':
            all_pos_features_list, all_neg_1_features_list, all_neg_2_features_list, all_neg_3_features_list = get_all_factor_features(dataloader, tokenizer, model_ent)
            json.dump([all_pos_features_list, all_neg_1_features_list, all_neg_2_features_list, all_neg_3_features_list], f_out,indent=4)
        elif dataset_cat == 'state':
            all_features, all_labels = get_all_state_features(dataloader, tokenizer,  model_ent)
            json.dump([all_features, all_labels, output_cat], f_out,indent=4)
        elif dataset_cat == 'humor':
            all_features, all_labels = get_all_state_features(dataloader, tokenizer,  model_ent)
            json.dump([all_features, all_labels, output_cat, output_label_reg], f_out,indent=4)

tokenizer = AutoTokenizer.from_pretrained(small_model_name, truncation_side='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = 2048

print('loading all models')
use_a4 = False
use_a56 = False
use_a10 = False
if '_a4_' in ent_model_path:
    use_a4 = True
if '_a6_' in ent_model_path:
    use_a4 = True
    use_a56 = True
if '_a10_' in ent_model_path:
    use_a4 = True
    use_a56 = True
    use_a10 = True
model_ent = GPTNeoXForEntropyClassification.from_pretrained(ent_model_path, log_model_size=log_model_size, use_a4=use_a4, use_a56=use_a56, use_a10=use_a10)
#print( 'use_a4',  model_ent.use_a4 )
model_ent = model_ent.to(device_ent)

if use_deepspeed:
    import deepspeed
    model_ent = deepspeed.init_inference(model_ent,mp_size=1,dtype=torch.half,replace_with_kernel_inject=True).module

model_ent.eval()

for dataset_cat in dataset_cat_d2_names:
    #data_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/'
    data_folder = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/' + dataset_cat + '/' 
    for dataset_name in dataset_cat_d2_names[dataset_cat]:
        for data_split_name in data_splits:
            print(ent_model_name, dataset_cat, dataset_name, data_split_name)
            file_format = '.json'
            if dataset_cat == 'factor' or dataset_cat == 'state' or dataset_cat == 'humor':
                file_format = '.csv'
            elif dataset_cat == 'HaDes':
                file_format = '.txt'
            data_path = data_folder + dataset_name + '_' + data_split_name + file_format
            output_path = data_folder + 'features/' + dataset_name + '_'+ data_split_name + ent_model_name + '.json'
            run_main(dataset_cat, data_path, output_path, model_ent, tokenizer)


