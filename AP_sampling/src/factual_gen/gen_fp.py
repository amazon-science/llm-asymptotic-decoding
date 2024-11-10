import sys

import torch
import random

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import argparse
from tqdm import tqdm
import json
import time
from datetime import datetime

#from model import GPTNeoXForEntropyClassification

from torch.utils.data import Dataset, DataLoader
from sampling_method import FactualTopPLogitsWarper, LogitsProcessorList, FETopPLogitsWarper, PeriodFactualTopPLogitsWarper, ContrastiveDecodingLogitsWarper, APTopKLogitsWarper, CDTopKLogitsWarper, APTopPALogitsWarper, APCDTopPALogitsWarper

model_d2_short = {'EleutherAI/pythia-6.9b-deduped': '6.9b', 'EleutherAI/pythia-70m-deduped': '70M', 'openlm-research/open_llama_7b_v2': 'OpenLLaMA2-7b', 'facebook/opt-6.7b': 'OPT-6.7b', 'openai-community/gpt2-xl': 'GPT2-XL', 'Qwen/Qwen1.5-7b': 'Qwen1.5-7b', 'Qwen/Qwen1.5-7b-Chat': 'Qwen1.5-7b-Chat', 'Qwen/Qwen1.5-4b': 'Qwen1.5-4b', 'Qwen/Qwen1.5-4b-Chat': 'Qwen1.5-4b-Chat', 'apple/OpenELM-3B': 'OpenELM-3B', 'apple/OpenELM-3B-Instruct': 'OpenELM-3B-Instruct'}
data_d2_short = {'fever_factual_100_final.jsonl': 'factual_100', 'fever_factual_1000_final.jsonl': 'factual_1000', 'fever_factual_final.jsonl': 'factual', 'fever_factual_test7k_final.jsonl': 'factual_test7k', 'fever_nonfactual_100_final.jsonl': 'nonfactual_100', 'fever_nonfactual_1000_final.jsonl': 'nonfactual_1000', 'fever_nonfactual_final.jsonl': 'nonfactual', 'fever_nonfactual_test7k_final.jsonl': 'nonfactual_test7k', 'prompt_10.jsonl': 'story_10', 'prompt_100.jsonl': 'story_100', 'prompt_1000.jsonl': 'story_1000', 'prompt_start2_b1_1000.jsonl': 'story_start2_1000', 'prompt_start2_b1_100.jsonl': 'story_start2_100', 'wp_test_100.jsonl': 'wp_test_100', 'wp_test_1000.jsonl': 'wp_test_1000', 'news_test_1000.jsonl': 'news_test_1000', 'news_test_100.jsonl': 'news_test_100'}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default = 'EleutherAI/pythia-70m-deduped')
    #parser.add_argument("--model_name", type=str, default = 'EleutherAI/pythia-6.9b-deduped')
    parser.add_argument("--input_file_name", type=str, default = '/mnt/efs/Haw-Shiuan/Hallucination_repos/FactualityPrompt/prompts/fever_factual_100_final.jsonl')
    #parser.add_argument("--input_file_name", type=str, default = '/mnt/efs/Haw-Shiuan/Hallucination_repos/FactualityPrompt/prompts/fever_nonfactual_100_final.jsonl')
    #parser.add_argument("--input_file_name", type=str, default = '/mnt/efs/Haw-Shiuan/Hallucination_repos/FactualityPrompt/prompts/fever_factual_1000_final.jsonl')
    #parser.add_argument("--input_file_name", type=str, default = '/mnt/efs/Haw-Shiuan/Hallucination_repos/FactualityPrompt/prompts/fever_nonfactual_1000_final.jsonl')
    #parser.add_argument("--output_path", type=str, default = 'outputs/factual_gen/factual_100_70M')
    parser.add_argument("--output_path_prefix", type=str, default = 'outputs/factual_gen/')
    #parser.add_argument("--final_entropy_model_path", type=str, default = 'models/wiki_1e6_70M_bsz_8_exp_pred_last_a4_e3')
    #parser.add_argument("--final_entropy_model_path", type=str, default = 'models/wiki_1e6_410M_bsz_8_exp_pred_last')
    parser.add_argument("--final_entropy_model_path", type=str, default = 'models/OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a6_e3')
    parser.add_argument("--sample_method", type=str, default = 'fe_topp')
    parser.add_argument("--use_pythia_real_with_AP", type=bool, default = True)
    #parser.add_argument("--sample_method", type=str, default = 'fecut_topp')
    #parser.add_argument("--window_size", type=int, default=20)
    parser.add_argument("--window_size", type=int, default=40)
    #parser.add_argument("--window_size", type=int, default=1024)
    #parser.add_argument("--window_size", type=int, default=1)
    #parser.add_argument("--sample_sub_method", type=str, default = 'exp_e_only_win')
    #parser.add_argument("--sample_sub_method", type=str, default = 'exp_raw_e_only_win')
    #parser.add_argument("--sample_sub_method", type=str, default = 'exp_real_e_only_win')
    #parser.add_argument("--sample_sub_method", type=str, default = 'exp_large_win')
    parser.add_argument("--sample_sub_method", type=str, default = 'exp_1_win')
    #parser.add_argument("--sample_sub_method", type=str, default = 'exp_1')
    #parser.add_argument("--sample_sub_method", type=str, default = 'exp_1_norm')
    #parser.add_argument("--sample_sub_method", type=str, default = 'exp_2')
    parser.add_argument("--cut_sample_sub_method", type=str, default = 'norm')
    #parser.add_argument("--batch_size", type=int, default=5)
    #parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--cuda_idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--decay_temperature", type=float, default=1)
    parser.add_argument("--p", type=float, default=1)
    #parser.add_argument("--p", type=float, default=0.7)
    #parser.add_argument("--p", type=float, default=0.3)
    #parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--max_len", type=int, default=128)
    #parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--num_return_sequences", type=int, default=1)
    #parser.add_argument("--num_existing_seeds", type=int, default=0)
    parser.add_argument("--num_existing_seeds", type=int, default=1)

    args = parser.parse_args()
    return args

class factual_dataset(Dataset):
    def __init__(self, f_in, tokenizer):
        prompt_list = []
        self.id_list = []
        self.ref_list = []
        for line in f_in:
            input_dict = json.loads(line)
            prompt = input_dict['prompt']
            idx = input_dict['id']
            
            prompt = " " + prompt if prompt[0] != ' ' else prompt
            if 'Chat' in tokenizer.name_or_path:
                #prompt = ' Here is a sentence from the Wikipedia: ' + prompt + ' Please generate the continuation of the above Wikipedia sentence using at least 5 sentences: '
                #prompt = ' Please complete the following paragraph using at least 5 sentences:' + prompt + ' '
                #prompt = ' Please complete the following paragraph using at least 5 English sentences:' + prompt + ' '
                prompt = ' Please DO NOT output any Chinese and complete the following Wikipedia paragraph using at least 5 English sentences:' + prompt + ' '
                #prompt = ' Your task is to write a paragraph for Wikipedia and the paragraph should contain more than 5 sentences:' + prompt
            prompt_list.append( prompt )
            self.id_list.append( idx )
            if 'ref' in input_dict:
                ref = input_dict['ref']
                self.ref_list.append(ref)
        assert len(self.ref_list) == 0 or len(self.ref_list) == len(self.id_list)

        #self.text_tensor = tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True, max_length = tokenizer.model_max_length)
        self.text_tensor = tokenizer(prompt_list, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True)

    def __len__(self):
        return self.text_tensor["input_ids"].size(0)

    def __getitem__(self, idx):
        return self.text_tensor["input_ids"][idx,:], self.text_tensor['attention_mask'][idx,:]


def save_gen(f_out, output_list, id_list, ref_list):
    assert len(output_list) == len(id_list)
    for i in range(len(output_list)):
        if len(ref_list) > 0:
            output_dict = {'prompt': output_list[i][0], 'text': output_list[i][1], 'id': id_list[i], 'ref': ref_list[i] }
        else:
            output_dict = {'prompt': output_list[i][0], 'text': output_list[i][1], 'id': id_list[i] }
        f_out.write( json.dumps(output_dict) + '\n' )

def set_random_seeds(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    set_random_seeds(args.cuda_idx)
    input_file_name = os.path.basename( args.input_file_name )

    AP_entropy_model_path = args.final_entropy_model_path
    output_folder = args.output_path_prefix + data_d2_short[input_file_name].replace('nonfactual','factual') + '_' + model_d2_short[args.model_name] + '_' + args.sample_method

    if args.sample_method == 'fe_topp':
        sub_method_name = args.sample_sub_method
        if sub_method_name[-4:] == '_win':
            sub_method_name += '_' + str(args.window_size)
        output_folder += '_'+sub_method_name+'_dt_'+ str(args.decay_temperature) +'_p' + str(args.p) + "_fixed_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'fe_CD':
        sub_method_name = args.sample_sub_method
        if sub_method_name[-4:] == '_win':
            sub_method_name += '_' + str(args.window_size)
        output_folder += '_'+sub_method_name+'_dt_'+ str(args.decay_temperature) +'_log_alpha' + str(args.p) + "_fixed_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'fe_CD_topp':
        sub_method_name = args.sample_sub_method
        if sub_method_name[-4:] == '_win':
            sub_method_name += '_' + str(args.window_size)
        output_folder += '_'+sub_method_name+'_dt_'+ str(args.decay_temperature) +'_log_p' + str(args.p) + "_fixed_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'fe_AP_topp':
        if args.use_pythia_real_with_AP:
            AP_entropy_model_path = '/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3'
        else:
            if 'pythia' in args.model_name:
                AP_entropy_model_path = '/mnt/efs/Haw-Shiuan/true_entropy/models/OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3'
            elif 'opt' in args.model_name:
                AP_entropy_model_path = '/mnt/efs/Haw-Shiuan/true_entropy/models/OPT_OWT_wiki_1e7_70M_bsz_64_exp_pred_last_a10_e3'
            elif 'gpt2' in args.model_name:
                AP_entropy_model_path = '/mnt/efs/Haw-Shiuan/true_entropy/models/GPT2_OWT_wiki_1e7_70M_bsz_64_exp_pred_last_a10_e3'
        sub_method_name = args.sample_sub_method
        if sub_method_name[-4:] == '_win':
            sub_method_name += '_' + str(args.window_size)
        output_folder += '_'+sub_method_name+'_dt_'+ str(args.decay_temperature) +'_log_ait' + str(args.p) + '_' + os.path.basename(AP_entropy_model_path) + "_fixed_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'fe_topp_period':
        sub_method_name = args.sample_sub_method
        if sub_method_name[-4:] == '_win':
            sub_method_name += '_' + str(args.window_size)
        output_folder += '_'+sub_method_name+'_dt_'+ str(args.decay_temperature) +'_our_period_r' + str(args.p) + "_fixed_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'fe_topk':
        sub_method_name = args.sample_sub_method
        if sub_method_name[-4:] == '_win':
            sub_method_name += '_' + str(args.window_size)
        output_folder += '_'+sub_method_name+'_dt_'+ str(args.decay_temperature) +'_k' + str(args.p) + "_fixed_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'CD_topk':
        output_folder += '_dt_'+ str(args.decay_temperature) +'_k' + str(args.p) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'CD_toppk20':
        output_folder += '_dt_'+ str(args.decay_temperature) +'_log_p' + str(args.p) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'CD_topp':
        output_folder += '_dt_'+ str(args.decay_temperature) +'_log_p' + str(args.p) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'CD':
        output_folder += '_dt_'+ str(args.decay_temperature) +'_log_p' + str(args.p) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'CDk20':
        output_folder += '_dt_'+ str(args.decay_temperature) +'_log_p' + str(args.p) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'AP_topk':
        output_folder += '_k' + str(args.p) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'AP_topp':
        output_folder += '_log_p' + str(args.p) + "_dt_" + str(args.decay_temperature) +  "_" + os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'AP_toppk20':
        output_folder += '_log_p' + str(args.p) + "_dt_" + str(args.decay_temperature) +  "_" + os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'AP_CD_toppk20':
        output_folder += '_log_p' + str(args.p) + "_dt_" + str(args.decay_temperature) +  "_" + os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'AP_topa':
        output_folder += '_log_alpha' + str(args.p) + "_dt_" + str(args.decay_temperature) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'AP_topak20':
        output_folder += '_log_alpha' + str(args.p) + "_dt_" + str(args.decay_temperature) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'fecut_topp':
        output_folder += '_'+args.cut_sample_sub_method+ '_dt_'+ str(args.decay_temperature) + "_" +  os.path.basename(args.final_entropy_model_path)
    elif args.sample_method == 'topp':
        output_folder += '_p' + str(args.p) + '_temp_'+ str(args.decay_temperature)
    elif args.sample_method == 'toppk20':
        output_folder += '_p' + str(args.p) + '_temp_'+ str(args.decay_temperature)
    elif args.sample_method == 'topk':
        output_folder += '_k' + str(args.p) 
    elif args.sample_method == 'CS':
        output_folder += '_k' + str(args.p) + '_alpha_'+ str(args.decay_temperature)
    elif args.sample_method == 'eta':
        output_folder += '_p' + str(args.p)
    elif args.sample_method == 'typical':
        output_folder += '_p' + str(args.p)
    elif args.sample_method == 'decay' or args.sample_method == 'decay_period':
        output_folder += '_p' + str(args.p) + '_dt_'+ str(args.decay_temperature)
    
    if 'Chat' in args.model_name:
        output_folder += '_wikieng'
        #output_folder += '_Eng'

    time.sleep(args.cuda_idx)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_name = output_folder  + '/' + data_d2_short[input_file_name] + '_' + model_d2_short[args.model_name] + '_' + args.sample_method + '_p' + str(args.p) + '_gen_seed{}.jsonl'
    print(output_name)
    model_name = args.model_name
    if 'OpenELM' in model_name:
        tok_model_name = "Xenova/OpenELM-270M-Instruct"
    else:
        tok_model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(tok_model_name, padding_side='left', model_max_length=1024, trust_remote_code=True)
    #if 'OpenELM' in tok_model_name:
    #    tokenizer.pad_token = tokenizer.bos_token
    #else:
    tokenizer.pad_token = tokenizer.eos_token
    #tokenizer.model_max_length = 2048
    
    ent_tok_name = 'EleutherAI/pythia-6.9b-deduped'
    if 'OPT_' in AP_entropy_model_path:
        ent_tok_name = 'facebook/opt-6.7b'
    elif 'GPT2_' in AP_entropy_model_path:
        ent_tok_name = 'openai-community/gpt2-xl'
    print("tokenizer entropy", ent_tok_name)

    tokenizer_ent = AutoTokenizer.from_pretrained(ent_tok_name, padding_side='left', model_max_length=1024, trust_remote_code=True)
    tokenizer_ent.pad_token = tokenizer_ent.eos_token
    tokenizer_ent.model_max_length = 2048
    
    device = torch.device("cuda:"+str(args.cuda_idx))
    #device = torch.device("cpu")
    
    logits_processor_i = None
    if args.sample_method == 'decay':
        logits_processor_i = FactualTopPLogitsWarper(top_p = args.p, top_p_decay_rate=args.decay_temperature, top_p_lower_cap = 0.3, reset_patience = 5, filter_value = -float("Inf"), min_tokens_to_keep=1)
    elif args.sample_method == 'decay_period':
        logits_processor_i = PeriodFactualTopPLogitsWarper(top_p = args.p, top_p_decay_rate=args.decay_temperature, top_p_lower_cap = 0.3, tokenizer=tokenizer, filter_value = -float("Inf"), min_tokens_to_keep=1)
    #elif args.sample_method == 'fecut_topp':
    #    logits_processor_i = FECutLogitsWarper(decay_temperature = args.decay_temperature, final_entropy_model_path = args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = args.cut_sample_sub_method, filter_value = -float("Inf"), min_tokens_to_keep=1, device=device)
    elif args.sample_method == 'fe_topp':
        logits_processor_i = FETopPLogitsWarper(top_p = args.p, decay_temperature = args.decay_temperature, final_entropy_model_path = args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = args.sample_sub_method, window_size = args.window_size, filter_value = -float("Inf"), min_tokens_to_keep=1, device=device)
    elif args.sample_method == 'fe_topk':
        logits_processor_i = FETopPLogitsWarper(top_p = args.p, decay_temperature = args.decay_temperature, final_entropy_model_path = args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = args.sample_sub_method, window_size = args.window_size, filter_value = -float("Inf"), min_tokens_to_keep=1, use_top_k = True, device=device)
    elif args.sample_method == 'fe_topp_period':
        logits_processor_i = FETopPLogitsWarper(top_p = 1, decay_temperature = args.decay_temperature, final_entropy_model_path = args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = args.sample_sub_method, window_size = args.window_size, sent_idx_decay_rate = args.p, filter_value = -float("Inf"), min_tokens_to_keep=1, device=device)
    elif args.sample_method == 'fe_CD':
        logits_processor_i = FETopPLogitsWarper(top_p = args.p, decay_temperature = args.decay_temperature, final_entropy_model_path = args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = args.sample_sub_method, window_size = args.window_size, filter_value = -float("Inf"), min_tokens_to_keep=1, student_model_name='EleutherAI/pythia-70m-deduped', use_CD_alpha= True, use_AP=False, device=device, use_log_softmax=True)
    elif args.sample_method == 'fe_CD_topp':
        if 'pythia' in args.model_name:
            student_model_name='EleutherAI/pythia-70m-deduped'
        elif 'opt' in args.model_name:
            student_model_name='facebook/opt-125m'
        elif 'gpt2' in args.model_name:
            student_model_name='openai-community/gpt2'
        assert 'open_llama' not in args.model_name #open_llama does not have small enough student model in the family
        print(student_model_name)
        #student_use_gen_tokenizer = True
        #if 'open_llama' in args.model_name:
        #    student_use_gen_tokenizer = False #The student model uses tokenizer_ent
        logits_processor_i = FETopPLogitsWarper(top_p = args.p, decay_temperature = args.decay_temperature, final_entropy_model_path = args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = args.sample_sub_method, window_size = args.window_size, filter_value = -float("Inf"), min_tokens_to_keep=1, student_model_name=student_model_name, use_CD_alpha= False, use_AP=False, device=device, use_log_softmax=True)
    elif args.sample_method == 'fe_AP_topp':
        logits_processor_i = FETopPLogitsWarper(top_p = args.p, decay_temperature = args.decay_temperature, final_entropy_model_path = AP_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = args.sample_sub_method, window_size = args.window_size, filter_value = -float("Inf"), min_tokens_to_keep=1, student_model_name=args.final_entropy_model_path, use_CD_alpha= False, use_AP=True, device=device)
    elif args.sample_method == 'AP_topk':
        logits_processor_i = APTopKLogitsWarper(top_k = args.p, student_model_name=args.final_entropy_model_path, filter_value = -float("Inf"), device=device)
    elif args.sample_method == 'AP_topp':
        logits_processor_i = APTopPALogitsWarper(top_p = args.p, student_model_name=args.final_entropy_model_path, filter_value = -float("Inf"), device=device, use_alpha=False, temperature = args.decay_temperature, use_log_softmax=True)
    elif args.sample_method == 'AP_toppk20':
        logits_processor_i = APTopPALogitsWarper(top_p = args.p, student_model_name=args.final_entropy_model_path, filter_value = -float("Inf"), device=device, use_alpha=False, temperature = args.decay_temperature, top_k=20, use_log_softmax=True)
    elif args.sample_method == 'AP_CD_toppk20':
        if 'pythia' in args.model_name:
            CD_model_name='EleutherAI/pythia-70m-deduped'
            CD_inv_temp = 0.5
        elif 'opt' in args.model_name:
            CD_model_name='facebook/opt-125m'
            CD_inv_temp = 0.25
        elif 'gpt2' in args.model_name:
            CD_model_name='openai-community/gpt2'
            CD_inv_temp = 0.5
        logits_processor_i = APCDTopPALogitsWarper(top_p = args.p, student_model_name=args.final_entropy_model_path, filter_value = -float("Inf"), device=device, use_alpha=False, temperature = args.decay_temperature, top_k=20, CD_model_name = CD_model_name, CD_inv_temp = CD_inv_temp)
    elif args.sample_method == 'AP_topa':
        logits_processor_i = APTopPALogitsWarper(top_p = args.p, student_model_name=args.final_entropy_model_path, filter_value = -float("Inf"), device=device, use_alpha=True, temperature = args.decay_temperature, use_log_softmax=True)
    elif args.sample_method == 'AP_topak20':
        logits_processor_i = APTopPALogitsWarper(top_p = args.p, student_model_name=args.final_entropy_model_path, filter_value = -float("Inf"), device=device, use_alpha=True, temperature = args.decay_temperature, top_k=20, use_log_softmax=True)
    elif args.sample_method == 'CD':
        tokenizer_st = None
        #if 'open_llama' in args.model_name:
        #    assert 'pythia' in args.final_entropy_model_path
        #    tokenizer_st = tokenizer_ent
            #assume that we use pythia
        logits_processor_i = ContrastiveDecodingLogitsWarper(alpha = args.p, temperature = args.decay_temperature, student_model_name=args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_st=tokenizer_st, filter_value = -float("Inf"), device=device, use_alpha=True)
    elif args.sample_method == 'CDk20':
        tokenizer_st = None
        logits_processor_i = ContrastiveDecodingLogitsWarper(alpha = args.p, temperature = args.decay_temperature, student_model_name=args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_st=tokenizer_st, filter_value = -float("Inf"), device=device, use_alpha=True, top_k=20, use_log_softmax=True)
    elif args.sample_method == 'CD_topp':
        tokenizer_st = None
        logits_processor_i = ContrastiveDecodingLogitsWarper(alpha = args.p, temperature = args.decay_temperature, student_model_name=args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_st=tokenizer_st, filter_value = -float("Inf"), device=device, use_alpha=False, use_log_softmax=True)
    elif args.sample_method == 'CD_toppk20':
        tokenizer_st = None
        logits_processor_i = ContrastiveDecodingLogitsWarper(alpha = args.p, temperature = args.decay_temperature, student_model_name=args.final_entropy_model_path, tokenizer=tokenizer, tokenizer_st=tokenizer_st, filter_value = -float("Inf"), device=device, use_alpha=False, top_k=20, use_log_softmax=True)
    elif args.sample_method == 'CD_topk':
        logits_processor_i = CDTopKLogitsWarper(top_k = args.p, temperature = args.decay_temperature, student_model_name=args.final_entropy_model_path, filter_value = -float("Inf"), device=device, use_log_softmax=True)

    logits_processor = LogitsProcessorList()
    logits_processor.append(logits_processor_i)

    #model = AutoModelWithLMHead.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
    #if 'Qwen' in model_name:
    #    model.half()
    #elif 'OpenELM' in model_name:
    #    model = model.half()
    #device = torch.cuda.device(args.cuda_idx)
    model.eval()
    model.to(device)

    with open(args.input_file_name) as f_in:
        fact_dataset = factual_dataset(f_in, tokenizer)
    print( fact_dataset.text_tensor["input_ids"].size(1) )
    #args.max_len += fact_dataset.text_tensor["input_ids"].size(1)
    max_len = args.max_len + fact_dataset.text_tensor["input_ids"].size(1)

    dataloader = DataLoader(fact_dataset, batch_size=args.batch_size, shuffle=False)
    
    gen_time_list = []
    output_arr = [ [] for i in range(args.num_return_sequences) ]
    for idx, batch in enumerate(dataloader):
        print(args.cuda_idx, idx / len(dataloader))
        #print(batch)
        input_ids, attention_mask = batch
        t1 = datetime.now()
        if args.sample_method == 'topp':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.decay_temperature, top_p=args.p, do_sample=True, max_new_tokens = args.max_len)
        elif args.sample_method == 'toppk20':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.decay_temperature, top_p=args.p, top_k=20, do_sample=True, max_new_tokens = args.max_len)
        elif args.sample_method == 'topk':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature,  top_k=int(args.p), do_sample=True, max_new_tokens = args.max_len)
                #max_length=tokenizer.model_max_length,
        elif args.sample_method == 'CS':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature, penalty_alpha=args.decay_temperature, top_k=int(args.p), max_new_tokens = args.max_len)
        elif args.sample_method == 'eta':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature, eta_cutoff=args.p, do_sample=True, max_new_tokens = args.max_len)
        elif args.sample_method == 'typical':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature, typical_p=args.p, do_sample=True, max_new_tokens = args.max_len)
        elif args.sample_method == 'decay' or args.sample_method == 'decay_period':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature, logits_processor=logits_processor, do_sample=True, max_new_tokens = args.max_len)
        elif args.sample_method == 'fe_topp' or args.sample_method == 'fe_topp_period' or args.sample_method == 'fe_topk' or args.sample_method == 'fecut_topp' or args.sample_method == 'fe_CD' or args.sample_method == 'fe_CD_topp':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature, logits_processor=logits_processor, do_sample=True, max_new_tokens = args.max_len )
        elif args.sample_method == 'AP_topk' or args.sample_method == 'AP_topp' or args.sample_method == 'AP_toppk20' or args.sample_method == 'AP_CD_toppk20' or args.sample_method == 'AP_topa' or args.sample_method == 'fe_AP_topp' or args.sample_method == 'AP_topak20':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature, logits_processor=logits_processor, do_sample=True, max_new_tokens = args.max_len )
        elif args.sample_method == 'CD' or args.sample_method == 'CD_topk' or args.sample_method == 'CD_topp' or args.sample_method == 'CD_toppk20' or args.sample_method == 'CDk20':
            output_sequences = model.generate(input_ids=input_ids.to(device), attention_mask = attention_mask.to(device), pad_token_id=tokenizer.eos_token_id, num_return_sequences=args.num_return_sequences, temperature=args.temperature, logits_processor=logits_processor, do_sample=True, max_new_tokens = args.max_len )
        t2 = datetime.now()

        gen_time_list.append( (t2-t1).total_seconds()  )

        #sys.exit(0)

        input_len = input_ids.size(-1)
        input_prompt = output_sequences[:,:input_len]
        output_con = output_sequences[:,input_len:]
        prompt_text = tokenizer.batch_decode(input_prompt, skip_special_tokens=True)
        output_text = tokenizer.batch_decode(output_con, skip_special_tokens=True)
        bsz = int( len(output_text) / args.num_return_sequences)
        for i in range(args.num_return_sequences):
            for j in range(bsz):
                #print(prompt_text[j*args.num_return_sequences + i])
                #print(output_text[j*args.num_return_sequences + i])
                output_arr[i].append( (prompt_text[j*args.num_return_sequences + i], output_text[j*args.num_return_sequences + i]) )

    print('avg_inference time: ', np.mean(gen_time_list))
    print('std_inference time: ', np.std(gen_time_list))

    for i in range( args.num_return_sequences ):
        output_list = output_arr[i]
        with open(output_name.format(args.num_existing_seeds+i+1) , 'w') as f_out:
            save_gen(f_out, output_list, fact_dataset.id_list, fact_dataset.ref_list)
    print(output_name)

if __name__ == "__main__":
    args = parse_args()
    main(args)
