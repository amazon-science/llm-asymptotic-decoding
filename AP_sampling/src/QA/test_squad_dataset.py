import sys
sys.path.append('/mnt/efs/Haw-Shiuan/AP_sampling/src')
import numpy as np

try:
    from model_mlp_logit import GPTNeoXForLogitCorrection, OPTForLogitCorrection, Qwen2ForLogitCorrection
except:
    from model_mlp_logit import GPTNeoXForLogitCorrection, OPTForLogitCorrection

from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch
import json
from torch import nn

import pandas as pd

cuda_idx = 0
#cuda_idx = 2
#cuda_idx = 4
#cuda_idx = 6

device_llm = 'cuda:'+str(cuda_idx)
device_tlm = 'cuda:'+str(cuda_idx)
device_ap = 'cuda:'+str(cuda_idx)
device_hlm = 'cuda:'+str(cuda_idx+1)

#device_llm = 'cuda:0'
#device_tlm = 'cuda:1'
#device_ap = 'cuda:4'
#device_hlm = 'cuda:5'

small_prob_for_not_inf = 1e-2

large_model_name = 'EleutherAI/pythia-6.9b-deduped'
#large_model_name = 'facebook/opt-6.7b'
#large_model_name = 'Qwen/Qwen1.5-4b'

if 'pythia' in large_model_name:
    huge_model_name = 'EleutherAI/pythia-12b-deduped'
    small_model_name = 'EleutherAI/pythia-70m-deduped'
    ap_model_name = "models/prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4"
elif 'opt' in large_model_name:
    huge_model_name = 'facebook/opt-13b'
    small_model_name = 'facebook/opt-125m'
    ap_model_name = "models/prob_opt_wiki_ext_new_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_04_logit_exp_decay_lr-4"
elif 'Qwen1.5-4b' in large_model_name:
    huge_model_name = 'Qwen/Qwen1.5-7b'
    small_model_name = 'Qwen/Qwen1.5-0.5b'
    ap_model_name = "models/prob_Qwen_4b_wiki_ext_new_1e6_500M_bsz_32_e5_only_top_last_w_10_l1_reg_w_1_logit_exp_decay_lr-4"

#max_len = 256
#max_len = 512
max_len = 1024


#input_file = "outputs/multirc/multirc_train.json"
#output_file = "outputs/multirc/final_Pythia_logsoftmax_multirc_train_topk20_new.csv"
#output_file = "outputs/multirc/final_OPT_logsoftmax_multirc_train_topk20.csv"
#output_file = "outputs/multirc/final_Qwen_w1_logsoftmax_multirc_train_topk20.csv"

#input_file = "outputs/squad/squad_val.json"
#output_file = "outputs/squad/final_Pythia_logsoftmax_squad_val_topk20_new.csv"
#output_file = "outputs/squad/final_OPT_logsoftmax_squad_val_topk20.csv"
#output_file = "outputs/squad/final_Qwen_w1_logsoftmax_squad_val_topk20.csv"

#input_file = "outputs/squad/squad_val_no_pass.json"
#output_file = "outputs/squad/final_Pythia_logsoftmax_squad_val_no_pass_topk20_new.csv"
#output_file = "outputs/squad/final_OPT_logsoftmax_squad_val_no_pass_topk20.csv"
#output_file = "outputs/squad/final_Qwen_w1_logsoftmax_squad_val_no_pass_topk20.csv"

input_file = "outputs/lambda/openai_test.json"
output_file = "outputs/lambda/final_Pythia_logsoftmax_lambda_topk20_1e-2_new.csv"
#output_file = "outputs/lambda/final_OPT_logsoftmax_lambda_topk20_1e-2.csv"
#output_file = "outputs/lambda/final_Qwen_w1_logsoftmax_lambda_topk20_1e-2.csv"

print(input_file)

inv_temp_cd_arr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
#inv_temp_cd_arr = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#inv_temp_cd_arr = [0.05, 0.1, 0.125, 0.15, 0.25, 0.5]
#inv_temp_cd_arr = [0.0125,0.025, 0.05, 0.1, 0.125, 0.15]
#inv_temp_cd_arr = [0.15, 0.2, 0.25, 0.3, 0.35, 0.5]

inv_temp_ap_arr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 1.0]
#inv_temp_ap_arr = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#inv_temp_ap_arr = [0.1, 0.2, 0.25, 0.3, 0.5, 1]
#inv_temp_ap_arr = [0.025, 0.05, 0.1, 0.2, 0.25, 0.3]
#inv_temp_ap_arr = [0.25, 0.3, 0.35, 0.4, 0.5, 1]

batch_size = 1
#batch_size = 2

#vis_output = False
vis_output = True
if vis_output:
    assert batch_size == 1
    vis_cd_temp = 0.4
    vis_ap_temp = 0.6


assert len(inv_temp_cd_arr) == len(inv_temp_ap_arr)
assert len(inv_temp_cd_arr) > 10 # in order to squeeze global information into dataframe at the end


topk_th = 20

tokenizer = AutoTokenizer.from_pretrained(large_model_name, truncation_side='left')
tokenizer.pad_token = tokenizer.eos_token
tokenizer.model_max_length = max_len

model_logit_decay = False
if '_ld_' in ap_model_name:
    model_logit_decay = True

poly_degree = 10
if '_exp_decay_' in ap_model_name:
    decay_function='exp'
elif '_logistic_decay_' in ap_model_name:
    decay_function='logistic'
elif 'scaled_a' in ap_model_name:
    decay_function='scaled_poly'
    poly_degree = extract_param(ap_model_name, '_a', '_')
else:
    decay_function='poly'
    poly_degree = extract_param(ap_model_name, '_a', '_')

if 'prob_Qwen' in ap_model_name:
    log_model_size = [19.5469252164,21.1456953943,21.9934232469]
    model_ap = Qwen2ForLogitCorrection.from_pretrained(ap_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device_ap)
elif 'prob_opt' in ap_model_name:
    log_model_size = [18.2771614, 19.5373201, 20.9161984, 21.6486751, 22.5877428]
    model_ap = OPTForLogitCorrection.from_pretrained(ap_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device_ap)
else:
    log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
    model_ap = GPTNeoXForLogitCorrection.from_pretrained(ap_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device_ap)

model_ap.eval()

def load_squad_file(f_in):
    output_list = []
    for line in f_in:
        input_dict = json.loads(line)
        org_left = input_dict['prompt']
        pos_left_text = input_dict['prompt'] + input_dict['answer']

        output_list.append( [org_left, pos_left_text] )
    return output_list

class squad_dataset(Dataset):
    def __init__(self, examples, tokenizer, max_len):
        self.org_left_len_arr = []
        self.pos_left_len_arr = []
        pos_left_text_arr = []
        for example in examples:
            org_left, pos_left_text = example
            org_left_tok = tokenizer.tokenize(org_left)
            pos_left_tok = tokenizer.tokenize(pos_left_text)
            left_len = len(org_left_tok)
            pos_left_len = len(pos_left_tok)
            if 'opt' in large_model_name: #plus 1 because the tokenizer() in OPT would add a start of sequence token at the beginning
                left_len = left_len + 1
                pos_left_len = pos_left_len + 1

            if pos_left_len >=  max_len:
                continue
            self.org_left_len_arr.append(left_len)
            self.pos_left_len_arr.append(pos_left_len)

            pos_left_text_arr.append(pos_left_text)
        pos_left_text_tok = tokenizer.tokenize(pos_left_text_arr[0])
        if 'opt' in large_model_name:
            print(pos_left_text_tok[:self.org_left_len_arr[0]-1])
            print(pos_left_text_tok[self.org_left_len_arr[0]-1:])
        else:
            print(pos_left_text_tok[:self.org_left_len_arr[0]])
            print(pos_left_text_tok[self.org_left_len_arr[0]:])
        #self.pos_left_text_tensor = tokenizer(pos_left_text_arr, padding=True, truncation=True, return_tensors='pt', max_length = tokenizer.model_max_length)
        self.pos_left_text_tensor = tokenizer(pos_left_text_arr, padding='max_length', return_tensors='pt', max_length = max_len)

    def __len__(self):
        return self.pos_left_text_tensor["input_ids"].size(0)

    def __getitem__(self, idx):
        return self.org_left_len_arr[idx] - 1, self.pos_left_len_arr[idx] - 1, self.pos_left_text_tensor["input_ids"][idx,:]

with open(input_file) as f_in:
    output_list = load_squad_file(f_in)
dataset = squad_dataset(output_list, tokenizer, max_len)
#dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

model_hlm = AutoModelForCausalLM.from_pretrained(huge_model_name, torch_dtype=torch.float16)
model_hlm = model_hlm.to(device_hlm)

model_llm = AutoModelForCausalLM.from_pretrained(large_model_name, torch_dtype=torch.float16)
model_llm = model_llm.to(device_llm)

model_st = AutoModelForCausalLM.from_pretrained(small_model_name)
model_st = model_st.to(device_tlm)

model_hlm.eval()
model_llm.eval()
model_st.eval()

ppl_llm_arr = []
#ppl_llm_all_arr = []
mrr_llm_arr = []
#mrr_llm_all_arr = []
ppl_hlm_arr = []
mrr_hlm_arr = []
ppl_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
ppl_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}
mrr_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
mrr_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}
#ppl_cd_all_dict = {inv_temp: [] for inv_temp in inv_temp_arr}
#ppl_ap_all_dict = {inv_temp: [] for inv_temp in inv_temp_arr}
#mrr_cd_all_dict = {inv_temp: [] for inv_temp in inv_temp_arr}
#mrr_ap_all_dict = {inv_temp: [] for inv_temp in inv_temp_arr}

def vis_top_w_pred(prob_topk, text_pos, prefix):
    top_k_out = torch.topk(prob_topk, topk_th)
    top_k_value = top_k_out[0][text_pos, :]
    top_k_w_idx = top_k_out[1][text_pos, :]
    #print(top_k_w_idx)
    #print(top_k_value)
    #top_token = [tokenizer.convert_ids_to_tokens(top_k_w_idx[i]) for i in range(len(top_k_w_idx))]
    top_token = tokenizer.convert_ids_to_tokens(top_k_w_idx)
    print( prefix, list(zip( top_token, top_k_value.tolist() ) ))


def contrastive_prob(scores, scores_org, scores_cd_small, inv_temp, label):
    scores_cd = scores.clone()
    scores_cd[:,:,:scores_cd_small.size(-1)] = scores_cd[:,:,:scores_cd_small.size(-1)] - inv_temp * scores_cd_small
    #scores_cd = scores_cd.masked_fill(indices_to_remove_topk, -float("Inf"))
    #prob_cd_dict[inv_temp] = scores_cd.softmax(dim=-1)
    scores_cd_all = scores_org.clone()
    scores_cd_all[:,:,:scores_cd_small.size(-1)] = scores_cd_all[:,:,:scores_cd_small.size(-1)] - inv_temp * scores_cd_small
    #prob_cd_all_dict[inv_temp] = scores_cd_all.softmax(dim=-1)
    #gt_logits_cd = torch.gather(scores_cd_all, dim=-1, index=label)
    #gt_rank_cd = (gt_logits_cd <= scores_cd_all).to(torch.int32).sum(dim=-1)
    gt_logits_cd = torch.gather(scores_cd, dim=-1, index=label)
    gt_rank_cd = (gt_logits_cd <= scores_cd).to(torch.int32).sum(dim=-1)
    return scores_cd.softmax(dim=-1), scores_cd_all.softmax(dim=-1), 1 / gt_rank_cd


for idx, batch in enumerate(dataloader):
    print(idx / len(dataloader))

    with torch.no_grad():
        org_left_len, pos_left_len, input_ids = batch 

        label = input_ids[:, 1:].unsqueeze(dim=-1).to(model_ap.device)

        outputs_llm = model_llm(input_ids.to(model_llm.device), return_dict = True)
        scores_org = nn.functional.log_softmax(outputs_llm.logits[:,:-1,:].to(model_ap.device), dim=-1)
        
        outputs_hlm = model_hlm(input_ids.to(model_hlm.device), return_dict = True)
        scores_hlm = nn.functional.log_softmax(outputs_hlm.logits[:,:-1,:].to(model_ap.device), dim=-1)
        
        outputs_students = model_ap(input_ids.to(model_ap.device), return_dict = True)
        #scores_ap_small = nn.functional.log_softmax(outputs_students.logits[:,:-1,:],dim=-1)
        scores_ap_small = nn.functional.log_softmax(outputs_students.logits[:,:-1,:],dim=-1)

        outputs_CD = model_st(input_ids.to(model_st.device), return_dict = True)
        #scores_cd_small = outputs_CD.logits[:,:-1,:].to(model_ap.device)
        scores_cd_small = nn.functional.log_softmax(outputs_CD.logits[:,:-1,:].to(model_ap.device), dim=-1)
        
        gt_logits = torch.gather(scores_org, dim=-1, index=label)
        gt_rank = (gt_logits <= scores_org).to(torch.int32).sum(dim=-1)
    
        indices_to_remove_topk = scores_org < torch.topk(scores_org, topk_th)[0][..., -1, None]
        scores = scores_org.clone().masked_fill(indices_to_remove_topk, -float("Inf"))
        
        indices_to_remove_topk_hlm = scores_hlm < torch.topk(scores_hlm, topk_th)[0][..., -1, None]
        scores_hlm_masked = scores_hlm.clone().masked_fill(indices_to_remove_topk_hlm, -float("Inf"))
        #if scores_hlm.size(-1) != indices_to_remove_topk.size(-1):
        #    indices_to_remove_topk_ext = torch.full_like(scores_hlm, True, dtype=torch.bool)
        #    indices_to_remove_topk_ext[:,:,:indices_to_remove_topk.size(-1)] = indices_to_remove_topk
        #else:
        #    indices_to_remove_topk_ext = indices_to_remove_topk

        #scores_hlm_masked = scores_hlm.clone().masked_fill(indices_to_remove_topk_ext, -float("Inf"))
        
        #gt_logits_hlm_org = torch.gather(scores_hlm, dim=-1, index=label)
        #gt_rank_hlm_org = (gt_logits_hlm_org <= scores_hlm).to(torch.int32).sum(dim=-1)
        
        gt_logits_hlm = torch.gather(scores_hlm_masked, dim=-1, index=label)
        gt_rank_hlm = (gt_logits_hlm <= scores_hlm_masked).to(torch.int32).sum(dim=-1)

        prob_topk = scores.softmax(dim=-1)
        prob_hlm_topk = scores_hlm_masked.softmax(dim=-1)
        #prob_org = scores_org.softmax(dim=-1)
        prob_cd_dict = {}
        prob_ap_dict = {}
        prob_cd_all_dict = {}
        prob_ap_all_dict = {}
        mrr_cd_raw_dict = {}
        mrr_ap_raw_dict = {}
        #print(scores)
        #print(scores_org)
        for inv_temp in inv_temp_cd_arr:
            prob_cd_dict[inv_temp], prob_cd_all_dict[inv_temp], mrr_cd_raw_dict[inv_temp] = contrastive_prob(scores, scores_org, scores_cd_small, inv_temp, label)
        for inv_temp in inv_temp_ap_arr:
            prob_ap_dict[inv_temp], prob_ap_all_dict[inv_temp], mrr_ap_raw_dict[inv_temp] = contrastive_prob(scores, scores_org, scores_ap_small, inv_temp, label)
            #print(prob_cd_all_dict[inv_temp])
            #print(prob_ap_all_dict[inv_temp])
            #scores_ap = scores.clone()
            #scores_ap[:,:,:scores_ap_small.size(-1)] = scores_ap[:,:,:scores_ap_small.size(-1)] - inv_temp * scores_ap_small
            #scores_ap = scores_ap.masked_fill(indices_to_remove_topk, -float("Inf"))
            #prob_ap_dict[inv_temp] = scores_ap.softmax(dim=-1)

        #top_w = torch.topk(scores, 20, dim=-1)[1]

        #for i in range(batch_size):
        for i in range(input_ids.size(0)):
            #print(label[i,org_left_len[i]])
            #print(tokenizer.decode(label[i,org_left_len[i]]) )
            #print( tokenizer.decode( top_w[i, org_left_len[i],:] ))
            #print( gt_rank[i, org_left_len[i]:pos_left_len[i] ] <= topk_th )
            #all_in_topk = torch.all( gt_rank[i, org_left_len[i]:pos_left_len[i] ] <= topk_th )
            
            all_in_topk = torch.all( gt_rank[i, org_left_len[i]:pos_left_len[i] ] <= topk_th ) and torch.all( gt_rank_hlm[i, org_left_len[i]:pos_left_len[i] ] <= topk_th )
            #all_in_topk = True
            target_label = label[i, org_left_len[i]:pos_left_len[i], :]
            
            #mrr_all = ( 1 / gt_rank[i, org_left_len[i]:pos_left_len[i] ] ).mean().item()
            #ppl_all_org = -torch.log( torch.gather( prob_org[i, org_left_len[i]:pos_left_len[i], : ], dim=-1, index = target_label ) + small_prob_for_not_inf ).mean().item()
            #ppl_llm_all_arr.append(ppl_all_org)
            #mrr_llm_all_arr.append(mrr_all)

            if all_in_topk:
                ppl_org = -torch.log( torch.gather( prob_topk[i, org_left_len[i]:pos_left_len[i], : ], dim=-1, index = target_label ) + small_prob_for_not_inf ).mean().item()
                ppl_llm_arr.append(ppl_org)
                mrr_all = ( 1 / gt_rank[i, org_left_len[i]:pos_left_len[i] ] ).mean().item()
                mrr_llm_arr.append(mrr_all)
                
                ppl_hlm = -torch.log( torch.gather( prob_hlm_topk[i, org_left_len[i]:pos_left_len[i], : ], dim=-1, index = target_label ) + small_prob_for_not_inf ).mean().item()
                ppl_hlm_arr.append(ppl_hlm)
                mrr_hlm = ( 1 / gt_rank_hlm[i, org_left_len[i]:pos_left_len[i] ] ).mean().item()
                mrr_hlm_arr.append(mrr_hlm) 
            for inv_temp in prob_cd_dict:
                #ppl_all_cd = -torch.log( torch.gather( prob_cd_all_dict[inv_temp][i, org_left_len[i]:pos_left_len[i], : ], dim=-1, index = target_label ) + small_prob_for_not_inf  ).mean().item()
                #print(ppl_all_cd)
                #ppl_cd_all_dict[inv_temp].append(ppl_all_cd)
                #mrr_cd_all = mrr_cd_raw_dict[inv_temp][i, org_left_len[i]:pos_left_len[i]].mean().item()
                #mrr_cd_all_dict[inv_temp].append(mrr_cd_all)

                #ppl_all_ap = -torch.log( torch.gather( prob_ap_all_dict[inv_temp][i, org_left_len[i]:pos_left_len[i], : ], dim=-1, index = target_label ) + small_prob_for_not_inf  ).mean().item()
                #print(ppl_all_ap)
                #ppl_ap_all_dict[inv_temp].append(ppl_all_ap)
                #mrr_ap_all = mrr_ap_raw_dict[inv_temp][i, org_left_len[i]:pos_left_len[i]].mean().item()
                #mrr_ap_all_dict[inv_temp].append(mrr_ap_all)
                if all_in_topk:
                    ppl_cd = -torch.log( torch.gather( prob_cd_dict[inv_temp][i, org_left_len[i]:pos_left_len[i], : ], dim=-1, index = target_label ) + small_prob_for_not_inf ).mean().item()
                    ppl_cd_dict[inv_temp].append(ppl_cd)
                    mrr_cd_all = mrr_cd_raw_dict[inv_temp][i, org_left_len[i]:pos_left_len[i]].mean().item()
                    mrr_cd_dict[inv_temp].append(mrr_cd_all)
            for inv_temp in prob_ap_dict:
                if all_in_topk:
                    ppl_ap = -torch.log( torch.gather( prob_ap_dict[inv_temp][i, org_left_len[i]:pos_left_len[i], : ], dim=-1, index = target_label ) + small_prob_for_not_inf ).mean().item()
                    ppl_ap_dict[inv_temp].append(ppl_ap)
                    mrr_ap_all = mrr_ap_raw_dict[inv_temp][i, org_left_len[i]:pos_left_len[i]].mean().item()
                    mrr_ap_dict[inv_temp].append(mrr_ap_all)
            
            if vis_output and all_in_topk and  ppl_cd_dict[vis_cd_temp][-1] > ppl_ap_dict[vis_ap_temp][-1] and mrr_cd_dict[vis_cd_temp][-1] < mrr_ap_dict[vis_ap_temp][-1]:
            #if vis_output and all_in_topk and  ppl_llm_arr[-1] > ppl_cd_dict[vis_cd_temp][-1] and ppl_cd_dict[vis_cd_temp][-1] > ppl_ap_dict[vis_ap_temp][-1] and mrr_cd_dict[vis_cd_temp][-1] < mrr_ap_dict[vis_ap_temp][-1]:
            #if vis_output and all_in_topk: 
                print('vis ', output_list[idx][1])
                print(ppl_llm_arr[-1], ppl_cd_dict[vis_cd_temp][-1], ppl_ap_dict[vis_ap_temp][-1], mrr_cd_dict[vis_cd_temp][-1], mrr_ap_dict[vis_ap_temp][-1])
                vis_top_w_pred(prob_topk[i,:,:], org_left_len[i], 'llm')
                vis_top_w_pred(prob_cd_dict[vis_cd_temp][i,:,:], org_left_len[i], 'cd')
                vis_top_w_pred(prob_ap_dict[vis_ap_temp][i,:,:], org_left_len[i], 'ap')

        if idx % 10 == 0 or idx == len(dataloader)-1:
            print(len(ppl_llm_arr))
            print('topk20 org ppl', np.mean(ppl_llm_arr), ' mrr', np.mean(mrr_llm_arr) )
            print('topk20 hlm ppl', np.mean(ppl_hlm_arr), ' mrr', np.mean(mrr_hlm_arr) )
            ppl_cd_results = [ (inv_temp, np.mean(ppl_cd_dict[inv_temp])) for inv_temp in ppl_cd_dict ]
            mrr_cd_results = [ (inv_temp, np.mean(mrr_cd_dict[inv_temp])) for inv_temp in mrr_cd_dict ]
            best_ppl_cd = min(ppl_cd_results,key=lambda x:x[1])
            best_mrr_cd = max(mrr_cd_results,key=lambda x:x[1])
            ppl_ap_results = [ (inv_temp, np.mean(ppl_ap_dict[inv_temp])) for inv_temp in ppl_ap_dict ]
            mrr_ap_results = [ (inv_temp, np.mean(mrr_ap_dict[inv_temp])) for inv_temp in mrr_ap_dict ]
            best_ppl_ap = min(ppl_ap_results,key=lambda x:x[1])
            best_mrr_ap = max(mrr_ap_results,key=lambda x:x[1])
            print('topk20 cd best ppl', '{} {}'.format(best_ppl_cd[1], best_ppl_cd[0]), 'best mrr', '{} {}'.format(best_mrr_cd[1], best_mrr_cd[0]), 'ppl', ppl_cd_results, ' mrr', mrr_cd_results)
            print('topk20 ap best ppl', '{} {}'.format(best_ppl_ap[1], best_ppl_ap[0]), 'best mrr', '{} {}'.format(best_mrr_ap[1], best_mrr_ap[0]), 'ppl', ppl_ap_results, ' mrr', mrr_ap_results)
        #print('all org ppl', np.mean(ppl_llm_all_arr), ' mrr', np.mean(mrr_llm_all_arr)  )
        #print('all cd ppl', [ (inv_temp, np.mean(ppl_cd_all_dict[inv_temp])) for inv_temp in ppl_cd_all_dict ], ' mrr', [ (inv_temp, np.mean(mrr_cd_all_dict[inv_temp])) for inv_temp in mrr_cd_all_dict ] )
        #print('all ap ppl', [ (inv_temp, np.mean(ppl_ap_all_dict[inv_temp])) for inv_temp in ppl_ap_all_dict ], ' mrr', [ (inv_temp, np.mean(mrr_ap_all_dict[inv_temp])) for inv_temp in mrr_ap_all_dict ] )

print('topk20 std org ppl', np.std(ppl_llm_arr), ' mrr', np.std(mrr_llm_arr) )
print('topk20 std hlm ppl', np.std(ppl_hlm_arr), ' mrr', np.std(mrr_hlm_arr) )
print('topk20 std cd ppl', [ (inv_temp, np.std(ppl_cd_dict[inv_temp])) for inv_temp in ppl_cd_dict ], ' mrr', [ (inv_temp, np.std(mrr_cd_dict[inv_temp])) for inv_temp in mrr_cd_dict ]  )
print('topk20 std ap ppl', [ (inv_temp, np.std(ppl_ap_dict[inv_temp])) for inv_temp in ppl_ap_dict ], ' mrr', [ (inv_temp, np.std(mrr_ap_dict[inv_temp])) for inv_temp in mrr_ap_dict ] )
print(len(ppl_llm_arr))
print(len(ppl_llm_arr) / dataset.pos_left_text_tensor["input_ids"].size(0))
print(input_file)
print(large_model_name)

cd_ppl_results = [ (inv_temp, np.mean(ppl_cd_dict[inv_temp]), np.std(ppl_cd_dict[inv_temp])) for inv_temp in ppl_cd_dict ]
cd_mrr_results = [ (inv_temp, np.mean(mrr_cd_dict[inv_temp]), np.std(mrr_cd_dict[inv_temp])) for inv_temp in mrr_cd_dict ]
ap_ppl_results = [ (inv_temp, np.mean(ppl_ap_dict[inv_temp]), np.std(ppl_ap_dict[inv_temp])) for inv_temp in ppl_ap_dict ]
ap_mrr_results = [ (inv_temp, np.mean(mrr_ap_dict[inv_temp]), np.std(mrr_ap_dict[inv_temp])) for inv_temp in mrr_ap_dict ]

ap_ppl_temp_arr, ap_ppl_mean_arr, ap_ppl_std_arr  = zip(* ap_ppl_results )
cd_ppl_temp_arr, cd_ppl_mean_arr, cd_ppl_std_arr  = zip(* cd_ppl_results )
ap_mrr_temp_arr, ap_mrr_mean_arr, ap_mrr_std_arr  = zip(* ap_mrr_results )
cd_mrr_temp_arr, cd_mrr_mean_arr, cd_mrr_std_arr  = zip(* cd_mrr_results )

global_stats = [np.mean(ppl_llm_arr), np.std(ppl_llm_arr),  np.mean(mrr_llm_arr), np.std(mrr_llm_arr), np.mean(ppl_hlm_arr), np.std(ppl_hlm_arr),  np.mean(mrr_hlm_arr), np.std(mrr_hlm_arr), len(ppl_llm_arr), len(ppl_llm_arr) / dataset.pos_left_text_tensor["input_ids"].size(0)]
global_stats_ext = [0] * len(inv_temp_cd_arr)
for i in range(len(global_stats)):
    global_stats_ext[i] = global_stats[i]

output_dict = {'ppl_llm_mean,ppl_llm_std,mrr_llm_mean,mrr_llm_std, ppl_hlm_mean,ppl_hlm_std,mrr_hlm_mean,mrr_hlm_std,num,ratio ': global_stats_ext, 
               'ap_ppl_temp': ap_ppl_temp_arr, 'ap_ppl_mean': ap_ppl_mean_arr, 'ap_ppl_std': ap_ppl_std_arr, 'ap_mrr_temp': ap_mrr_temp_arr, 'ap_mrr_mean': ap_mrr_mean_arr, 'ap_mrr_std': ap_mrr_std_arr,
               'cd_ppl_temp': cd_ppl_temp_arr, 'cd_ppl_mean': cd_ppl_mean_arr, 'cd_ppl_std': cd_ppl_std_arr, 'cd_mrr_temp': cd_mrr_temp_arr, 'cd_mrr_mean': cd_mrr_mean_arr, 'cd_mrr_std': cd_mrr_std_arr,
               }

df = pd.DataFrame.from_dict(output_dict, orient='index')
df.to_csv(output_file)

#print( np.mean(ppl_llm_arr) )
#for inv_temp in ppl_ap_dict:
#    print('ap', inv_temp, np.mean(ppl_ap_dict[inv_temp]) )

#dataset.save(output_file)
#with open(output_path, 'w') as f_out:
#    json.dump([all_pos_features_list, all_neg_1_features_list, all_neg_2_features_list, all_neg_3_features_list], f_out,indent=4)
