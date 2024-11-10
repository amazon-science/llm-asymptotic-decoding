import sys
sys.path.append('/mnt/efs/Haw-Shiuan/AP_sampling/src')
import numpy as np

from model_mlp_logit import GPTNeoXForLogitCorrection, OPTForLogitCorrection, Qwen2ForLogitCorrection
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import torch
import json
from torch import nn

import pandas as pd

cuda_idx = 0
cuda_idx = 2
cuda_idx = 4
cuda_idx = 6

device_llm = 'cuda:'+str(cuda_idx)
device_tlm = 'cuda:'+str(cuda_idx)
device_ap = 'cuda:'+str(cuda_idx)
device_hlm = 'cuda:'+str(cuda_idx+1)

small_prob_for_not_inf = 1e-2
#small_prob_for_not_inf = 1e-5
#small_prob_for_not_inf = 1e-10

#large_model_name = 'EleutherAI/pythia-70m-deduped'
#large_model_name = 'EleutherAI/pythia-6.9b-deduped'
#large_model_name = 'facebook/opt-6.7b'
#large_model_name = 'Qwen/Qwen1.5-4b-Chat'
large_model_name = 'Qwen/Qwen1.5-4b'

if 'pythia' in large_model_name:
    #huge_model_name = 'EleutherAI/pythia-70m-deduped'
    huge_model_name = 'EleutherAI/pythia-12b-deduped'
    small_model_name = 'EleutherAI/pythia-70m-deduped'
    ap_model_name = "models/prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4"
elif 'opt' in large_model_name:
    huge_model_name = 'facebook/opt-13b'
    small_model_name = 'facebook/opt-125m'
    ap_model_name = "models/prob_opt_wiki_ext_new_1e6_70M_bsz_32_e5_only_top_last_w_10_l1_reg_w_04_logit_exp_decay_lr-4"
elif 'Qwen1.5-4b-Chat' in large_model_name:
    huge_model_name = 'Qwen/Qwen1.5-7b-Chat'
    small_model_name = 'Qwen/Qwen1.5-0.5b-Chat'
    #ap_model_name = "models/prob_Qwen_4b_Chat_wiki_ext_new_1e6_500M_bsz_32_e5_only_top_last_w_10_l1_reg_w_12_logit_exp_decay_lr-4"
    ap_model_name = "models/prob_Qwen_4b_Chat_wiki_ext_new_1e6_500M_bsz_32_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4"
elif 'Qwen1.5-4b' in large_model_name:
    huge_model_name = 'Qwen/Qwen1.5-7b'
    small_model_name = 'Qwen/Qwen1.5-0.5b'
    #ap_model_name = "models/prob_Qwen_4b_wiki_ext_new_1e6_500M_bsz_32_e5_only_top_last_w_10_l1_reg_w_12_logit_exp_decay_lr-4"
    ap_model_name = "models/prob_Qwen_4b_wiki_ext_new_1e6_500M_bsz_32_e5_only_top_last_w_10_l1_reg_w_1_logit_exp_decay_lr-4"
    #ap_model_name = "models/prob_Qwen_4b_wiki_ext_new_1e6_500M_bsz_32_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4"


#max_len = 512
max_len = 1024



#input_file = "outputs/qasc/qasc_neg_fact_train.json"
#output_file = "outputs/qasc/final_Pythia_qasc_neg_fact_train_topk20.csv"
#output_file = "outputs/qasc/final_OPT_qasc_neg_fact_train_topk20.csv"
#output_file = "outputs/qasc/final_Qwen_w1_qasc_neg_fact_train_topk20.csv"

input_file = "outputs/qasc/qasc_neg_train.json"
#output_file = "outputs/qasc/final_Pythia_qasc_neg_train_topk20.csv"
#output_file = "outputs/qasc/final_OPT_qasc_neg_train_topk20.csv"
output_file = "outputs/qasc/final_Qwen_w1_qasc_neg_train_topk20.csv"

#input_file = "outputs/arc/arc_neg_all_train.json"
#output_file = "outputs/arc/final_Pythia_arc_neg_all_train_topk20.csv"
#output_file = "outputs/arc/final_OPT_arc_neg_all_train_topk20.csv"
#output_file = "outputs/arc/final_Qwen_w1_arc_neg_all_train_topk20.csv"

#input_file = "outputs/socialiqa/socialiqa_neg_train.json"
#output_file = "outputs/socialiqa/final_Pythia_socialiqa_neg_train_topk20.csv"
#output_file = "outputs/socialiqa/final_OPT_socialiqa_neg_train_topk20.csv"
#output_file = "outputs/socialiqa/final_Qwen_w1_socialiqa_neg_train_topk20.csv"

#input_file = "outputs/commonqa/commonqa_neg_train.json"
#output_file = "outputs/commonqa/final_Pythia_commonqa_neg_train_topk20.csv"
#output_file = "outputs/commonqa/final_OPT_commonqa_neg_train_topk20.csv"
#output_file = "outputs/commonqa/final_Qwen_w1_commonqa_neg_train_topk20.csv"

#if 'multirc' in input_file:
#    max_len = 768


print(input_file)

inv_temp_cd_arr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#inv_temp_cd_arr = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#inv_temp_cd_arr = [0.05, 0.1, 0.125, 0.15, 0.25, 0.5]
#inv_temp_cd_arr = [0.0125,0.025, 0.05, 0.1, 0.125, 0.15]
#inv_temp_cd_arr = [0.15, 0.2, 0.25, 0.3, 0.35, 0.5]

inv_temp_ap_arr = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
#inv_temp_ap_arr = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
#inv_temp_ap_arr = [0.1, 0.2, 0.25, 0.3, 0.5, 1]
#inv_temp_ap_arr = [0.025, 0.05, 0.1, 0.2, 0.25, 0.3]
#inv_temp_ap_arr = [0.25, 0.3, 0.35, 0.4, 0.5, 1]

#vis_output = False
#vis_output = True
#if vis_output:
#    assert batch_size == 1
#    vis_cd_temp = 0.4
#    vis_ap_temp = 0.6

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
    context_list = []
    option_list = []
    for line in f_in:
        input_dict = json.loads(line)
        context_list.append(input_dict['prompt'])
        option_list.append(input_dict['all_ans'])

    return context_list, option_list

with open(input_file) as f_in:
    context_list, option_list = load_squad_file(f_in)

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
mrr_llm_arr = []
acc_llm_arr = []
ppl_diff_llm_arr = []

ppl_hlm_arr = []
mrr_hlm_arr = []
acc_hlm_arr = []
ppl_diff_hlm_arr = []

ppl_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
ppl_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}
mrr_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
mrr_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}
acc_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
acc_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}
ppl_diff_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
ppl_diff_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}

def vis_top_w_pred(prob_topk, text_pos, prefix):
    top_k_out = torch.topk(prob_topk, topk_th)
    top_k_value = top_k_out[0][text_pos, :]
    top_k_w_idx = top_k_out[1][text_pos, :]
    top_token = tokenizer.convert_ids_to_tokens(top_k_w_idx)
    print( prefix, list(zip( top_token, top_k_value.tolist() ) ))


def contrastive_prob(scores, scores_org, scores_cd_small, inv_temp, label):
    scores_cd = scores.clone()
    scores_cd[:,:scores_cd_small.size(-1)] = scores_cd[:,:scores_cd_small.size(-1)] - inv_temp * scores_cd_small
    scores_cd_all = scores_org.clone()
    scores_cd_all[:,:scores_cd_small.size(-1)] = scores_cd_all[:,:scores_cd_small.size(-1)] - inv_temp * scores_cd_small
    gt_logits_cd = torch.gather(scores_cd, dim=-1, index=label)
    gt_rank_cd = (gt_logits_cd <= scores_cd).to(torch.int32).sum(dim=-1)
    return scores_cd.softmax(dim=-1), scores_cd_all.softmax(dim=-1), 1 / gt_rank_cd



def get_score_options( model_llm, context_tensor, label_arr, ap_device ):
    outputs_llm = model_llm( context_tensor.to(model_llm.device), return_dict = True)
    scores_org_last = nn.functional.log_softmax(outputs_llm.logits[0,-1,:].unsqueeze(dim=0), dim=-1)

    scores_org_arr = []
    for j in range(len(label_arr)):
        option_tensor = label_arr[j].squeeze(dim=-1).unsqueeze(dim=0)
        outputs_llm_options = model_llm( option_tensor.to(model_llm.device), past_key_values=outputs_llm.past_key_values, return_dict = True)
        
        scores_org_option = nn.functional.log_softmax(outputs_llm_options.logits[0,:-1,:], dim=-1)
        #print(scores_org_last.size(), scores_org_option.size())
        scores_org = torch.concat( (scores_org_last, scores_org_option), dim=0 )
        
        scores_org_arr.append(scores_org.to(ap_device))

    return scores_org_arr

def compute_gt_rank(scores_org, label):
    #print(scores_org.size(), label.size())
    gt_logits = torch.gather(scores_org, dim=-1, index=label)
    gt_rank = (gt_logits <= scores_org).to(torch.int32).sum(dim=-1)
    return gt_rank

#def compute_gt_rank(scores_org_arr, label_arr):
#    gt_rank_arr = []
#    for j in range(len(scores_org_arr)):
#        scores_org = scores_org_arr[j]
#        label = label_arr[j]
#        gt_logits = torch.gather(scores_org, dim=-1, index=label)
#        gt_rank = (gt_logits <= scores_org).to(torch.int32).sum(dim=-1)
#        gt_rank_arr.append(gt_rank)
#    return gt_rank_arr

def compute_metrics(prob_topk, inv_gt_rank, target_label, ppl_i_arr, mrr_i_arr):
    ppl_org = -torch.log( torch.gather( prob_topk, dim=-1, index = target_label ) + small_prob_for_not_inf ).mean().item()
    mrr_all = inv_gt_rank.mean().item()
    ppl_i_arr.append(ppl_org)
    mrr_i_arr.append(mrr_all)


def diff_metrics(ppl_i_arr, acc_llm_arr, ppl_diff_llm_arr):
    if min(ppl_i_arr) == ppl_i_arr[0]:
        acc_llm_arr.append(1)
    else:
        acc_llm_arr.append(0)
    best_neg_ppl = min(ppl_i_arr[1:])
    ppl_diff_llm_arr.append(ppl_i_arr[0] - best_neg_ppl)

def print_all_results(ppl_llm_arr, mrr_llm_arr, ppl_hlm_arr, mrr_hlm_arr, ppl_cd_dict, mrr_cd_dict, ppl_ap_dict, mrr_ap_dict, m1, m2):
    print('topk20 org'+m1, np.mean(ppl_llm_arr), m2, np.mean(mrr_llm_arr) )
    print('topk20 hlm'+m1, np.mean(ppl_hlm_arr), m2, np.mean(mrr_hlm_arr) )
    ppl_cd_results = [ (inv_temp, np.mean(ppl_cd_dict[inv_temp])) for inv_temp in ppl_cd_dict ]
    mrr_cd_results = [ (inv_temp, np.mean(mrr_cd_dict[inv_temp])) for inv_temp in mrr_cd_dict ]
    best_ppl_cd = min(ppl_cd_results,key=lambda x:x[1])
    best_mrr_cd = max(mrr_cd_results,key=lambda x:x[1])
    ppl_ap_results = [ (inv_temp, np.mean(ppl_ap_dict[inv_temp])) for inv_temp in ppl_ap_dict ]
    mrr_ap_results = [ (inv_temp, np.mean(mrr_ap_dict[inv_temp])) for inv_temp in mrr_ap_dict ]
    best_ppl_ap = min(ppl_ap_results,key=lambda x:x[1])
    best_mrr_ap = max(mrr_ap_results,key=lambda x:x[1])
    print('topk20 cd best'+m1, '{} {}'.format(best_ppl_cd[1], best_ppl_cd[0]), 'best'+m2, '{} {}'.format(best_mrr_cd[1], best_mrr_cd[0]), m1, ppl_cd_results, m2, mrr_cd_results)
    print('topk20 ap best'+m1, '{} {}'.format(best_ppl_ap[1], best_ppl_ap[0]), 'best'+m2, '{} {}'.format(best_mrr_ap[1], best_mrr_ap[0]), m1, ppl_ap_results, m2, mrr_ap_results)

for i in range(len(context_list)):
    print(i / len(context_list))
    with torch.no_grad():
        context_tensor = tokenizer(context_list[i], truncation=True, return_tensors='pt', max_length = max_len)['input_ids']
        context_len = context_tensor.size(-1)
        
        label_arr = []
        for j in range(len(option_list[i])):
            option_tensor = tokenizer(option_list[i][j], add_special_tokens=False, padding=True, truncation=True, return_tensors='pt', max_length = max_len - context_len)['input_ids']
            label = option_tensor[0,:].unsqueeze(dim=-1)
            label_arr.append(label.to(model_ap.device))
        
        scores_org_arr = get_score_options( model_llm, context_tensor, label_arr, model_ap.device )
        scores_org_hlm_arr = get_score_options( model_hlm, context_tensor, label_arr, model_ap.device )
        scores_org_ap_arr = get_score_options( model_ap, context_tensor, label_arr, model_ap.device )
        scores_org_cd_arr = get_score_options( model_st, context_tensor, label_arr, model_ap.device )
        
        #gt_rank_arr = compute_gt_rank(scores_org_arr, label_arr)
        #gt_rank_hlm_arr = compute_gt_rank(scores_org_hlm_arr, label_arr)
        
        ppl_i_arr = []
        mrr_i_arr = []
        ppl_i_hlm_arr = []
        mrr_i_hlm_arr = []
        ppl_i_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
        ppl_i_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}
        mrr_i_cd_dict = {inv_temp: [] for inv_temp in inv_temp_cd_arr}
        mrr_i_ap_dict = {inv_temp: [] for inv_temp in inv_temp_ap_arr}
        for j in range(len(option_list[i])):
            scores_org = scores_org_arr[j]
            scores_hlm = scores_org_hlm_arr[j]
            scores_ap_small = scores_org_ap_arr[j]
            scores_cd_small = scores_org_cd_arr[j]
            label = label_arr[j]

            indices_to_remove_topk = scores_org < torch.topk(scores_org, topk_th)[0][..., -1, None]
            scores = scores_org.clone().masked_fill(indices_to_remove_topk, -float("Inf"))
            gt_rank = compute_gt_rank(scores, label)
            prob_topk = scores.softmax(dim=-1)
            
            indices_to_remove_topk_hlm = scores_hlm < torch.topk(scores_hlm, topk_th)[0][..., -1, None]
            scores_hlm_masked = scores_hlm.clone().masked_fill(indices_to_remove_topk_hlm, -float("Inf"))
            gt_rank_hlm = compute_gt_rank(scores_hlm, label)
            prob_hlm_topk = scores_hlm_masked.softmax(dim=-1)
        
            prob_cd_dict = {}
            prob_ap_dict = {}
            prob_cd_all_dict = {}
            prob_ap_all_dict = {}
            mrr_cd_raw_dict = {}
            mrr_ap_raw_dict = {}
            for inv_temp in inv_temp_cd_arr:
                prob_cd_dict[inv_temp], prob_cd_all_dict[inv_temp], mrr_cd_raw_dict[inv_temp] = contrastive_prob(scores, scores_org, scores_cd_small, inv_temp, label)
            for inv_temp in inv_temp_ap_arr:
                prob_ap_dict[inv_temp], prob_ap_all_dict[inv_temp], mrr_ap_raw_dict[inv_temp] = contrastive_prob(scores, scores_org, scores_ap_small, inv_temp, label)

            all_in_topk = torch.all( gt_rank <= topk_th ) and torch.all( gt_rank_hlm <= topk_th )
            if not all_in_topk:
                if j == 0:
                    break
                else:
                    continue

            compute_metrics(prob_topk, 1 / gt_rank, label, ppl_i_arr, mrr_i_arr)
            compute_metrics(prob_hlm_topk, 1 / gt_rank_hlm, label, ppl_i_hlm_arr, mrr_i_hlm_arr)
            for inv_temp in prob_cd_dict:
                compute_metrics(prob_cd_dict[inv_temp], mrr_cd_raw_dict[inv_temp], label, ppl_i_cd_dict[inv_temp], mrr_i_cd_dict[inv_temp])
            for inv_temp in prob_ap_dict:
                compute_metrics(prob_ap_dict[inv_temp], mrr_ap_raw_dict[inv_temp], label, ppl_i_ap_dict[inv_temp], mrr_i_ap_dict[inv_temp])
            
            #if vis_output and all_in_topk and  ppl_llm_arr[-1] > ppl_cd_dict[vis_cd_temp][-1] and ppl_cd_dict[vis_cd_temp][-1] > ppl_ap_dict[vis_ap_temp][-1] and mrr_cd_dict[vis_cd_temp][-1] < mrr_ap_dict[vis_ap_temp][-1]:
            #if vis_output and all_in_topk: 
            #    print(output_list[idx][1])
            #    print(ppl_llm_arr[-1], ppl_cd_dict[vis_cd_temp][-1], ppl_ap_dict[vis_ap_temp][-1])
            #    vis_top_w_pred(prob_topk[i,:,:], org_left_len[i], 'llm')
            #    vis_top_w_pred(prob_cd_dict[vis_cd_temp][i,:,:], org_left_len[i], 'cd')
            #    vis_top_w_pred(prob_ap_dict[vis_ap_temp][i,:,:], org_left_len[i], 'ap')
        if len(ppl_i_arr) > 0:
            ppl_llm_arr.append(ppl_i_arr[0])
            mrr_llm_arr.append(mrr_i_arr[0])

            ppl_hlm_arr.append(ppl_i_hlm_arr[0])
            mrr_hlm_arr.append(mrr_i_hlm_arr[0])

            for inv_temp in prob_cd_dict:
                ppl_cd_dict[inv_temp].append(ppl_i_cd_dict[inv_temp][0])
                mrr_cd_dict[inv_temp].append(mrr_i_cd_dict[inv_temp][0])

            for inv_temp in prob_ap_dict:
                ppl_ap_dict[inv_temp].append(ppl_i_ap_dict[inv_temp][0])
                mrr_ap_dict[inv_temp].append(mrr_i_ap_dict[inv_temp][0])

        if len(ppl_i_arr) > 1:
            diff_metrics(ppl_i_arr, acc_llm_arr, ppl_diff_llm_arr)
            diff_metrics(ppl_i_hlm_arr, acc_hlm_arr, ppl_diff_hlm_arr)
            for inv_temp in prob_cd_dict:
                diff_metrics(ppl_i_cd_dict[inv_temp], acc_cd_dict[inv_temp], ppl_diff_cd_dict[inv_temp])
            for inv_temp in prob_ap_dict:
                diff_metrics(ppl_i_ap_dict[inv_temp], acc_ap_dict[inv_temp], ppl_diff_ap_dict[inv_temp])


        if i % 10 == 0 or i == len(context_list)-1:
            print(len(ppl_diff_llm_arr))
            print_all_results(ppl_diff_llm_arr, acc_llm_arr, ppl_diff_hlm_arr, acc_hlm_arr, ppl_diff_cd_dict, acc_cd_dict, ppl_diff_ap_dict, acc_ap_dict, ' ppl diff', ' acc')
            print(len(ppl_llm_arr))
            print_all_results(ppl_llm_arr, mrr_llm_arr, ppl_hlm_arr, mrr_hlm_arr, ppl_cd_dict, mrr_cd_dict, ppl_ap_dict, mrr_ap_dict, ' ppl', ' mrr')

print('topk20 std org ppl', np.std(ppl_llm_arr), ' mrr', np.std(mrr_llm_arr) )
print('topk20 std hlm ppl', np.std(ppl_hlm_arr), ' mrr', np.std(mrr_hlm_arr) )
print('topk20 std cd ppl', [ (inv_temp, np.std(ppl_cd_dict[inv_temp])) for inv_temp in ppl_cd_dict ], ' mrr', [ (inv_temp, np.std(mrr_cd_dict[inv_temp])) for inv_temp in mrr_cd_dict ]  )
print('topk20 std ap ppl', [ (inv_temp, np.std(ppl_ap_dict[inv_temp])) for inv_temp in ppl_ap_dict ], ' mrr', [ (inv_temp, np.std(mrr_ap_dict[inv_temp])) for inv_temp in mrr_ap_dict ] )
print(len(ppl_llm_arr))
print(len(ppl_llm_arr) / len(context_list))

print('topk20 std org ppl diff', np.std(ppl_diff_llm_arr), ' acc', np.std(acc_llm_arr) )
print('topk20 std hlm ppl diff', np.std(ppl_diff_hlm_arr), ' acc', np.std(acc_hlm_arr) )
print('topk20 std cd ppl diff', [ (inv_temp, np.std(ppl_diff_cd_dict[inv_temp])) for inv_temp in ppl_diff_cd_dict ], ' acc', [ (inv_temp, np.std(acc_cd_dict[inv_temp])) for inv_temp in acc_cd_dict ]  )
print('topk20 std ap ppl diff', [ (inv_temp, np.std(ppl_diff_ap_dict[inv_temp])) for inv_temp in ppl_diff_ap_dict ], ' acc', [ (inv_temp, np.std(acc_ap_dict[inv_temp])) for inv_temp in acc_ap_dict ] )
print(len(ppl_diff_llm_arr))
print(len(ppl_diff_llm_arr) / len(context_list))

print(input_file)
print(large_model_name)

cd_ppl_results = [ (inv_temp, np.mean(ppl_cd_dict[inv_temp]), np.std(ppl_cd_dict[inv_temp]), np.mean(ppl_diff_cd_dict[inv_temp]), np.std(ppl_diff_cd_dict[inv_temp])) for inv_temp in ppl_cd_dict ]
cd_mrr_results = [ (inv_temp, np.mean(mrr_cd_dict[inv_temp]), np.std(mrr_cd_dict[inv_temp]), np.mean(acc_cd_dict[inv_temp]), np.std(acc_cd_dict[inv_temp])) for inv_temp in mrr_cd_dict ]
ap_ppl_results = [ (inv_temp, np.mean(ppl_ap_dict[inv_temp]), np.std(ppl_ap_dict[inv_temp]), np.mean(ppl_diff_ap_dict[inv_temp]), np.std(ppl_diff_ap_dict[inv_temp])) for inv_temp in ppl_ap_dict ]
ap_mrr_results = [ (inv_temp, np.mean(mrr_ap_dict[inv_temp]), np.std(mrr_ap_dict[inv_temp]), np.mean(acc_ap_dict[inv_temp]), np.std(acc_ap_dict[inv_temp])) for inv_temp in mrr_ap_dict ]

ap_ppl_temp_arr, ap_ppl_mean_arr, ap_ppl_std_arr, ap_ppl_diff_mean_arr, ap_ppl_diff_std_arr  = zip(* ap_ppl_results )
cd_ppl_temp_arr, cd_ppl_mean_arr, cd_ppl_std_arr, cd_ppl_diff_mean_arr, cd_ppl_diff_std_arr  = zip(* cd_ppl_results )
ap_mrr_temp_arr, ap_mrr_mean_arr, ap_mrr_std_arr, ap_acc_mean_arr, ap_acc_std_arr  = zip(* ap_mrr_results )
cd_mrr_temp_arr, cd_mrr_mean_arr, cd_mrr_std_arr, cd_acc_mean_arr, cd_acc_std_arr  = zip(* cd_mrr_results )

global_stats = [np.mean(ppl_llm_arr), np.std(ppl_llm_arr),  np.mean(mrr_llm_arr), np.std(mrr_llm_arr), np.mean(ppl_hlm_arr), np.std(ppl_hlm_arr),  np.mean(mrr_hlm_arr), np.std(mrr_hlm_arr), len(ppl_llm_arr), len(ppl_llm_arr) / len(context_list) ]
global_stats_2 = [np.mean(ppl_diff_llm_arr), np.std(ppl_diff_llm_arr),  np.mean(acc_llm_arr), np.std(acc_llm_arr), np.mean(ppl_diff_hlm_arr), np.std(ppl_diff_hlm_arr),  np.mean(acc_hlm_arr), np.std(acc_hlm_arr), len(ppl_diff_llm_arr), len(ppl_diff_llm_arr) / len(context_list) ]
global_stats_ext = [0] * len(inv_temp_cd_arr)
global_stats_2_ext = [0] * len(inv_temp_cd_arr)
for i in range(len(global_stats)):
    global_stats_ext[i] = global_stats[i]
    global_stats_2_ext[i] = global_stats_2[i]

output_dict = {'ppl_llm_mean,ppl_llm_std,mrr_llm_mean,mrr_llm_std, ppl_hlm_mean,ppl_hlm_std,mrr_hlm_mean,mrr_hlm_std,num,ratio ': global_stats_ext,
               'ap_ppl_temp': ap_ppl_temp_arr, 'ap_ppl_mean': ap_ppl_mean_arr, 'ap_ppl_std': ap_ppl_std_arr, 'ap_mrr_temp': ap_mrr_temp_arr, 'ap_mrr_mean': ap_mrr_mean_arr, 'ap_mrr_std': ap_mrr_std_arr,
               'cd_ppl_temp': cd_ppl_temp_arr, 'cd_ppl_mean': cd_ppl_mean_arr, 'cd_ppl_std': cd_ppl_std_arr, 'cd_mrr_temp': cd_mrr_temp_arr, 'cd_mrr_mean': cd_mrr_mean_arr, 'cd_mrr_std': cd_mrr_std_arr,
               'ppl_diff_llm_mean,ppl_diff_llm_std,acc_llm_mean,acc_llm_std, ppl_diff_hlm_mean,ppl_diff_hlm_std,acc_hlm_mean,acc_hlm_std,num,ratio ': global_stats_2_ext,
               'ap_ppl_diff_temp': ap_ppl_temp_arr, 'ap_ppl_diff_mean': ap_ppl_diff_mean_arr, 'ap_ppl_diff_std': ap_ppl_diff_std_arr, 'ap_acc_temp': ap_mrr_temp_arr, 'ap_acc_mean': ap_acc_mean_arr, 'ap_acc_std': ap_acc_std_arr,
               'cd_ppl_diff_temp': cd_ppl_temp_arr, 'cd_ppl_diff_mean': cd_ppl_diff_mean_arr, 'cd_ppl_diff_std': cd_ppl_diff_std_arr, 'cd_acc_temp': cd_mrr_temp_arr, 'cd_acc_mean': cd_acc_mean_arr, 'cd_acc_std': cd_acc_std_arr,
               }

df = pd.DataFrame.from_dict(output_dict, orient='index')
df.to_csv(output_file)

#print( np.mean(ppl_llm_arr) )
#for inv_temp in ppl_ap_dict:
#    print('ap', inv_temp, np.mean(ppl_ap_dict[inv_temp]) )

#dataset.save(output_file)
#with open(output_path, 'w') as f_out:
#    json.dump([all_pos_features_list, all_neg_1_features_list, all_neg_2_features_list, all_neg_3_features_list], f_out,indent=4)
