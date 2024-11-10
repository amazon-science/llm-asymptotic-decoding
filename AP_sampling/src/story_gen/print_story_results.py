from os import listdir
from os.path import isfile, join, isdir
import json
import numpy as np
import pandas as pd

dataset_name = 'story'

repeat_num = 8

#dataset_prefix = "factual_1000_"
if dataset_name == 'story':
    dataset_prefix = "story_start2_1000_"
    eval_name = 'scores_s3'
    #eval_name = 'scores'

record_dir = '/mnt/efs/Haw-Shiuan/AP_sampling/outputs/' + dataset_name
#output_file = 'outputs/factual_gen/all_{}.csv'.format(eval_name)
output_file = 'outputs/{}/{}_1000_{}.csv'.format(dataset_name, dataset_name, eval_name)

def extract_div(file_name):
    with open(file_name) as f_in:
        for line in f_in:
            input_dict = json.loads(line)
            #print(input_dict)
            prefix = 'factual'
            div_4 = input_dict[prefix+'-distinct-4']
            div_3 = input_dict[prefix+'-distinct-3']
            div_2 = input_dict[prefix+'-distinct-2']

    return div_4, div_3, div_2

def extract_scores(file_name, rouge1_arr, rouge2_arr, sim_arr, ppl_arr, mauve_arr, prefix):
    rouge1 = np.nan 
    rouge2 = np.nan 
    sim = np.nan 
    ppl = np.nan 
    mauve = np.nan 
    with open(file_name) as f_in:
        for line in f_in:
            input_dict = json.loads(line)
            if prefix+'rouge1' in input_dict:
                rouge1 = input_dict[prefix+'rouge1']
            if prefix+'rouge2' in input_dict:
                rouge2 = input_dict[prefix+'rouge2']
            if prefix+'text_sim' in input_dict:
                sim = input_dict[prefix+'text_sim']
            if 'ppl' in input_dict:
                ppl = input_dict['ppl']
            if prefix+'mauve' in input_dict:
                mauve = input_dict[prefix+'mauve']
    rouge1_arr.append(rouge1)
    rouge2_arr.append(rouge2)
    sim_arr.append(sim)
    if ppl_arr is not None:
        ppl_arr.append(ppl)
    mauve_arr.append(mauve)

def extract_repeat(file_name, Repetition_arr):
    Repetition_arr_new = [] 
    with open(file_name) as f_in:
        for line in f_in:
            input_dict = json.loads(line)
            if 'repetition_ratio' in input_dict:
                Repetition_arr_new.append(input_dict['repetition_ratio'])
    #print(file_name)
    #if len(NE_ER_arr_new) == len(Entail_arr_new) and len(Repetition_arr_new) == len(Entail_arr_new):
    if len(Repetition_arr_new) == 0:
        Repetition_arr_new = [np.nan]
    if len(Repetition_arr_new) > 0:
        Repetition_arr.append(Repetition_arr_new[-1])

method_d2_scores = {}
output_col = [
    'div_factual_2', 'div_factual_3', 'div_factual_4', 'rouge1_ref', 'rouge2_ref', 'sim_ref', 'mauve_ref', 'ppl', 'rouge1_prompt', 'rouge2_prompt', 'sim_prompt', 'mauve_prompt', 'Repetition_factual', 
    ]

print(dataset_prefix+'6.9b_AP')

for result_dir in listdir(record_dir):
    #print(result_dir)
    if not result_dir.startswith(dataset_prefix+'6.9b_AP') and not result_dir.startswith(dataset_prefix+'6.9b_CS') and not result_dir.startswith(dataset_prefix+'6.9b_CD') and not result_dir.startswith(dataset_prefix+'6.9b_fe_AP_topp') and not result_dir.startswith(dataset_prefix+'6.9b_fe_CD_topp') and not result_dir.startswith(dataset_prefix+'6.9b_fe_top') and not result_dir.startswith(dataset_prefix+'6.9b_fecut_topp') and not result_dir.startswith(dataset_prefix+'6.9b_topp') and not result_dir.startswith(dataset_prefix+'6.9b_topk') and not result_dir.startswith(dataset_prefix+'6.9b_typical') and not result_dir.startswith(dataset_prefix+'6.9b_eta') and not result_dir.startswith(dataset_prefix+'6.9b_decay') and not result_dir.startswith(dataset_prefix+'6.9b_p') and not result_dir.startswith(dataset_prefix+'OpenLLaMA2-7b') and not result_dir.startswith(dataset_prefix+'OPT-6.7b') and not result_dir.startswith(dataset_prefix+'GPT2'):
        continue
    score_dir_path = join(record_dir, result_dir, eval_name)
    if not isdir(score_dir_path):
        continue
    print(score_dir_path)
    Repetition_factual_arr = []
    rouge1_ref_arr = []
    rouge2_ref_arr = []
    sim_ref_arr = []
    ppl_arr = []
    mauve_ref_arr = []
    rouge1_prompt_arr = []
    rouge2_prompt_arr = []
    sim_prompt_arr = []
    mauve_prompt_arr = []
    for score_file in listdir(score_dir_path):
        score_file_path = join(score_dir_path, score_file)
        if not isfile(score_file_path):
            continue
        if score_file.startswith('repetition_'):
            continue
        if score_file.startswith('factual_'):
            div_4, div_3, div_2 = extract_div(score_file_path)
        elif score_file.endswith('_results.jsonl') :
            extract_repeat(score_file_path, Repetition_factual_arr)
        elif score_file.endswith('_ref.jsonl') :
            extract_scores(score_file_path, rouge1_ref_arr, rouge2_ref_arr, sim_ref_arr, ppl_arr, mauve_ref_arr, 'ref_')
        elif score_file.endswith('_prompt.jsonl') :
            extract_scores(score_file_path, rouge1_prompt_arr, rouge2_prompt_arr, sim_prompt_arr, None, mauve_prompt_arr, 'prompt_')
    if len(rouge1_ref_arr) < repeat_num:
        continue
    output_arr = [div_2, div_3, div_4, np.mean(rouge1_ref_arr), np.mean(rouge2_ref_arr), np.mean(sim_ref_arr), np.mean(mauve_ref_arr), np.mean(ppl_arr), np.mean(rouge1_prompt_arr), np.mean(rouge2_prompt_arr), np.mean(sim_prompt_arr), np.mean(mauve_prompt_arr), np.mean(Repetition_factual_arr) ]

    method_d2_scores[result_dir] = output_arr
    if 'topp_p1.0_temp_1.0' in result_dir:
        method_d2_scores[result_dir.replace('topp_p1.0_temp_1.0','topk_kinf')] = output_arr
    if 'topk_k1.0' in result_dir:
        method_d2_scores[result_dir.replace('topk_k1.0','topp_p0_temp_1.0')] = output_arr

df = pd.DataFrame.from_dict(method_d2_scores, orient='index', columns=output_col)
df.to_csv(output_file)
