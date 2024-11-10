from os import listdir
from os.path import isfile, join, isdir
import json
import numpy as np
import pandas as pd

eval_name = 'scores_s3'
#eval_name = 'scores'

repeat_num = 4

#dataset_prefix = "factual_1000_"
dataset_prefix = "factual_test7k_"

record_dir = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/factual_gen'
#output_file = 'outputs/factual_gen/all_{}.csv'.format(eval_name)
output_file = 'outputs/factual_gen/all_test_{}.csv'.format(eval_name)

def extract_div(file_name):
    with open(file_name) as f_in:
        for line in f_in:
            input_dict = json.loads(line)
            #print(input_dict)
            if 'factual-distinct-4' in input_dict:
                prefix = 'factual'
            else:
                prefix = 'nonfactual'
            div_4 = input_dict[prefix+'-distinct-4']
            div_3 = input_dict[prefix+'-distinct-3']
            div_2 = input_dict[prefix+'-distinct-2']

    return div_4, div_3, div_2

def extract_entailment_repeat(file_name, NE_ER_arr, Entail_arr, Repetition_arr):
    NE_ER_arr_new = [] 
    Entail_arr_new = [] 
    Repetition_arr_new = [] 
    with open(file_name) as f_in:
        for line in f_in:
            input_dict = json.loads(line)
            if 'avg_hallu_ner_ratio' in input_dict:
                NE_ER_arr_new.append(input_dict['avg_hallu_ner_ratio'])
                Entail_arr_new.append(input_dict['nli_entail_class_ratio'])
            elif 'repetition_ratio' in input_dict:
                Repetition_arr_new.append(input_dict['repetition_ratio'])
    #print(file_name)
    #if len(NE_ER_arr_new) == len(Entail_arr_new) and len(Repetition_arr_new) == len(Entail_arr_new):
    if len(Repetition_arr_new) == 0:
        Repetition_arr_new = [np.nan]
    if len(NE_ER_arr_new) >0 and len(Repetition_arr_new) > 0:
        NE_ER_arr.append(NE_ER_arr_new[-1])
        Entail_arr.append(Entail_arr_new[-1])
        Repetition_arr.append(Repetition_arr_new[-1])
    assert len(NE_ER_arr) == len(Entail_arr)
    assert len(Repetition_arr) == len(Entail_arr)

method_d2_scores = {}
output_col = [
    'div_factual_2', 'div_factual_3', 'div_factual_4', 
    'NE_ER_factual', 'Entail_factual', 'Repetition_factual', 
    'div_nonfactual_2', 'div_nonfactual_3', 'div_nonfactual_4', 
    'NE_ER_nonfactual', 'Entail_nonfactual', 'Repetition_nonfactual', 
    ]

for result_dir in listdir(record_dir):
    if not result_dir.startswith(dataset_prefix+'6.9b_CS') and not result_dir.startswith(dataset_prefix+'6.9b_CD') and not result_dir.startswith(dataset_prefix+'6.9b_fe_CD_topp') and not result_dir.startswith(dataset_prefix+'6.9b_fe_top') and not result_dir.startswith(dataset_prefix+'6.9b_fecut_topp') and not result_dir.startswith(dataset_prefix+'6.9b_topp') and not result_dir.startswith(dataset_prefix+'6.9b_topk') and not result_dir.startswith(dataset_prefix+'6.9b_typical') and not result_dir.startswith(dataset_prefix+'6.9b_eta') and not result_dir.startswith(dataset_prefix+'6.9b_decay') and not result_dir.startswith(dataset_prefix+'6.9b_p') and not result_dir.startswith(dataset_prefix+'OpenLLaMA2-7b') and not result_dir.startswith(dataset_prefix+'OPT-6.7b'):
    #if not result_dir.startswith(dataset_prefix+'6.9b_CD') and not result_dir.startswith(dataset_prefix+'6.9b_fe_CD_topp') and not result_dir.startswith(dataset_prefix+'6.9b_fe_top') and not result_dir.startswith(dataset_prefix+'6.9b_fecut_topp') and not result_dir.startswith(dataset_prefix+'6.9b_topp') and not result_dir.startswith(dataset_prefix+'6.9b_topk') and not result_dir.startswith(dataset_prefix+'6.9b_typical') and not result_dir.startswith(dataset_prefix+'6.9b_eta') and not result_dir.startswith(dataset_prefix+'6.9b_decay') and not result_dir.startswith(dataset_prefix+'6.9b_p'):
        continue
    score_dir_path = join(record_dir, result_dir, eval_name)
    if not isdir(score_dir_path):
        continue
    print(score_dir_path)
    NE_ER_factual_arr = []
    Entail_factual_arr = []
    Repetition_factual_arr = []
    NE_ER_nonfactual_arr = []
    Entail_nonfactual_arr = []
    Repetition_nonfactual_arr = []
    for score_file in listdir(score_dir_path):
        score_file_path = join(score_dir_path, score_file)
        if not isfile(score_file_path):
            continue
        if score_file.startswith('repetition_'):
            continue
        if score_file.startswith('factual_factual_'):
            div_4, div_3, div_2 = extract_div(score_file_path)
        elif score_file.startswith('nonfactual_factual_'):
            div_non_4, div_non_3, div_non_2 = extract_div(score_file_path)
        elif score_file.startswith('factual_'):
            extract_entailment_repeat(score_file_path, NE_ER_factual_arr, Entail_factual_arr, Repetition_factual_arr)
        elif score_file.startswith('nonfactual_'):
            extract_entailment_repeat(score_file_path, NE_ER_nonfactual_arr, Entail_nonfactual_arr, Repetition_nonfactual_arr)
    #print(NE_ER_factual_arr)
    if len(NE_ER_factual_arr) < repeat_num:
        continue
    output_arr = [div_2, div_3, div_4, 
            np.mean(NE_ER_factual_arr), np.mean(Entail_factual_arr), np.mean(Repetition_factual_arr), 
            div_non_2, div_non_3, div_non_4, 
            np.mean(NE_ER_nonfactual_arr), np.mean(Entail_nonfactual_arr), np.mean(Repetition_nonfactual_arr) ]

    method_d2_scores[result_dir] = output_arr
    if 'topp_p1.0_temp_1.0' in result_dir:
        method_d2_scores[result_dir.replace('topp_p1.0_temp_1.0','topk_kinf')] = output_arr
    if 'topk_k1.0' in result_dir:
        method_d2_scores[result_dir.replace('topk_k1.0','topp_p0_temp_1.0')] = output_arr

df = pd.DataFrame.from_dict(method_d2_scores, orient='index', columns=output_col)
df.to_csv(output_file)
