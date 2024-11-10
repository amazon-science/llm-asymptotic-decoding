import pandas as pd
import numpy as np
import os
import json


input_folder = 'outputs/GPT_exp/comp_GPT3.5_responses_500/'
#input_folder = 'outputs/wp/GPT_exp/test_GPT3.5_responses_500/'


result_dict = {'file_name': [], 'avg_win_rate_F': [], 'avg_win_rate_C': [], 'avg_win_rate_L': [], 'avg_win_rate_O': [] }

all_bad_idx = []

for result_file in os.listdir(input_folder):
    file_path = input_folder+result_file
    if not os.path.isfile(file_path):
        continue
    with open(file_path) as f_in:
        all_inputs = json.load(f_in)
    bad_idx_list = []
    if len(all_inputs) == 5:
        pred_method_name, base_method_name, system_prompt1, bad_idx_list, all_list = all_inputs   
    all_bad_idx = all_bad_idx + bad_idx_list

all_bad_idx_set = set(all_bad_idx)

print(all_bad_idx_set)

#for result_file in input_file_list:
for result_file in os.listdir(input_folder):
    file_path = input_folder+result_file
    if not os.path.isfile(file_path):
        continue
    with open(file_path) as f_in:
        all_inputs = json.load(f_in)
    if len(all_inputs) == 4:
        pred_method_name, base_method_name, system_prompt1, all_list = all_inputs
    elif len(all_inputs) == 5:
        pred_method_name, base_method_name, system_prompt1, bad_idx_list, all_list = all_inputs   
    id_list, context_list_pred, gen_list_pred, gen_list_base, ref_list, prompt_list, first_res_list, response_list, parse_win_list = zip(*all_list)
    avg_win_rate_F = []
    avg_win_rate_C = []
    avg_win_rate_L = []
    avg_win_rate_O = []
    for i in range(len(id_list)):
        if i in all_bad_idx_set:
            continue
        avg_win_rate_F.append(int(parse_win_list[i]['F'] == 'pred'))
        avg_win_rate_C.append(int(parse_win_list[i]['C'] == 'pred'))
        avg_win_rate_L.append(int(parse_win_list[i]['L'] == 'pred'))
        avg_win_rate_O.append(int(parse_win_list[i]['O'] == 'pred'))

    result_dict['file_name'].append(result_file)
    result_dict['avg_win_rate_F'].append(np.mean(avg_win_rate_F))
    result_dict['avg_win_rate_C'].append(np.mean(avg_win_rate_C))
    result_dict['avg_win_rate_L'].append(np.mean(avg_win_rate_L))
    result_dict['avg_win_rate_O'].append(np.mean(avg_win_rate_O))

df = pd.DataFrame.from_dict(result_dict)

#pd.set_option('display.max_columns', None)
pd.options.display.max_colwidth = 150

df_sort = df.set_index('file_name').sort_values(by=['file_name'])

#print(df_sort[ ['avg_win_rate_O', 'avg_diff_score_O']])
print(df_sort)
#print(df['file_name'])
