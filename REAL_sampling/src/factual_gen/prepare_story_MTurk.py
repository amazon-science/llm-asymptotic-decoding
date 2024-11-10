import json
import pandas as pd
from nltk.tokenize import sent_tokenize
import random
import numpy as np

sample_numbers = 1000
#sample_numbers = 100

#min_gen_char_len = 10
#max_gen_char_len = 250
#max_gen_char_len = 500
min_gen_char_len = 50
max_gen_char_len = 1000

#3,4,2,1
input_file_dict = {'Ours': "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers),
                   'Top-p': 'outputs/factual_gen/story_start2_{}_6.9b_topp_p0.5_temp_1.0/story_start2_{}_6.9b_topp_p0.5_gen_seed1.jsonl'.format(sample_numbers,sample_numbers),
                   'CD': 'outputs/factual_gen/story_start2_{}_6.9b_CD_dt_1.0_p0.3_pythia-70m-deduped/story_start2_{}_6.9b_CD_p0.3_gen_seed1.jsonl'.format(sample_numbers,sample_numbers),
                   'Ours+CD': 'outputs/factual_gen/story_start2_{}_6.9b_fe_CD_topp_exp_1_win_40_dt_2.0_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_start2_{}_6.9b_fe_CD_topp_p1.0_gen_seed1.jsonl'.format(sample_numbers,sample_numbers)
                   }

#input_file_dict = {'Ours': "outputs/factual_gen/story_{}_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_{}_6.9b_fe_topp_p1.0_gen_seed2.jsonl".format(sample_numbers,sample_numbers),
#                   'Top-p': 'outputs/factual_gen/story_{}_6.9b_topp_p0.5_temp_1.0/story_{}_6.9b_topp_p0.5_gen_seed2.jsonl'.format(sample_numbers,sample_numbers),
#                   'CD': 'outputs/factual_gen/story_{}_6.9b_CD_dt_1.0_p0.3_pythia-70m-deduped/story_{}_6.9b_CD_p0.3_gen_seed2.jsonl'.format(sample_numbers,sample_numbers),
#                   'Ours+CD': 'outputs/factual_gen/story_{}_6.9b_fe_CD_topp_exp_1_win_40_dt_2.0_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_{}_6.9b_fe_CD_topp_p1.0_gen_seed2.jsonl'.format(sample_numbers,sample_numbers)
#                   }

#input_file_dict = {'Ours': "outputs/factual_gen/story_{}_6.9b_fe_topp_exp_1_win_40_dt_2.0_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers),
#                   'Top-p': 'outputs/factual_gen/story_{}_6.9b_topp_p0.6_temp_1.0/story_{}_6.9b_topp_p0.6_gen_seed1.jsonl'.format(sample_numbers,sample_numbers),
#                   'CD': 'outputs/factual_gen/story_{}_6.9b_CD_dt_1.0_p0.3_pythia-70m-deduped/story_{}_6.9b_CD_p0.3_gen_seed1.jsonl'.format(sample_numbers,sample_numbers),
#                   'Ours+CD': 'outputs/factual_gen/story_{}_6.9b_fe_CD_topp_exp_1_win_40_dt_1.5_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_{}_6.9b_fe_CD_topp_p1.0_gen_seed1.jsonl'.format(sample_numbers,sample_numbers)
#                   }

output_csv = 'outputs/MTurk/story/gen_start2_max_250_{}.csv'.format(sample_numbers)

method_list = list(input_file_dict.keys())

def load_gen(input_file):
    id_list = []
    context_list = []
    gen_list = []
    with open(input_file) as f_in:
        for i, line in enumerate(f_in):
            gen_obj = json.loads(line.strip())
            context = gen_obj['id'].strip()
            
            text = gen_obj['text'].strip()
            #sents = sent_tokenize(text)
            #gen = sents[0].replace('\n',' ')
            if '---' not in text:
                gen = ""
            else:
                gen = text.split('---')[0].split('\n')[0]

            id_list.append(i)
            context_list.append(context)
            gen_list.append(gen)
            if len(id_list) >= sample_numbers:
                break
    return id_list, context_list, gen_list

prev_context_list = None

all_res_dict = {}

for method_name in input_file_dict:
    file_name = input_file_dict[method_name]
    print(file_name)
    id_list, context_list, gen_list  = load_gen(file_name)
    print(method_name, sum([len(gen) for gen in gen_list ]) / sample_numbers )
    if prev_context_list is None:
        prev_context_list = context_list
        all_res_dict['id'] = id_list 
        all_res_dict['context'] = context_list 
    else:
        for i in range(len(context_list)):
            assert context_list[i] == prev_context_list[i], print(context_list[i], prev_context_list[i])
        prev_context_list = context_list
    all_res_dict['story_'+method_name] = gen_list

df = pd.DataFrame(all_res_dict)
print(df)

num_method = len(method_list)

output_dict = {'id': [], 'context': []}
for i in range(num_method):
    output_dict['story_'+str(i+1)] = []
    output_dict['source_'+str(i+1)] = []

#drop_idx = []

method_d2_len = {x:[] for x in method_list}

for index, row in df.iterrows():
    gen_list = []

    for method_name in method_list:
        gen_list.append(row['story_'+method_name].strip())
    if any([len(gen)<min_gen_char_len or len(gen)>max_gen_char_len for gen in gen_list]) or len(gen_list) != len(set(gen_list)):
        #drop_idx.append(index)
        continue
    output_dict['id'].append(row['id'])
    output_dict['context'].append(row['context'])
    idx_rnd = list(range(num_method))
    random.shuffle(idx_rnd)
    for i, idx in enumerate(idx_rnd):
        output_dict['story_'+str(i+1)].append(gen_list[idx])
        output_dict['source_'+str(i+1)].append(method_list[idx])
        method_d2_len[method_list[idx]].append(len(gen_list[idx]))

for method in method_d2_len:
    print( method, np.mean(method_d2_len[method]) )

df = pd.DataFrame(output_dict).set_index('id')
#df = df.drop(drop_idx)


print(df)
df.to_csv(output_csv)
