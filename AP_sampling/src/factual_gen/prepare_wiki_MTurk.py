import json
import pandas as pd
from nltk.tokenize import sent_tokenize
import random

sample_numbers = 1000

repo_dir = '/mnt/efs/Haw-Shiuan/true_entropy/'

input_file_dict = {'AP': repo_dir + "outputs/factual_gen/factual_test7k_6.9b_AP_toppk20_log_p0.8_dt_1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/factual_test7k_6.9b_AP_toppk20_p0.8_gen_seed1.jsonl",
                   'Top-p': repo_dir + 'outputs/factual_gen/factual_test7k_6.9b_topp_p1.0_temp_1.0/factual_test7k_6.9b_topp_p1.0_gen_seed1.jsonl',
                   'CD': repo_dir + 'outputs/factual_gen/factual_test7k_6.9b_CD_toppk20_dt_0.5_log_p0.8_pythia-70m-deduped/factual_test7k_6.9b_CD_toppk20_p0.8_gen_seed1.jsonl', 
                   'REAL+CD': repo_dir + 'outputs/factual_gen/factual_test7k_6.9b_fe_CD_topp_exp_1_win_40_dt_4.0_p0.5_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/factual_test7k_6.9b_fe_CD_topp_p0.5_gen_seed1.jsonl',
                   'REAL+AP': repo_dir + 'outputs/factual_gen/factual_test7k_6.9b_fe_AP_topp_exp_1_win_40_dt_4.0_ait1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/factual_test7k_6.9b_fe_AP_topp_p1.0_gen_seed1.jsonl'
                   }

output_csv = 'outputs/MTurk/wiki/gen_1000.csv'

method_list = list(input_file_dict.keys())

def load_gen(input_file):
    id_list = []
    context_list = []
    gen_list = []
    with open(input_file) as f_in:
        for line in f_in:
            gen_obj = json.loads(line.strip())
            context = gen_obj['prompt'].strip()
            id_res = int(gen_obj['id'])
            
            text = gen_obj['text'].strip()
            sents = sent_tokenize(text)
            gen = sents[0].replace('\n',' ')

            id_list.append(id_res)
            context_list.append(context)
            gen_list.append(gen)
            if len(id_list) >= sample_numbers:
                break
    return id_list, context_list, gen_list

prev_id_list = None

all_res_dict = {}

for method_name in input_file_dict:
    file_name = input_file_dict[method_name]
    print(file_name)
    id_list, context_list, gen_list  = load_gen(file_name)
    print(method_name, sum([len(gen) for gen in gen_list ]) / sample_numbers )
    if prev_id_list is None:
        prev_id_list = id_list
        all_res_dict['id'] = id_list 
        all_res_dict['context'] = context_list 
    else:
        for i in range(len(id_list)):
            assert id_list[i] == prev_id_list[i]
        prev_id_list = id_list
    all_res_dict['gen_'+method_name] = gen_list

df = pd.DataFrame(all_res_dict)
print(df)

num_method = len(method_list)

output_dict = {'id': [], 'context': []}
for i in range(num_method):
    output_dict['gen_'+str(i+1)] = []
    output_dict['method_'+str(i+1)] = []

#drop_idx = []

for index, row in df.iterrows():
    gen_list = []

    for method_name in method_list:
        gen_list.append(row['gen_'+method_name])
    if any([len(gen)<10 or 'External links' in gen for gen in gen_list]) or len(gen_list) != len(set(gen_list)):
        #drop_idx.append(index)
        continue
    output_dict['id'].append(row['id'])
    output_dict['context'].append(row['context'])
    idx_rnd = list(range(num_method))
    random.shuffle(idx_rnd)
    for i, idx in enumerate(idx_rnd):
        output_dict['gen_'+str(i+1)].append(gen_list[idx])
        output_dict['method_'+str(i+1)].append(method_list[idx])

df = pd.DataFrame(output_dict).set_index('id')
#df = df.drop(drop_idx)


print(df)
df.to_csv(output_csv)
