import json
import os
import argparse
from nltk.tokenize import word_tokenize, sent_tokenize

#sample_numbers = 1000
#sample_numbers = 100
sample_numbers = 10000000000000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_result_path", type=str, default='')
    parser.add_argument("--baseline_result_path", type=str, default='')
    parser.add_argument("--eval_output_folder", type=str, default='')
    parser.add_argument("--num_eval_sent", type=int, default=3)

    args = parser.parse_args()
    return args

args = parse_args()

#sent_num = 3
sent_num = args.num_eval_sent
#eval_num = 4

eval_output_folder = 'outputs/GPT_exp/res_pair/'

sample_numbers = 1000
#baseline_results = 'outputs/factual_gen/story_start2_{}_6.9b_topp_p0.3_temp_1.0/story_start2_{}_6.9b_topp_p0.3_gen_seed1.jsonl'.format(sample_numbers,sample_numbers)
#pred_results = "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_wiki_OWT_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers)
#pred_results = "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5_w0/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers)
#pred_results = "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers)
baseline_results = 'outputs/factual_gen/story_start2_{}_6.9b_topp_p0.5_temp_1.0/story_start2_{}_6.9b_topp_p0.5_gen_seed1.jsonl'.format(sample_numbers,sample_numbers)
#pred_results = "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers)
pred_results = "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_fixed_wiki_OWT_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers)

#pred_results = "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers)
#baseline_results = 'outputs/factual_gen/story_start2_{}_6.9b_topp_p0.7_temp_1.0/story_start2_{}_6.9b_topp_p0.7_gen_seed1.jsonl'.format(sample_numbers,sample_numbers)
#pred_results = "outputs/factual_gen/story_start2_{}_6.9b_fe_topp_exp_1_win_40_dt_3.5_p1.0_fixed_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_start2_{}_6.9b_fe_topp_p1.0_gen_seed1.jsonl".format(sample_numbers,sample_numbers)
#pred_results = 'outputs/factual_gen/story_start2_{}_6.9b_CD_dt_1.0_p0.3_pythia-70m-deduped/story_start2_{}_6.9b_CD_p0.3_gen_seed1.jsonl'.format(sample_numbers,sample_numbers)
#pred_results = 'outputs/factual_gen/story_start2_{}_6.9b_fe_CD_topp_exp_1_win_40_dt_2.0_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_start2_{}_6.9b_fe_CD_topp_p1.0_gen_seed1.jsonl'.format(sample_numbers,sample_numbers)


os.makedirs(eval_output_folder, exist_ok=True)

#baseline_results = args.baseline_result_path
#pred_results = args.pred_result_path
#eval_output_folder = args.eval_output_folder

print(pred_results)

def load_gen(input_file):
    id_list = []
    ref_list = []
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
                gen = ' '.join(sent_tokenize(text)[:sent_num])
            else:
                gen = text.split('---')[0].split('\n')[0]

            if len(gen.strip()) == 0:
                gen = "None" #MAUVE cannot handle empty input
            id_list.append(i)
            context_list.append(context)
            ref_list.append('dummy')
            gen_list.append(gen)
            if len(id_list) >= sample_numbers:
                break
    return id_list, context_list, gen_list, ref_list


id_list, context_list_pred, gen_list_pred, ref_list = load_gen(pred_results)
id_list, context_list_base, gen_list_base, ref_list = load_gen(baseline_results)

base_method_name = baseline_results.split('/')[-2]
pred_method_name = pred_results.split('/')[-2]
all_list = list(zip(id_list, context_list_pred, gen_list_pred, gen_list_base, ref_list))

eval_output_path = eval_output_folder + pred_method_name + '.json'
with open(eval_output_path, 'w') as f_out:
    json.dump( [pred_method_name, base_method_name, all_list], f_out, indent=2)
