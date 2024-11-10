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
    #parser.add_argument("--num_eval_sent", type=int, default=3)
    #parser.add_argument("--num_eval_sent", type=int, default=1)
    parser.add_argument("--num_eval_sent", type=int, default=5)

    args = parser.parse_args()
    return args

args = parse_args()

seed = 1

#sent_num = 3
sent_num = args.num_eval_sent
#eval_num = 4

eval_output_folder = 'outputs/story/GPT_exp/res_pair/'
#eval_output_folder = 'outputs/story/GPT_exp/res_pair_{}/'.format(seed)



baseline_results = "outputs/story/story_start2_1000_6.9b_topp_p0.95_temp_1.0/story_start2_1000_6.9b_topp_p0.95_gen_seed1.jsonl"

pred_results = "outputs/story/story_start2_1000_6.9b_AP_toppk20_p0.8_dt_1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/story_start2_1000_6.9b_AP_toppk20_p0.8_gen_seed1.jsonl"

#pred_results = "outputs/story/story_start2_1000_6.9b_AP_topp_p0.8_dt_1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/story_start2_1000_6.9b_AP_topp_p0.8_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_AP_topp_p0.8_dt_1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_1_logit_exp_decay_lr-4/story_start2_1000_6.9b_AP_topp_p0.8_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_CDk20_dt_0.5_p0.1_pythia-70m-deduped/story_start2_1000_6.9b_CDk20_p0.1_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_CD_toppk20_dt_0.75_p0.8_pythia-70m-deduped/story_start2_1000_6.9b_CD_toppk20_p0.8_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_CD_toppk20_dt_0.5_p0.8_pythia-70m-deduped/story_start2_1000_6.9b_CD_toppk20_p0.8_gen_seed1.jsonl"
# pred_results = "outputs/story/story_start2_1000_6.9b_CD_toppk20_dt_0.5_p0.8_pythia-70m-deduped/story_start2_1000_6.9b_CD_toppk20_p0.8_gen_seed2.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_CD_toppk20_dt_0.5_p0.4_pythia-70m-deduped/story_start2_1000_6.9b_CD_toppk20_p0.4_gen_seed1.jsonl"

#pred_results = "outputs/story/story_start2_1000_6.9b_topp_p0.4_temp_1.0/story_start2_1000_6.9b_topp_p0.4_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_toppk20_p1.0_temp_1.0/story_start2_1000_6.9b_toppk20_p0.4_gen_seed1.jsonl"


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
            if 'story_' in pred_results:
                context = gen_obj['id'].strip()
            else:
                context = gen_obj['prompt'].strip()

            text = gen_obj['text'].strip()
            ref = gen_obj['ref'].strip()
            #sents = sent_tokenize(text)
            #gen = sents[0].replace('\n',' ')
            if 'story_' in pred_results:
                if '---' not in text:
                    gen = ' '.join(sent_tokenize(text)[:sent_num])
                else:
                    gen = text.split('---')[0].split('\n')[0]
            else:
                gen = ' '.join(sent_tokenize(text)[:sent_num])
                ref = ' '.join(sent_tokenize(ref)[:sent_num])

            if len(gen.strip()) == 0:
                gen = "None" #MAUVE cannot handle empty input
            if len(ref.strip()) == 0:
                ref = "None"
            id_list.append(i)
            ref_list.append(ref)
            context_list.append(context)
            gen_list.append(gen)
            if len(id_list) >= sample_numbers:
                break
    return id_list, context_list, gen_list, ref_list


id_list, context_list_pred, gen_list_pred, ref_list = load_gen(pred_results)
id_list, context_list_base, gen_list_base, ref_list = load_gen(baseline_results)

base_method_name = baseline_results.split('/')[-2]
pred_method_name = pred_results.split('/')[-2]
all_list = list(zip(id_list, context_list_pred, gen_list_pred, gen_list_base, ref_list))

eval_output_path = eval_output_folder + pred_method_name + '_s{}.json'.format(seed)
with open(eval_output_path, 'w') as f_out:
    json.dump( [pred_method_name, base_method_name, all_list], f_out, indent=2)
