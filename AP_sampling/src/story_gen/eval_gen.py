import pandas as pd
import json
from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM
)
import numpy as np
import math
import torch
import evaluate
import sentence_transformers
from nltk.tokenize import word_tokenize, sent_tokenize
import os

import argparse

#sample_numbers = 1000
#sample_numbers = 100
sample_numbers = 10000000000000

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda_idx", type=int, default=0)
    parser.add_argument("--gen_max_length", type=int, default=128)
    parser.add_argument("--pred_result_path", type=str, default='')
    parser.add_argument("--eval_output_folder", type=str, default='scores')
    parser.add_argument("--num_eval_sent", type=int, default=3)

    args = parser.parse_args()
    return args

args = parse_args()

#sent_num = 3
sent_num = args.num_eval_sent
#eval_num = 4

#pred_results = "outputs/story/story_start2_100_6.9b_CD_topp_dt_0.5_p0.2_pythia-70m-deduped/story_start2_100_6.9b_CD_topp_p0.2_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_CD_dt_0.5_p0.025_pythia-70m-deduped/story_start2_1000_6.9b_CD_p0.025_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_topp_p0.95_temp_1.0/story_start2_1000_6.9b_topp_p0.95_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_fe_CD_topp_exp_1_win_40_dt_16.0_p0.5_fixed_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3/story_start2_1000_6.9b_fe_CD_topp_p0.5_gen_seed1.jsonl"
#pred_results ="outputs/story/story_start2_1000_6.9b_fe_AP_topp_exp_1_win_40_dt_16.0_ait1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3_fixed_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/story_start2_1000_6.9b_fe_AP_topp_p1.0_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_CD_topp_dt_0.5_p0.4_pythia-70m-deduped/story_start2_1000_6.9b_CD_topp_p0.4_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_CD_topp_dt_0.5_p0.6_pythia-70m-deduped/story_start2_1000_6.9b_CD_topp_p0.6_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_AP_topp_p0.4_dt_1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/story_start2_1000_6.9b_AP_topp_p0.4_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_AP_topp_p0.4_dt_1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/story_start2_1000_6.9b_AP_topp_p0.4_gen_seed2.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_AP_topp_p0.6_dt_1.0_prob_wiki_ext2_1e6_70M_bsz_64_e5_only_top_last_w_10_l1_reg_w_08_logit_exp_decay_lr-4/story_start2_1000_6.9b_AP_topp_p0.6_gen_seed1.jsonl"
#pred_results = "outputs/story/story_start2_1000_6.9b_topp_p0.6_temp_1.0/story_start2_1000_6.9b_topp_p0.6_gen_seed1.jsonl"

pred_results = args.pred_result_path

#output_path = "outputs/story/story_start2_100_6.9b_CD_topp_dt_0.5_p0.2_pythia-70m-deduped/scores/story_start2_100_6.9b_CD_topp_p0.2_gen_seed1.jsonl"

output_path = os.path.dirname(pred_results) + '/' +  args.eval_output_folder + '/' + os.path.basename(pred_results)

ppl_model_name = 'openlm-research/open_llama_3b_v2'
sim_model_name = 'all-mpnet-base-v2'

gen_max_length = args.gen_max_length
batch_size_ppl = 4

device_id_for_mauve = args.cuda_idx
device_1 = torch.device("cuda:"+str(device_id_for_mauve) if torch.cuda.is_available() else "cpu")
#device_2 = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

tokenizer_ppl = AutoTokenizer.from_pretrained(ppl_model_name)
tokenizer_ppl.pad_token = tokenizer_ppl.eos_token
model_ppl = AutoModelForCausalLM.from_pretrained(ppl_model_name)
model_ppl = model_ppl.to(device_1)
model_ppl.eval()

sim_model = sentence_transformers.SentenceTransformer(sim_model_name, device = device_1)

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


def compute_sim(decoded_preds, decoded_labels):
    decoded_preds_emb = sim_model.encode(decoded_preds, convert_to_numpy=True)
    decoded_label_emb = sim_model.encode(decoded_labels, convert_to_numpy=True)
    score_arr = []
    for i in range(len(decoded_preds)):
        if len(decoded_label_emb) == 1:
            cosine_scores = sentence_transformers.util.cos_sim(decoded_preds_emb[i], decoded_label_emb[0]).squeeze().tolist()
        else:
            cosine_scores = sentence_transformers.util.cos_sim(decoded_preds_emb[i], decoded_label_emb[i]).squeeze().tolist()
        score_arr.append(cosine_scores)
    return score_arr

metric = evaluate.load("rouge")
metric_mauve = evaluate.load("mauve")

def compute_sim_metric(decoded_preds, decoded_labels):
    score_arr = compute_sim(decoded_preds, decoded_labels)
    #scores_avg = np.mean(score_arr)
    scores_avg, scores_se = weighted_avg_and_se(score_arr)
    return scores_avg, scores_se

def weighted_avg_and_se(values, weights=None):
    #https://stackoverflow.com/questions/2413522/weighted-standard-deviation-in-numpy
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, math.sqrt(variance/len(values)))

def perplexity(generated_sents):
    device = next(model_ppl.parameters()).device
    output_story_len = gen_max_length
    input_ids_all = tokenizer_ppl(generated_sents, truncation=True, padding="max_length", max_length=output_story_len, return_tensors="pt").input_ids.to(device)
    nb_batches = math.ceil( input_ids_all.size(0) /batch_size_ppl)

    ppl_arr = []
    weight_arr = []
    for i in range(nb_batches):
        input_ids = input_ids_all[i*batch_size_ppl:(i+1)*batch_size_ppl, :]
        label_ids = torch.clone(input_ids)
        label_ids[label_ids==tokenizer_ppl.pad_token] = -100 #let the ppl ignore the eos loss
        outputs_GPT2LMHeadModel= model_ppl(input_ids, labels=label_ids)
        ppl = outputs_GPT2LMHeadModel[0].item()
        ppl_arr.append(ppl)
        weight_arr.append(input_ids.size(0))
    ppl_avg, ppl_se = weighted_avg_and_se(ppl_arr, weight_arr)
    return ppl_avg, ppl_se


def compute_metrics(decoded_preds, decoded_labels, metric_prefix, eval_output_path):
    # Some simple post-processing
    #decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    #result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #result = {k: round(v * 100, 4) for k, v in result.items()}
    result_all = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True, use_aggregator = False)
    result = {}
    for k, v in result_all.items():
        rouge_avg, rouge_se = weighted_avg_and_se(v)
        result[metric_prefix + '_' +k] = rouge_avg*100
        result[metric_prefix + '_' +k+'_se'] = rouge_se*100

    try:
        result[metric_prefix + '_'+'text_sim'], result[metric_prefix+'text_sim_se'] = compute_sim_metric(decoded_preds, decoded_labels) #ADD for joint
    except:
        result[metric_prefix + '_'+'text_sim'] = -1
        result[metric_prefix + '_'+'text_sim_se'] = -1
    result["ppl"], result["ppl_se"]= perplexity(decoded_preds)

    #result["dist_1"], result["dist_1_se"] = compute_distinct_n_arr(decoded_preds, 1)
    #result["dist_2"], result["dist_2_se"] = compute_distinct_n_arr(decoded_preds, 2)

    result_mauve = metric_mauve.compute(predictions=decoded_preds, references=decoded_labels, device_id=device_id_for_mauve)
    result[metric_prefix + '_'+'mauve'] = result_mauve.mauve

    prediction_lens = [len(word_tokenize(pred)) for pred in decoded_preds]
    result["gen_len"] = np.mean(prediction_lens)
    print(result)
    with open(eval_output_path, 'a') as f_out:
        f_out.write(json.dumps(result) + '\n')

    return result

#df = pd.read_csv(pred_results)

id_list, context_list, gen_list, ref_list = load_gen(pred_results)

#print(gen_list[:3])
#print(ref_list[:3])
#print(context_list[:3])

metric_prefix = 'ref'
results = compute_metrics(gen_list, ref_list, metric_prefix, output_path.replace('.jsonl',  '_' + metric_prefix+ '.jsonl'))

metric_prefix = 'prompt'
results = compute_metrics(gen_list, context_list, metric_prefix, output_path.replace('.jsonl',  '_' + metric_prefix+ '.jsonl') )

#print('div d1', compute_diversity(df['output_pred'].tolist(), df['input_prompt'].tolist(), 1))
#print('div d2', compute_diversity(df['output_pred'].tolist(), df['input_prompt'].tolist(), 2))


def compute_diversity(decoded_preds, decoded_inputs, n):
    current_input = decoded_inputs[0]
    current_text = []
    dist_n_arr = []
    for i in range(len(decoded_preds)):
        if decoded_inputs[i] != current_input:
            dist_n = compute_distinct_n_single('\n'.join(current_text), n)
            dist_n_arr.append(dist_n)
            current_input = decoded_inputs[i]
            current_text = []
        current_text.append(decoded_preds[i])
    if len(current_text) > 1:
        dist_n = compute_distinct_n_single('\n'.join(current_text), n)
        dist_n_arr.append(dist_n)
    print(len(decoded_preds), '=', eval_num, 'X 2 X', len(dist_n_arr))
    assert len(decoded_preds) == eval_num*2*len(dist_n_arr)

    return weighted_avg_and_se(dist_n_arr)

def update_ngrams(tokens, ngrams, n):
    # Update ngram dict based on new list of tokens
    for i in range(len(tokens) - (n-1)):
        ngram = ' '.join(tokens[i:i + n])
        ngrams.append(ngram)
    return ngrams

def compute_distinct_n_single(text, n):
    ngrams = []
    tokens = [x.lower() for x in word_tokenize(text)]
    #tokens = [x for x in simple_tokenize(text.lower()) if x not in stop_word_org_set]
    ngrams = update_ngrams(tokens, ngrams, n)
    vocab_n = len(set(ngrams))
    return vocab_n / float(len(ngrams))

def compute_distinct_n_arr(text_arr, n):
    dist_n_arr = []
    for text in text_arr:
        dist_n = compute_distinct_n_single(text, n)
        dist_n_arr.append(dist_n)
    return weighted_avg_and_se(dist_n_arr)
