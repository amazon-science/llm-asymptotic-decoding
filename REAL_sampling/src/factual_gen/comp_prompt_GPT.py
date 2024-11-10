import json
import os
import argparse
from openai import OpenAI
import random

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_result_path", type=str, default='')
    parser.add_argument("--eval_output_folder", type=str, default='')
    #parser.add_argument("--sample_num", type=int, default='2')
    #parser.add_argument("--sample_num", type=int, default='100')
    parser.add_argument("--sample_num", type=int, default='500')

    args = parser.parse_args()
    return args

args = parse_args()

openai_key = 'sk-proj-'
client = OpenAI(api_key=openai_key)

gpt_model_name = 'gpt-3.5-turbo-0125'

max_redo = 5

#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_wiki_OWT_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_fixed_wiki_OWT_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5.json'

#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5_w0.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3_lr-5.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_fixed_ROC_70M_bsz_128_exp_pred_last_a10_e3.json'

#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.2_p1.0_fixed_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3.json'
pred_result_path = 'outputs/GPT_exp/old/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_3.5_p1.0_fixed_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_CD_dt_1.0_p0.3_pythia-70m-deduped.json'
#pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_CD_topp_exp_1_win_40_dt_2.0_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3.json'

eval_output_folder = 'outputs/wp/GPT_exp/test_GPT3.5_responses_{}/'.format(args.sample_num)

input_file_name = os.path.basename(pred_result_path)
eval_output_path = eval_output_folder + input_file_name

#pred_result_path = args.pred_result_path
#eval_output_folder = args.eval_output_folder

os.makedirs(eval_output_folder, exist_ok=True)

if 'story' in pred_result_path:
    allow_ans_incomplete_str = ''
else:
    allow_ans_incomplete_str = 'It is ok that the continuation is not complete.\n'

system_prompt1 = """You are an English writing expert and you can compare and evaluate two continuations on these metrics with the following definitions -
    1. Fluency: Which continuation has better writing and grammar comparitively?
    2. Coherence: Which continuation has a better logical flow and the writing fits together with respect to the plot?
    3. Likability: Which continuation is more interesting and enjoyable to read?

You will be given two continuations - continuation A and continuation B.
Specify which continuation you prefer for each metric by responding with just the letter “A” or “B” followed by a hyphen and two line justifications for your preference.
{}
Assign an overall winner continuation as the letter “A” or “B” based on the category wins and provide two line justifications. 
IMPORTANT - DO NOT GIVE ANY OTHER TEXT APART FROM THE METRICS, PREFERENCE, AND JUSTIFICATIONO.

EXAMPLE OUTPUT 1:
Fluency: B
A: A has some complex sentences that are difficult to follow, with occasional grammatical errors.
B: B is well-written with minor grammatical mistakes and clear sentence structures.
Coherence: B
A: The plot of A is somewhat confusing and disjointed, especially with the sudden introduction of an old sage.
B: B maintains a coherent narrative, with each event logically building on the previous one, enhancing the continuation’s flow.
Likability: B
A: A is heartfelt but its erratic narrative structure detracts from its overall appeal.
B: B is compelling and maintains consistent character development, making it more enjoyable and engaging.
Overall Winner: B
A: A is moderately fluent, coherent, and interesting.
B: B is perfect except for some minor grammar issues.

EXAMPLE OUTPUT 2:
Fluency: A
A: A has a few minor grammatical issues, but overall, it demonstrates strong control of language.
B: B is well-written but has slightly more noticeable issues in grammar and sentence structure.
Coherence: A
A: B has a strong coherence, effectively conveying the progression of events.
B: A maintains a consistent and engaging narrative flow, though some parts are a bit abstract.
Likability: A
A: B’s realistic and emotional narrative is likely to resonate more with a wide range of readers.
B: A is imaginative and intriguing, but its abstract nature might not appeal to all readers.
Overall Winner: A
A: A is very good and it would be better if it can be more interesting.
B: B is too abstract to be interesting.
""".format(allow_ans_incomplete_str)

print(pred_result_path)
with open(pred_result_path) as f_in:
    pred_method_name, base_method_name, all_list = json.load(f_in)

id_list, context_list_pred, gen_list_pred, gen_list_base, ref_list = zip(*all_list[-args.sample_num:])

prompt_list = [] 
first_res_list = []
response_list = []
parse_win_list = []
bad_idx_list = []

def parse_results(res_list, response, idx, prefix, rand_num):
    err_return = ('', False)
    preference_res = res_list[idx].strip()
    if prefix not in preference_res:
        print('{} not in {}'.format(prefix, response))
        return err_return
    if (preference_res[-1] == 'A' and rand_num == 0) or (preference_res[-1] == 'B' and rand_num == 1):
        win = 'pred'
    elif (preference_res[-1] == 'A' and rand_num == 1) or (preference_res[-1] == 'B' and rand_num == 0):
        win = 'base'
    else:
        print('AB Preference not in {} for idx {}'.format(response, idx))
        return err_return
    return win, True

for i in range(len(id_list)):
    print(i / len(id_list))
    #rand_num = random.randint(0,1)
    rand_num = i % 2
    if rand_num == 0:
        first_res_list.append('pred')
        c1 = gen_list_pred[i]
        c2 = gen_list_base[i]
    else:
        first_res_list.append('base')
        c2 = gen_list_pred[i]
        c1 = gen_list_base[i]

    prompt_i = "Context: {}\n\n Continuation A: {} {}\n\n Continuation B: {} {}\n\n".format(context_list_pred[i], context_list_pred[i], c1, context_list_pred[i], c2)
    prompt_list.append(prompt_i)
    prompt_all = system_prompt1 + '\n\n' + prompt_i

    parse_win_result = {}
    response_org = ''
    for j in range(max_redo):
        try:
            completion = client.chat.completions.create(
              model=gpt_model_name,
              messages=[
                {"role": "user", "content": prompt_all}
              ]
            )
        except:
            break
        response_org = completion.choices[0].message.content
        response = response_org.replace('\n\n','\n')
        res_list = response.split('\n')
        if len(res_list) < 12:
            print("{} is too short".format(response))
            continue
        win_f, succeed_f = parse_results(res_list, response, 0, 'Fluency: ', rand_num)
        win_c, succeed_c = parse_results(res_list, response, 3, 'Coherence: ', rand_num)
        win_l, succeed_l = parse_results(res_list, response, 6, 'Likability: ', rand_num)
        win_o, succeed_o = parse_results(res_list, response, 9, 'Overall Winner: ', rand_num)
        if not succeed_f or not succeed_c or not succeed_l or not succeed_o:
            continue

        #print(completion.choices[0].message.content)

        parse_win_result = {'F': win_f, 'C': win_c, 'L': win_l, 'O': win_o}
        break
    
    response_list.append(response_org)

    parse_win_list.append(parse_win_result)
    
    if len(parse_win_result) == 0:
        print('bad input ', prompt_i)
        bad_idx_list.append(i)
    #    break

all_list_res = list(zip(id_list, context_list_pred, gen_list_pred, gen_list_base, ref_list, prompt_list, first_res_list, response_list, parse_win_list))
with open(eval_output_path, 'w') as f_out:
    json.dump( [pred_method_name, base_method_name, system_prompt1, bad_idx_list, all_list_res], f_out, indent=2)

