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


pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_topp_exp_1_win_40_dt_1.8_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3.json'
pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_CD_dt_1.0_p0.3_pythia-70m-deduped.json'
pred_result_path = 'outputs/GPT_exp/res_pair/story_start2_1000_6.9b_fe_CD_topp_exp_1_win_40_dt_2.0_p1.0_OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3.json'

eval_output_folder = 'outputs/GPT_exp/GPT3.5_responses_{}/'.format(args.sample_num)


input_file_name = os.path.basename(pred_result_path)
eval_output_path = eval_output_folder + input_file_name

#pred_result_path = args.pred_result_path
#eval_output_folder = args.eval_output_folder

os.makedirs(eval_output_folder, exist_ok=True)

#allow_ans_incomplete_str = 'It is ok that the continuation is not complete.\n'
allow_ans_incomplete_str = ''

system_prompt1 = """You are an English writing expert and you can compare and evaluate two continuations on these metrics with the following definitions -
    1. Fluency: Which continuation has better writing and grammar comparitively?
    2. Coherence: Which continuation has a better logical flow and the writing fits together with respect to the plot?
    3. Likability: Which continuation is more interesting and enjoyable to read?

You will be given two continuations - continuation A and continuation B.
Add a rating out of 5 for each category, specify which continuation you prefer for each metric by responding with just the letter “A” or “B” followed by a hyphen and one line reasoning for your preference.
For each category, provide a category winner continuation as the letter “A” or “B”, based on the category ratings.
{}
Assign an overall winner continuation as the letter “A” or “B” based on the ratings and category wins. 
Finally, provide overall rate for each continuation and their justification.
IMPORTANT - DO NOT GIVE ANY OTHER TEXT APART FROM THE SCORE, METRICS AND PREFERENCE.

EXAMPLE OUTPUT 1:
Fluency: B
A - 3/5: A has some complex sentences that are difficult to follow, with occasional grammatical errors.
B - 4/5: B is well-written with minor grammatical mistakes and clear sentence structures.
Coherence: B
A - 2/5: The plot of A is somewhat confusing and disjointed, especially with the sudden introduction of an old sage.
B - 5/5: B maintains a coherent narrative, with each event logically building on the previous one, enhancing the continuation’s flow.
Likability: B
A - 3/5: A is heartfelt but its erratic narrative structure detracts from its overall appeal.
B - 5/5: B is compelling and maintains consistent character development, making it more enjoyable and engaging.
Overall Winner: B
A - 3/5: A is moderately fluent, coherent, and interesting.
B - 4.5/5: B is perfect except for some minor grammar issues.

EXAMPLE OUTPUT 2:
Fluency: A
A - 5/5: A has a few minor grammatical issues, but overall, it demonstrates strong control of language.
B - 4/5: B is well-written but has slightly more noticeable issues in grammar and sentence structure.
Coherence: A
A - 4.5/5: B has a strong coherence, effectively conveying the progression of events.
B - 4.5/5: A maintains a consistent and engaging narrative flow, though some parts are a bit abstract.
Likability: A
A - 4/5: B’s realistic and emotional narrative is likely to resonate more with a wide range of readers.
B - 3.5/5: A is imaginative and intriguing, but its abstract nature might not appeal to all readers.
Overall Winner: A
A - 4.5/5: A is very good and it would be better if it can be more interesting.
B - 3.5/5: B is too abstract to be interesting.
""".format(allow_ans_incomplete_str)

print(pred_result_path)
with open(pred_result_path) as f_in:
    pred_method_name, base_method_name, all_list = json.load(f_in)

id_list, context_list_pred, gen_list_pred, gen_list_base, ref_list = zip(*all_list[:args.sample_num])

prompt_list = [] 
first_res_list = []
response_list = []
parse_win_list = []
parse_score_pred_list = []
parse_score_base_list = []
bad_idx_list = []

def get_scores(res_list, idx, response):
    A_str = res_list[idx]
    score_loc = A_str.find('/5') - 1
    if score_loc < 0:
        print('/5 not in {}  for idx {}'.format(response, idx))
        return -1
    else:
        if ' ' not in A_str[:score_loc+1]:
            print('no space before /5 {}  for idx {}'.format(response, idx))
            return -1
        result_list = A_str[:score_loc+1].split(' ')
        try:
            return float(result_list[-1])
        except ValueError:
            print('cannot parse number before /5 {}  for idx {}'.format(response, idx))
            return -1

        #if A_str[score_loc - 1] == '.':
        #    return float(A_str[score_loc-2:score_loc+1])
        #else:
        #    return float(A_str[score_loc])

def parse_results(res_list, response, idx, prefix, rand_num, only_get_preference=False):
    err_return = ('', -1, -1, False)
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
    if only_get_preference:
        return win, -1, -1, True
    score_A = get_scores(res_list, idx+1, response)
    score_B = get_scores(res_list, idx+2, response)
    if score_A == -1 or score_B == -1:
        return err_return
    if rand_num == 0:
        score_pred = score_A
        score_base = score_B
    else:
        score_pred = score_B
        score_base = score_A

    return win, score_pred, score_base, True


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
    parse_score_pred_result = {}
    parse_score_base_result = {}
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
        win_f, score_pred_f, score_base_f, succeed_f = parse_results(res_list, response, 0, 'Fluency: ', rand_num)
        win_c, score_pred_c, score_base_c, succeed_c = parse_results(res_list, response, 3, 'Coherence: ', rand_num)
        win_l, score_pred_l, score_base_l, succeed_l = parse_results(res_list, response, 6, 'Likability: ', rand_num)
        #win_o, dummy, dummy, succeed = parse_results(res_list, response, 9, 'Overall Winner: ', rand_num, True)
        win_o, score_pred_o, score_base_o, succeed_o = parse_results(res_list, response, 9, 'Overall Winner: ', rand_num)
        if not succeed_f or not succeed_c or not succeed_l or not succeed_o:
            continue

        #print(completion.choices[0].message.content)

        parse_win_result = {'F': win_f, 'C': win_c, 'L': win_l, 'O': win_o}
        parse_score_pred_result = {'F': score_pred_f, 'C': score_pred_c, 'L': score_pred_l, 'O': score_pred_o}
        parse_score_base_result = {'F': score_base_f, 'C': score_base_c, 'L': score_base_l, 'O': score_base_o}
        break
    
    response_list.append(response_org)

    parse_win_list.append(parse_win_result)
    parse_score_pred_list.append(parse_score_pred_result)
    parse_score_base_list.append(parse_score_base_result)
    
    if len(parse_win_result) == 0:
        print('bad input ', prompt_i)
        bad_idx_list.append(i)
    #    break

all_list_res = list(zip(id_list, context_list_pred, gen_list_pred, gen_list_base, ref_list, prompt_list, first_res_list, response_list, parse_win_list, parse_score_pred_list, parse_score_base_list))
with open(eval_output_path, 'w') as f_out:
    json.dump( [pred_method_name, base_method_name, system_prompt1, bad_idx_list, all_list_res], f_out, indent=2)


