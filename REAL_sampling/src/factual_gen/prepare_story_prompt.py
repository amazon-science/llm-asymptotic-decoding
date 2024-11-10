import json
import pandas as pd

input_stories = "/mnt/efs/Haw-Shiuan/entailment_tree/datasets/ROCStories__spring2016.csv"
num_stories = 1000
shot_num = 3
prompt_sent_num = 2
output_prompt_file = "/mnt/efs/Haw-Shiuan/true_entropy/outputs/MTurk/story/prompt_start2_b2_{}.jsonl".format(num_stories)

delimiter = '---'
num_story_line = 5

df = pd.read_csv(input_stories)
df_sampled_stories = df.sample(n=num_stories, replace=False)
df_rest = df.drop(df_sampled_stories.index)

def prepare_id(row, prompt_sent_num):
    id_q = ''
    for i in range(prompt_sent_num):
        id_q += row['sentence'+str(i+1)] + ' '
    return id_q[:-1]

def str_story(row_examples, i, delimiter):
    story_str = 'Story {}:\n'.format(i+1)
    for i in range(num_story_line):
        story_str += row_examples['sentence'+str(i+1)] + ' '
    story_str += '\n' + delimiter + '\n'
    return story_str

output_list = []
for index, row in df_sampled_stories.iterrows():
    out_dict = {}
    id_q = prepare_id(row, prompt_sent_num)
    out_dict['id'] = id_q
    df_examples = df_rest.sample(n=shot_num, replace=False)
    prompt_str = ' Here are {} stories. Each story has five sentences.\n\n'.format(shot_num+1)
    for i, (index, row_examples) in enumerate(df_examples.iterrows()):
        prompt_str += str_story(row_examples, i, delimiter) 
    
    out_dict['prompt'] = prompt_str + 'Story {}:\n'.format(shot_num+1) + id_q
    output_list.append(out_dict)

with open(output_prompt_file, 'w') as f_out:
    for out_dict in output_list:
        f_out.write(json.dumps(out_dict) + '\n' )

