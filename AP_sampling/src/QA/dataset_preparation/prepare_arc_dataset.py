import pandas as pd
import datasets
import json
import ast

#dataset_config = {'name': 'allenai/ai2_arc', 'subset': 'ARC-Challenge', 'train_name': 'train' }
dataset_config = {'name': 'allenai/ai2_arc', 'subset': 'ARC-Easy', 'train_name': 'train' }

#validation
#output_f_name = './outputs/arc/arc_challenge_train.json'
#output_f_name = './outputs/arc/arc_easy_train.json'
#output_f_name = './outputs/arc/arc_neg_challenge_train.json'
output_f_name = './outputs/arc/arc_neg_easy_train.json'

dataset = datasets.load_dataset(dataset_config['name'], dataset_config['subset'] )
df_train = pd.DataFrame( dataset[ dataset_config['train_name'] ] )

print(len(df_train))

#example = ' Here is a question: What is the birthplace of Barack Obama?\n The answer is Honolulu, Hawaii.\n\n'
example = ' Question: Which kind of animals can fly?\n Answer: bird.\n\n'

with open(output_f_name, 'w') as f_out:
    for index, row in df_train.iterrows():
        #ans_dict = ast.literal_eval(row['answers'])
        #ans_dict = json.loads(row['answers'])
        ans = row['choices']['text'][ row['choices']['label'].index(row['answerKey']) ]
        #print(ans_str)
        #ans_dict = json.loads(ans_str)
        q = row['question']
        all_ans_raw = row['choices']['text'].copy()
        all_ans_raw.remove(ans)
        all_ans = [ans] + all_ans_raw
        all_ans = [' ' + x for x in all_ans]
        assert len(row['choices']['text']) == len(all_ans)
        #q = q[0].upper() + q[1:]
        prompt = example + ' Question: ' + q + '\n Answer:' 
        f_out.write(json.dumps({'id': index, 'question': q, 'prompt': prompt, 'answer': ' ' + ans, 'all_ans': all_ans  }) + '\n')
