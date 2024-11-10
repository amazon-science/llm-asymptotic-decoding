import pandas as pd
import datasets
import json
import ast

dataset_config = {'name': 'allenai/qasc', 'train_name': 'train' }

#validation
#output_f_name = './outputs/qasc/qasc_train.json'
#output_f_name = './outputs/qasc/qasc_fact_early_train.json'
output_f_name = './outputs/qasc/qasc_neg_train.json'
#output_f_name = './outputs/qasc/qasc_neg_fact_train.json'

dataset = datasets.load_dataset(dataset_config['name'] )
df_train = pd.DataFrame( dataset[ dataset_config['train_name'] ] )

#including_passage = True
including_passage = False

print(len(df_train))

#include_facts = True
include_facts = False

#example = ' Here is a question: What is the birthplace of Barack Obama?\n The answer is Honolulu, Hawaii.\n\n'

if include_facts:
    example = ' Question: Which kind of animals can fly?\n Fact 1: a bird is an animal.\n Fact 2: birds can fly.\n Answer: bird.\n\n'
    #example = ' Fact 1: a bird is an animal.\n Fact 2: birds can fly.\n Question: Which kind of animals can fly?\n Answer: bird.\n\n'
else:
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
        if include_facts:
            #prompt = example + ' Fact 1: ' + row['fact1'] + '\n Fact 2: ' + row['fact2'] + '\n Question: ' + q + '\n Answer:' 
            prompt = example + ' Question: ' + q + '\n Fact 1: ' + row['fact1'] + '\n Fact 2: ' + row['fact2'] + '\n Answer:' 
            f_out.write(json.dumps({'id': index, 'question': q, 'prompt': prompt, 'answer': ' ' + ans, 'all_ans': all_ans  }) + '\n')
        else:
            prompt = example + ' Question: ' + q + '\n Answer:' 
            f_out.write(json.dumps({'id': index, 'question': q, 'prompt': prompt, 'answer': ' ' + ans, 'all_ans': all_ans  }) + '\n')

