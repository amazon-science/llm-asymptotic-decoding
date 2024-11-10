import pandas as pd
import datasets
import json
import ast

dataset_config = {'name': 'allenai/social_i_qa', 'train_name': 'train' }
#dataset_config = {'name': 'allenai/social_i_qa', 'train_name': 'validation' }

#validation
#output_f_name = './outputs/socialiqa/socialiqa_val.json'
#output_f_name = './outputs/socialiqa/socialiqa_train.json'
output_f_name = './outputs/socialiqa/socialiqa_neg_train.json'

dataset = datasets.load_dataset(dataset_config['name'] )
df_train = pd.DataFrame( dataset[ dataset_config['train_name'] ] )

print(len(df_train))

#example = ' Here is a question: What is the birthplace of Barack Obama?\n The answer is Honolulu, Hawaii.\n\n'
#example = ' Question: Which kind of animals can fly?\n Answer: bird.\n\n'
example = ' Passage: John likes to go hiking, and his wife likes to cook.\n Question: Who likes to cook?\n Answer: his wife\n\n'

label_dict = {'1': 'A', '2': 'B', '3': 'C'}

with open(output_f_name, 'w') as f_out:
    for index, row in df_train.iterrows():
        #ans_dict = ast.literal_eval(row['answers'])
        #ans_dict = json.loads(row['answers'])
        #print(row)
        ans = row[ 'answer' + label_dict[ row['label'] ] ]
        all_ans_raw = [row['answer' + x]  for x in ['A','B','C']]
        all_ans_raw.remove(ans)
        all_ans = [ans] + all_ans_raw
        all_ans = [' ' + x for x in all_ans]
        assert 3 == len(all_ans)
        #print(ans_str)
        #ans_dict = json.loads(ans_str)
        prompt = example + ' Passage: ' + row['context'] + '\n Question: ' + row['question'] + '\n Answer:'
        f_out.write(json.dumps({'id': index, 'question': row['question'], 'context': row['context'], 'prompt': prompt, 'answer': ' ' + ans, 'all_ans': all_ans  }) + '\n')
