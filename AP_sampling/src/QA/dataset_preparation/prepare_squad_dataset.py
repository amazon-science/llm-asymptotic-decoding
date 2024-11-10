import pandas as pd
import datasets
import json

#dataset_config = {'name': 'rajpurkar/squad', 'train_name': 'train' }
dataset_config = {'name': 'rajpurkar/squad', 'train_name': 'validation' }

#validation
#output_f_name = './outputs/squad/squad_train.json'
#output_f_name = './outputs/squad/squad_val.json'
output_f_name = './outputs/squad/squad_val_no_pass.json'

dataset = datasets.load_dataset(dataset_config['name'] )
df_train = pd.DataFrame( dataset[ dataset_config['train_name'] ] )

including_passage = False

print(len(df_train))

if including_passage:
    example = ' Passage: John likes to go hiking, and his wife likes to cook.\n Here is a question: Who likes to cook?\n The answer is his wife\n\n'
else:
    example = ' Here is a question: What is the birthplace of Barack Obama?\n The answer is Honolulu, Hawaii.\n\n'

with open(output_f_name, 'w') as f_out:
    for index, row in df_train.iterrows():
        ans_dict = row['answers']
        #print(ans_str)
        #ans_dict = json.loads(ans_str)
        for ans in ans_dict['text']:
            if including_passage:
                prompt = example + ' Passage: ' + row['context'] + '\n Here is a question: ' + row['question'] + '\n The answer is' 
                f_out.write(json.dumps({'id': row['id'], 'passage': row['context'], 'question': row['question'], 'prompt': prompt, 'answer': ' ' + ans  }) + '\n')
            else:
                prompt = example + ' Here is a question: ' + row['question'] + '\n The answer is' 
                f_out.write(json.dumps({'id': row['id'], 'question': row['question'], 'prompt': prompt, 'answer': ' ' + ans  }) + '\n')

