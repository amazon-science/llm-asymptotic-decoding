import pandas as pd
import datasets
import json

dataset_config = {'name': 'EleutherAI/lambada_openai', 'subset': 'en', 'train_name': 'test' }

#validation
output_f_name = './outputs/lambda/openai_test.json'

dataset = datasets.load_dataset(dataset_config['name'], dataset_config['subset'] )
df_train = pd.DataFrame( dataset[ dataset_config['train_name'] ] )

print(len(df_train))

with open(output_f_name, 'w') as f_out:
    for index, row in df_train.iterrows():
        text = row['text']
        text_split = text.split(' ')
        prompt = ' '.join(text_split[:-1])
        ans = text_split[-1]
        f_out.write(json.dumps({'id': index, 'prompt': prompt, 'answer': ' ' + ans  }) + '\n')

