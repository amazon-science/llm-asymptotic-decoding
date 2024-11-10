import json

#input_file = 'outputs/multirc/train_456-fixedIds.json'
input_file = 'outputs/multirc/dev_83-fixedIds.json'

with open(input_file) as f_in:
    input_dict = json.load(f_in)


#output_f_name = './outputs/multirc/multirc_train.json'
output_f_name = './outputs/multirc/multirc_dev.json'

example = ' Passage: Sent 1: John likes to go hiking, and his wife likes to cook.\n Sent 2: His wife likes to cook.\n Here is a question: Who likes to cook?\n The answer is his wife\n\n'

with open(output_f_name, 'w') as f_out:
    for idx, data in enumerate(input_dict['data']):
        passage = data['paragraph']['text'].replace('<b>', ' ').replace('</b>', '').replace('<br>', '\n').replace('  ', ' ')
        for qa in data['paragraph']['questions']:
            q = qa['question']
            for a in qa['answers']:
                if a['isAnswer']:
                    a_text = a['text'].replace('  ', ' ')
                    prompt = example + ' Passage:' + passage + '\n Here is a question: ' + q + '\n The answer is'
                    f_out.write(json.dumps({'id': idx, 'passage': passage, 'question': q, 'prompt': prompt, 'answer': ' ' + a_text  }) + '\n')
                    
