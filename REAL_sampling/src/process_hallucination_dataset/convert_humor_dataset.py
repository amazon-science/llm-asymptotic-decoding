import pandas as pd

from transformers import AutoTokenizer

#input_path = "/mnt/efs/Haw-Shiuan/rJokesData/data/dev.tsv"
input_path = "/mnt/efs/Haw-Shiuan/rJokesData/data/test.tsv"

#output_path = "/mnt/efs/Haw-Shiuan/true_entropy/outputs/humor/all_128_train.csv"
output_path = "/mnt/efs/Haw-Shiuan/true_entropy/outputs/humor/all_128_val.csv"

cut_end = True

if cut_end:
    #max_token_num = 2048
    #max_token_num = 1024
    max_token_num = 128
    small_model_name = 'EleutherAI/pythia-70m-deduped'
    tokenizer = AutoTokenizer.from_pretrained(small_model_name, truncation_side='left')

label_reg_arr = []
label_arr = []
text_arr = [] 
cat_arr = [] 

def preprocessing_text(text):
    text_tok = tokenizer.tokenize(text)
    num_cut = len(text_tok) - max_token_num
    if num_cut > 0:
        print('cut ', num_cut)
        doc_trunc = tokenizer.convert_tokens_to_string( text_tok[:-(num_cut+10)] ) + ' ...'
        return doc_trunc, len(text_tok)
    else:
        return text, len(text_tok)

with open(input_path) as f_in:
    for line in f_in:
        #print(line.strip().split('\t',1))
        label_reg, text = line.strip().split('\t',1)
        label_reg = int(label_reg)
        text, org_len = preprocessing_text(text)
        if org_len < 2:
            print('skip too short example')
            continue
        text_arr.append(text)
        label_reg_arr.append(label_reg)
        if label_reg > 1:
            label = 1
        else:
            label = 0
        label_arr.append(label)
        cat_arr.append('rJoke')

print('positive ratio', sum(label_arr) / float( len(label_arr) ) )

df = pd.DataFrame({'statement': text_arr, 'label': label_arr, 'label_reg': label_reg_arr, 'category': cat_arr})

df.to_csv(output_path)
