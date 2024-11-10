import json

from transformers import AutoTokenizer

#input_path = '/mnt/efs/Haw-Shiuan/HaluEval/data/dialogue_data.json'
#input_path = '/mnt/efs/Haw-Shiuan/HaluEval/data/qa_data.json'
input_path = '/mnt/efs/Haw-Shiuan/HaluEval/data/summarization_data.json'

#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_data.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/qa_data.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/summarization_data.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/summarization_data_2048.json'
output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/summarization_data_1024.json'

#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/dialogue_data_knowledge.json'
#output_path = '/mnt/efs/Haw-Shiuan/true_entropy/outputs/Halu/qa_data_knowledge.json'

#include_knowledge = True
include_knowledge = False

cut_end = True

if cut_end:
    #max_token_num = 2048
    max_token_num = 1024
    small_model_name = 'EleutherAI/pythia-70m-deduped'
    tokenizer = AutoTokenizer.from_pretrained(small_model_name, truncation_side='left')

prepend_space = True
if prepend_space:
    space_prefix = ' '
    space_suffix = ''
else:
    space_prefix = ''
    space_suffix = ' '


output_list = []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        sample = json.loads(line)
        #pos_output_dict = {'factual': 1}#{'context': '', 'text': '', 'factual': ''}
        #neg_output_dict = {'factual': 0}
        output_dict = {}
        if "dialogue_history" in sample:
            output_dict['text_pos'] = space_prefix + sample['right_response']
            output_dict['text_neg'] = space_prefix + sample['hallucinated_response']
            
            context_raw = space_prefix + sample["dialogue_history"] + '[Assistant]:' + space_suffix
            if include_knowledge:
                context = space_prefix + sample['knowledge'] + '.' + space_suffix + context_raw
            else:
                context = context_raw
        elif "question" in sample:
            output_dict['text_pos'] = space_prefix + sample['right_answer']
            output_dict['text_neg'] = space_prefix + sample['hallucinated_answer']
            
            context_raw = space_prefix + 'Question: ' + sample["question"] + '. Answer:' + space_suffix
            if include_knowledge:
                context = space_prefix + sample['knowledge'] + space_suffix + context_raw
            else:
                context = context_raw
        elif "document" in sample:
            output_dict['text_pos'] = space_prefix + sample['right_summary']
            output_dict['text_neg'] = space_prefix + sample['hallucinated_summary']
            
            context = space_prefix + 'Document: ' + sample["document"] + ' Summary:' + space_suffix
            if cut_end:
                context_tok = tokenizer.tokenize(context)
                pos_tok = tokenizer.tokenize(output_dict['text_pos'])
                neg_tok = tokenizer.tokenize(output_dict['text_neg'])
                num_cut = len(context_tok) + max(len(pos_tok), len(neg_tok)) - max_token_num
                if num_cut > 0:
                    print('cut ', num_cut)
                    doc_tok = tokenizer.tokenize( sample["document"] )
                    doc_trunc = tokenizer.convert_tokens_to_string( doc_tok[:-(num_cut+10)] ) + '...'
                    context = space_prefix + 'Document: ' + doc_trunc + ' Summary:' + space_suffix

        output_dict['context'] = context
        output_list.append(output_dict)

with open(output_path, 'w') as f_out:
    for output_dict in output_list:
        f_out.write(json.dumps(output_dict)+'\n')
