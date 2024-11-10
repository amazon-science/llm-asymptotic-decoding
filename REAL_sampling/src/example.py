import sys
sys.path.append('./src/factual_gen/')
from sampling_method import FETopPLogitsWarper, LogitsProcessorList
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

sampling = 'REAL'
#sampling = 'REAL + CD'

LLM = 'Pythia'
#LLM = 'OPT'

final_entropy_model_path = 'models/OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3'
decay_temperature = 2
window_size = 40
device = torch.device("cuda:0")

if LLM == 'Pythia':
    LM_gen = 'EleutherAI/pythia-6.9b-deduped'
    tokenizer = AutoTokenizer.from_pretrained(LM_gen, padding_side='left', model_max_length=1024)
    tokenizer_ent = tokenizer
else:
    LM_gen = 'facebook/opt-6.7b'
    tokenizer = AutoTokenizer.from_pretrained(LM_gen, padding_side='left', model_max_length=1024)
    tokenizer_ent = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped', padding_side='left', model_max_length=1024)

tokenizer.pad_token = tokenizer.eos_token
tokenizer_ent.pad_token = tokenizer_ent.eos_token

model = AutoModelForCausalLM.from_pretrained(LM_gen)
model.eval()
model.to(device)

if sampling == 'REAL':
    logits_processor_i = FETopPLogitsWarper(top_p = 1, decay_temperature = decay_temperature, final_entropy_model_path = final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = 'exp_1_win', window_size = window_size, device=device)
else:
    if LLM == 'Pythia':
        student_model_name = 'EleutherAI/pythia-70m-deduped'
    else:
        student_model_name = 'facebook/opt-125m'
    logits_processor_i = FETopPLogitsWarper(top_p = 1, decay_temperature = decay_temperature, final_entropy_model_path = final_entropy_model_path, tokenizer=tokenizer, tokenizer_ent=tokenizer_ent, sample_sub_method = 'exp_1_win', window_size = window_size, student_model_name=student_model_name, use_CD_alpha= False, device=device)

logits_processor = LogitsProcessorList()
logits_processor.append(logits_processor_i)

input_prompt = " I like to go hiking."
input_ids = tokenizer(input_prompt, return_tensors="pt").input_ids

output_sequences = model.generate(input_ids=input_ids.to(device), pad_token_id=tokenizer.eos_token_id, logits_processor=logits_processor, do_sample=True )
input_len = input_ids.size(-1)
output_con = output_sequences[0,input_len:]
output_text = tokenizer.decode(output_con, skip_special_tokens=True)
print("Input: ", input_prompt)
print("Output: ", output_text)
