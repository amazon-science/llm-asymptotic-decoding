import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import argparse
from data_utils import load_corpus
from tqdm import tqdm
    
loss_fct = torch.nn.CrossEntropyLoss(reduction='none')

def word_ent(model, input_ids):
    #top_k_val = 5
    assert model is not None
    input_ids = input_ids.to(model.device)
    outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]
    #probs = logits.softmax(dim=-1)
    #ent = - (probs * (1e-23+probs).log() ).sum(dim=-1)
    #return ent.cpu()
    
    labels = input_ids
    # we are doing next-token prediction; shift prediction scores and input ids by one
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    bsz, seq_len_minus_one = shift_labels.size()
    lm_per = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bsz, seq_len_minus_one)    
    lm_per = torch.cat( (lm_per, torch.zeros( (bsz,1), device = model.device ) ), dim=1  )
    return lm_per.cpu()

def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--input_folder_name", type=str, required=True, default = 'data/processed/openwebtext17-18_1e6_Pythia')
    #parser.add_argument("--output_folder_name", type=str, required=True, default = 'data/processed/openwebtext17-18_1e6_Pythia')
    parser.add_argument("--output_tensor_folder", type=str, default = 'perplexity_tensor')
    parser.add_argument("--tensor_folder", type=str, default = 'tensors_all')
    parser.add_argument("--do_train", type=str2bool, nargs='?', default=True)
    parser.add_argument("--do_val", type=str2bool, nargs='?', default=True)
    parser.add_argument("--batch_size", type=int, default=8)
    #parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--bptt", type=int, default=256)
    parser.add_argument("--cuda_idx", type=int, default=0)
    
    args = parser.parse_args()
    return args

#model_name = 'EleutherAI/pythia-70m-deduped'

def compute_ent(args, model, model_name, dataloader, save_folder_name):
    output_entropy = []
    with torch.no_grad():
        #for i_batch, sample_batched in enumerate(dataloader_train):
        for sample_batched in tqdm(dataloader):
            entropy_tensor = word_ent( model, sample_batched )
            output_entropy.append(entropy_tensor)

    output_tensor = torch.cat(output_entropy, dim=0)
    print(model_name)
    print(args.cuda_idx)
    print(output_tensor)
    print(output_tensor.size())
    del output_entropy
    ouput_dir = args.input_folder_name + '/' + args.output_tensor_folder + '/' + save_folder_name
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    output_file_name = ouput_dir + '/per_' + model_name.replace('/','_') + '_bptt_' + str(args.bptt) + '.pt'
    torch.save(output_tensor, output_file_name)



def main(args):
    model_name = args.model_name
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #model = AutoModelWithLMHead.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    #device = torch.cuda.device(args.cuda_idx)
    device = torch.device("cuda:"+str(args.cuda_idx))
    model.eval()
    model.to(device)
    
    print(args.do_train)
    print(args.do_val)
    skip_training = False
    #dataloader_train, dataloader_val, dataloader_test = load_corpus(args.input_folder_name, args.batch_size, args.batch_size, args.bptt, device, args.tensor_folder, shuffle_train=False, skip_training = False, load_val = False, load_testing=False)
    dataloader_train, dataloader_val, dataloader_test = load_corpus(args.input_folder_name, args.batch_size, args.batch_size, args.bptt, device, args.tensor_folder, shuffle_train=False, skip_training = False, load_val = True, load_testing=False)
    
    if args.do_train:
        compute_ent(args, model, model_name, dataloader_train, 'train')
    if args.do_val:
        compute_ent(args, model, model_name, dataloader_val, 'val')

if __name__ == "__main__":
    args = parse_args()
    main(args)
