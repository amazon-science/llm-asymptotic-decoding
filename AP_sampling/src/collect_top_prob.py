import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import os
import argparse
from data_utils import load_corpus
from tqdm import tqdm

def top_word_prob(model, input_ids, top_w_idx_input, current_b_idx, top_k_list, sampling_range_list, vocab_size):

    assert model is not None
    input_ids = input_ids.to(model.device)
    outputs = model(input_ids, labels=input_ids)
    loss, logits = outputs[:2]

    rng = np.random.default_rng()

    probs = logits.softmax(dim=-1)
    if top_w_idx_input is None:
        sorted_probs, sorted_indices = torch.sort(probs[:,:,:vocab_size], descending=True)
        top_idx_arr = []
        bsz, seq_len, vocab_size = logits.size()
        for top_k, sampling_range in zip(top_k_list, sampling_range_list):
            lower_b, upper_b = sampling_range
            range_num = upper_b - lower_b 
            all_idx = sorted_indices[:,:,lower_b:upper_b ]
            if range_num > bsz * seq_len *  top_k:
                sampling_idx = np.sort( rng.choice( range_num, size=(bsz, seq_len, top_k), replace = False), axis=-1 )
            else:
                sampling_idx = np.broadcast_to( np.sort( rng.choice(range_num, size=top_k, replace = False) ), (bsz, seq_len, top_k) )
                #to_be_sampled = np.broadcast_to( np.arange(range_num), (bsz, seq_len, range_num) )
                #sampling_idx = rng.choice( to_be_sampled, size=top_k, axis=2, replace = False)

            top_idx_i = torch.gather(all_idx, dim=-1, index=torch.tensor(sampling_idx).to( all_idx.device ) )
            top_idx_arr.append(top_idx_i)

        top_idx = torch.cat( top_idx_arr, dim = -1 )
        #top_val, top_idx = torch.topk(probs, k=top_k, dim=-1)
        top_logits = torch.gather(logits, dim=-1, index=top_idx)
        top_val = torch.gather(probs, dim=-1, index=top_idx)

        return top_val.cpu(), top_logits.cpu(), top_idx.cpu()
    else:
        bsz = input_ids.size(0)
        top_idx = top_w_idx_input[current_b_idx:current_b_idx + bsz,:,:].to(model.device)
        top_val = torch.gather(probs, dim=-1, index=top_idx)
        top_logits = torch.gather(logits, dim=-1, index=top_idx)
        return top_val.cpu(), top_logits.cpu(), None
    
    #sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    #sorted_probs = sorted_logits.softmax(dim=-1)
    

    #probs = logits.softmax(dim=-1)
    #ent = - (probs * (1e-23+probs).log() ).sum(dim=-1)
    #top_val, top_idx= torch.topk(probs.squeeze(), k=top_k_val, dim=-1)
    #top_idx = top_idx.tolist()
    #print(top_idx)
    #top_tok = [tokenizer.convert_ids_to_tokens(top_idx[i]) for i in range(len(top_idx))]
    #return ent.cpu(), top_tok, top_val.cpu()

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
    parser.add_argument("--input_folder_name", type=str, required=True, default = '../true_entropy/data/processed/openwebtext17-18_1e6_Pythia')
    parser.add_argument("--output_folder", type=str, default = 'data/processed/openwebtext17-18_1e6_Pythia/prob_tensor_1024_ext')
    parser.add_argument("--top_w_idx_model_name", type=str, default = '')
    parser.add_argument("--tokenizer_name", type=str, default = '')
    parser.add_argument("--tensor_folder", type=str, default = 'tensors_all')
    parser.add_argument("--do_train", type=str2bool, nargs='?', default=True)
    parser.add_argument("--do_val", type=str2bool, nargs='?', default=True)
    #parser.add_argument("--top_k", type=int, default=10)
    #parser.add_argument("--sampling_methods", type=str, default = '0_10')
    parser.add_argument("--top_k", type=str, default='5,5')
    parser.add_argument("--sampling_methods", type=str, default = '10_100,100_inf')
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--cuda_idx", type=int, default=0)
    
    args = parser.parse_args()
    return args

def parse_sampling(top_k_str, sampling_methods, vocab_size):
    top_k_list = [int(k_str) for k_str in top_k_str.split(',')]
    sampling_range_list = []
    for i, method in enumerate(sampling_methods.split(',')):
        lower_b, upper_b = method.split('_')
        lower_b = int(lower_b)
        if upper_b == 'inf':
            upper_b = vocab_size
        else:
            upper_b = int(upper_b)
            #math.isinf(x)
        sampling_range_list.append( (lower_b, upper_b) )
        assert upper_b - lower_b + 1 >= top_k_list[i]

    return top_k_list, sampling_range_list

def compute_ent(args, model, model_name, dataloader, save_folder_name, device, vocab_size):
    #use_exist_top_w = True
    ouput_dir = args.output_folder + '/' + save_folder_name
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    #top_k_list, sampling_range_list = parse_sampling(args.top_k, args.sampling_methods, model.config.vocab_size)
    top_k_list, sampling_range_list = parse_sampling(args.top_k, args.sampling_methods, vocab_size)
    
    top_w_idx_input = None
    if args.top_w_idx_model_name != args.model_name:
        #top_w_idx_path = ouput_dir + '/top_w_' + args.top_w_idx_model_name.replace('/','_') + '_k_' + str(args.top_k) + '_bptt_' + str(args.bptt) + '.pt'
        top_w_idx_path = ouput_dir + '/top_w_' + args.top_w_idx_model_name.replace('/','_') + '_' + args.sampling_methods + '_k_' + str(args.top_k) + '_bptt_' + str(args.bptt) + '.pt'
        #top_w_idx_input = load_w_idx( top_w_idx_path )
        #top_w_idx_input = torch.load( top_w_idx_path ).to(device)
        top_w_idx_input = torch.load( top_w_idx_path )
        assert top_w_idx_input.size(-1) == sum(top_k_list)


    output_prob = []
    output_logits = []
    output_top_w = []
    with torch.no_grad():
        #for i_batch, sample_batched in enumerate(dataloader_train):
        current_b_idx = 0
        for sample_batched in tqdm(dataloader):
            #prob_tensor, logits_tensor, top_w_idx_i = top_word_prob( model, sample_batched, top_w_idx_input, current_b_idx * args.batch_size, args.top_k )
            prob_tensor, logits_tensor, top_w_idx_i = top_word_prob( model, sample_batched, top_w_idx_input, current_b_idx * args.batch_size, top_k_list, sampling_range_list, vocab_size )
            output_prob.append(prob_tensor)
            output_logits.append(logits_tensor)
            output_top_w.append(top_w_idx_i)
            current_b_idx += 1

    output_tensor = torch.cat(output_prob, dim=0)
    print(model_name)
    print(args.cuda_idx)
    #print(output_tensor)
    print(output_tensor.size())
    del output_prob
    output_file_name = ouput_dir + '/prob_' + model_name.replace('/','_') + '_w_idx_' + args.top_w_idx_model_name.replace('/','_') + '_' + args.sampling_methods + '_k_' + str(args.top_k) + '_bptt_' + str(args.bptt) + '.pt'
    torch.save(output_tensor, output_file_name)
    output_logits_tensor = torch.cat(output_logits,  dim=0)
    del output_logits
    output_file_name = ouput_dir + '/logits_' + model_name.replace('/','_') + '_' + args.sampling_methods + '_k_' + str(args.top_k) + '_bptt_' + str(args.bptt) + '.pt'
    torch.save(output_logits_tensor, output_file_name)
    if top_w_idx_input is None:
        output_top_w_tensor = torch.cat(output_top_w, dim=0)
        del output_top_w
        #print(output_top_w_tensor.size())
        output_file_name = ouput_dir + '/top_w_' + model_name.replace('/','_') + '_' + args.sampling_methods + '_k_' + str(args.top_k) + '_bptt_' + str(args.bptt) + '.pt'
        torch.save(output_top_w_tensor, output_file_name)


def main(args):
    model_name = args.model_name
    if len(args.tokenizer_name) > 0:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
    vocab_size = len(tokenizer)
    #model = AutoModelWithLMHead.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
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
        compute_ent(args, model, model_name, dataloader_train, 'train', device, vocab_size)
    if args.do_val:
        compute_ent(args, model, model_name, dataloader_val, 'val', device, vocab_size)

if __name__ == "__main__":
    args = parse_args()
    main(args)
