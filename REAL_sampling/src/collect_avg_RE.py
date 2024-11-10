import torch
import os
import argparse
from data_utils import load_corpus
from tqdm import tqdm
from model import GPTNeoXForEntropyClassification
import json

def compute_model_last(model_ent, org_left_word_tensor):
    org_left_word_tensor_ent = org_left_word_tensor.to(model_ent.device)
    with torch.no_grad():
        output = model_ent(org_left_word_tensor_ent, return_dict=False)
    ent_pred = output[1]
    logit_pos = output[2]
    c = logit_pos[:,:,0,0]
    pred_last_ent = ent_pred[:,:,-1]
    curve_last_ent = ent_pred[:,:,-2]
    uncertainty_score1 = (curve_last_ent - c)
    uncertainty_score2 = (pred_last_ent - c)
    return c, pred_last_ent, curve_last_ent, uncertainty_score1, uncertainty_score2


def str2bool(v):
    if v.lower() in ('yes', 'true', 'True', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'False', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default = 'models/OWT_wiki_1e7_70M_bsz_128_exp_pred_last_a10_e3')
    #parser.add_argument("--model_name", type=str, default = 'models/OWT_wiki_1e7_70M_bsz_32_exp_pred_last_a6_e3')
    parser.add_argument("--input_folder_name", type=str, default = 'data/processed/OWT_wiki_1e7_Pythia')
    #parser.add_argument("--input_folder_name", type=str, required=True, default = 'data/processed/wiki2021_1e6_Pythia')
    #parser.add_argument("--output_folder_name", type=str, required=True, default = 'data/processed/openwebtext17-18_1e6_Pythia')
    #parser.add_argument("--output_tensor_folder", type=str, default = 'entropy_tensor')
    parser.add_argument("--tensor_folder", type=str, default = 'tensors_all')
    parser.add_argument("--batch_size", type=int, default=32)
    #parser.add_argument("--eval_batch_size", type=int, default=16)
    #parser.add_argument("--bptt", type=int, default=256)
    parser.add_argument("--bptt", type=int, default=1024)
    parser.add_argument("--cuda_idx", type=int, default=0)
    
    args = parser.parse_args()
    return args

#model_name = 'EleutherAI/pythia-70m-deduped'

def compute_ent(args, model, model_name, dataloader, save_folder_name):
    all_c_sum = 0
    all_last_ent_sum = 0
    num_sample = 0
    with torch.no_grad():
        #for i_batch, sample_batched in enumerate(dataloader_train):
        for sample_batched in tqdm(dataloader):
            #print(sample_batched.size())
            num_sample += sample_batched.size(0)
            c, pred_last_ent, curve_last_ent, uncertainty_score1, uncertainty_score2 = compute_model_last(model, sample_batched)
            #print(c.size())
            all_c_sum += c.sum(dim=0)
            #print(all_c_sum.size())
            all_last_ent_sum += curve_last_ent.sum(dim=0)

    all_c_avg = all_c_sum / num_sample
    all_last_ent_avg = all_last_ent_sum / num_sample
    print(model_name)
    #print(args.cuda_idx)
    print(all_c_avg.size())
    print(all_c_avg)
    print(all_last_ent_avg)
    print(all_last_ent_avg - all_c_avg)
    ouput_dir = args.model_name + '/avg_c_pred_ent' 
    if not os.path.exists(ouput_dir):
        os.makedirs(ouput_dir)
    output_file_name = ouput_dir + '/ent_' + os.path.basename(args.input_folder_name) + '_' + save_folder_name  +  '_bptt_' + str(args.bptt) + '.json'
    with open(output_file_name, 'w') as f_out:
        json.dump([all_c_avg.tolist(), all_last_ent_avg.tolist(), (all_last_ent_avg - all_c_avg).tolist()], f_out)



def main(args):
    model_name = args.model_name
    log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]

    use_a4 = False
    use_a56 = False
    use_a10 = False
    if '_a4_' in model_name:
        use_a4 = True
    if '_a6_' in model_name:
        use_a4 = True
        use_a56 = True
    if '_a10_' in model_name:
        use_a4 = True
        use_a56 = True
        use_a10 = True

    model = GPTNeoXForEntropyClassification.from_pretrained(model_name, log_model_size=log_model_size, use_a4=use_a4, use_a56=use_a56, use_a10=use_a10)
    device = torch.device("cuda:"+str(args.cuda_idx))
    model.eval()
    model.to(device)
    
    skip_training = False
    #dataloader_train, dataloader_val, dataloader_test = load_corpus(args.input_folder_name, args.batch_size, args.batch_size, args.bptt, device, args.tensor_folder, shuffle_train=False, skip_training = False, load_val = False, load_testing=False)
    dataloader_train, dataloader_val, dataloader_test = load_corpus(args.input_folder_name, args.batch_size, args.batch_size, args.bptt, device, args.tensor_folder, shuffle_train=False, skip_training = True, load_val = True, load_testing=False)
    
    compute_ent(args, model, model_name, dataloader_val, 'val')

if __name__ == "__main__":
    args = parse_args()
    main(args)
