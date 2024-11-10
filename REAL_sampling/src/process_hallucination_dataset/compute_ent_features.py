import torch

def gather_target_word(input_tensor, target_word_idx, target_word_end_idx):
    if target_word_end_idx is None:
        target_word_idx = target_word_idx.view(-1,1)
        return torch.gather(input_tensor, dim=-1, index=target_word_idx).view(-1,1).cpu()
    else:
        input_all_sum = torch.cumsum(input_tensor, dim=1)
        target_word_end_idx = target_word_end_idx.view(-1,1)
        if not torch.is_tensor(target_word_idx) and target_word_idx == 0:
            input_span_sum = torch.gather(input_all_sum, dim=-1, index=target_word_end_idx).view(-1,1)
            return ( input_span_sum / (target_word_end_idx+1)  ).cpu()
        else:
            target_word_idx = target_word_idx.view(-1,1)
            input_span_sum = ( torch.gather(input_all_sum, dim=-1, index=target_word_end_idx) -  torch.gather(input_all_sum, dim=-1, index=target_word_idx) ).view(-1,1)
            return ( input_span_sum / (target_word_end_idx - target_word_idx)  ).cpu()

def word_ent_per(model, input_ids, org_left_len, org_left_text_len, mode, use_deepspeed=False):
    assert model is not None
    input_ids = input_ids.to(model.device)
    if torch.is_tensor(org_left_len):
        org_left_len = org_left_len.to(model.device)
    if torch.is_tensor(org_left_text_len):
        org_left_text_len = org_left_text_len.to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
    logits = outputs[1]
    #print(org_left_len, org_left_text_len)

    if mode == 'ent':
        probs = logits.softmax(dim=-1)
        if use_deepspeed:
            ent = - (probs * (1e-2+probs).log() ).sum(dim=-1)
        else:
            ent = - (probs * (1e-23+probs).log() ).sum(dim=-1)
        #ent_last = torch.gather(ent, dim=-1, index=org_left_len)
        ent_last = gather_target_word(ent, org_left_len, org_left_text_len)
        #print(ent)
        return ent_last
    elif mode == 'per':
        labels = input_ids
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        if use_deepspeed:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none', label_smoothing=1e-2)
        else:
            loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        bsz, seq_len_minus_one = shift_labels.size()
        lm_per = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(bsz, seq_len_minus_one)    
        #lm_per_last = torch.gather(lm_per, dim=-1, index=org_left_len) #the perplexity of the last word in the label, org_left_len = org_left_len+1-1
        lm_per_last = gather_target_word(lm_per, org_left_len, org_left_text_len)
        #lm_per = torch.cat( (lm_per, torch.zeros( (bsz,1), device = model.device ) ), dim=1  )
        return lm_per_last


def compute_model_feature(model_ent, org_left_word_tensor, org_left_len, org_left_text_len):
    org_left_word_tensor_ent = org_left_word_tensor.to(model_ent.device)
    if torch.is_tensor(org_left_len) != 0:
        org_left_len = org_left_len.to(model_ent.device)
    if torch.is_tensor(org_left_text_len):
        org_left_text_len = org_left_text_len.to(model_ent.device)
    with torch.no_grad():
        output = model_ent(org_left_word_tensor_ent, return_dict=False)
    ent_pred = output[1]
    logit_pos = output[2]
    c = gather_target_word(logit_pos[:,:,0,0], org_left_len, org_left_text_len)
    pred_last_ent = gather_target_word(ent_pred[:,:,-1], org_left_len, org_left_text_len)
    curve_last_ent = gather_target_word(ent_pred[:,:,-2], org_left_len, org_left_text_len)
    uncertainty_score1 = (curve_last_ent - c)
    uncertainty_score2 = (pred_last_ent - c)
    return c, pred_last_ent, curve_last_ent, uncertainty_score1, uncertainty_score2

def collect_features_Halu(org_left_len, org_left_text_len, org_left_text_tensor, model_small_lm, model_large_lm, model_ent, model_per, use_deepspeed):
    #print('small: ')
    entropy_tensor_small = word_ent_per( model_small_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'ent', use_deepspeed)
    perplexity_tensor_small = word_ent_per( model_small_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'per', use_deepspeed)
    #print('large: ')
    entropy_tensor_large = word_ent_per( model_large_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'ent', use_deepspeed)
    perplexity_tensor_large = word_ent_per( model_large_lm, org_left_text_tensor, org_left_len, org_left_text_len, 'per', use_deepspeed)

    c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2 = compute_model_feature(model_ent, org_left_text_tensor, org_left_len, org_left_text_len)
    c_per, pred_last_per, curve_last_per, per_score1, per_score2 = compute_model_feature(model_per, org_left_text_tensor, org_left_len, org_left_text_len)

    ent_score3 = torch.pow(entropy_tensor_large * torch.maximum(torch.tensor(0), entropy_tensor_small - entropy_tensor_large), 0.5 )
    per_score3 = torch.pow(perplexity_tensor_large * torch.maximum(torch.tensor(0), perplexity_tensor_small - perplexity_tensor_large), 0.5 )
    #1-3 real entropy, 4-6 real perplexity, 7-11 predicted entropy, 12-16 predicted perplexity
    all_features_i = [entropy_tensor_small, entropy_tensor_large, ent_score3, perplexity_tensor_small, perplexity_tensor_large, per_score3, c_ent, pred_last_ent, curve_last_ent, ent_score1, ent_score2, c_per, pred_last_per, curve_last_per, per_score1, per_score2]
    #print(all_features_i)
    all_features_i = torch.stack(all_features_i, dim=1).squeeze(dim=2).tolist()
    #all_features_i = torch.stack(all_features_i, dim=1).tolist()
    #print(all_features_i)
    return all_features_i
