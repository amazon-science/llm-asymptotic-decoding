from torch import nn
import torch

class ExpDecayCurve(nn.Module):
    def __init__(self, ans_len, topk_th, log_model_size, device):
        super(ExpDecayCurve, self).__init__()
        #self.coeff = nn.Parameter(0.01*torch.ones(ans_len, topk_th, 4, device=device, requires_grad=True, dtype=torch.float16))
        self.coeff = nn.Parameter(0.001*torch.ones(ans_len, topk_th, 4, device=device, requires_grad=True))
        self.coeff.data[:,:,0] = 1
        self.device = device
        num_models = len(log_model_size)
        #self.log_model_size = torch.tensor(log_model_size, device=device, dtype=torch.float16).expand(ans_len, topk_th,num_models)
        self.log_model_size = torch.tensor(log_model_size, device=device).expand(ans_len, topk_th, num_models)
        self.emphasize_last_w = 10

    def compute_coeff_pos(self):
        self.coeff.data = self.coeff.clamp(0.0)

    def forward(self, prob_all=None):
        ans_len, topk_th, num_models = self.log_model_size.size()

        ap = self.coeff[:,:,0].unsqueeze(dim=-1).expand(ans_len, topk_th, num_models)
        scale = self.coeff[:,:,1].unsqueeze(dim=-1).expand(ans_len, topk_th, num_models)
        exp_scale = self.coeff[:,:,2].unsqueeze(dim=-1).expand(ans_len, topk_th, num_models)
        exp_bias = self.coeff[:,:,3].unsqueeze(dim=-1).expand(ans_len, topk_th, num_models)

        small_num = 0
        pw = - torch.maximum(exp_scale * (self.log_model_size - exp_bias), torch.tensor(small_num, device=self.device) )
        pred = ap + scale * pw.exp()

        if prob_all is not None:
            loss_fct = nn.MSELoss()
            loss_rest = torch.pow( loss_fct(pred[:,:,1:], prob_all[:,:,1:]), 0.5)

            #loss_reg = torch.pow( loss_fct(self.coeff[:,:,0], prob_all[:,:,0]), 0.5 )
            top_err = torch.maximum(pred[:,:,0] - prob_all[:,:,0], torch.tensor(0,device=pred.device) )
            loss_last = torch.pow( (top_err).mean() , 0.5)
            #loss = loss_rest + 0.5 * loss_reg + self.emphasize_last_w * loss_last
            loss = loss_rest + self.emphasize_last_w * loss_last
        #model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_k,num_models) - g), torch.tensor(small_num, device=logits.device) )
            return loss, pred
        else:
            return pred

def estimate_para(prob_all, log_model_size):
    with torch.enable_grad():
        ans_len, topk_th, num_models = prob_all.size()
        prob_all = prob_all.to(dtype=torch.float32)
        #print(prob_all)
        #exit()
        lr = 1e-2
        max_iter = 400
        while(True):
            EDC = ExpDecayCurve(ans_len, topk_th, log_model_size, prob_all.device)
            #lr = 1
            #opt = torch.optim.Adam(EDC.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
            opt = torch.optim.Adam(EDC.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
            #opt = torch.optim.RMSprop(EDC.parameters(), lr=lr, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
            #opt = torch.optim.SGD(EDC.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)
            for i in range(max_iter):
                opt.zero_grad()
                loss, pred = EDC(prob_all)
                #print(EDC.coeff.data)
                #print(loss.item(), end =" ")
                loss.backward()
                opt.step()
                #print(EDC.coeff.data)
                EDC.compute_coeff_pos()

            prob_ap_unnorm = EDC.coeff.data[:,:,0]

            if not torch.isnan(prob_ap_unnorm).any():
                break
            lr = lr / 2
            print('lr', lr)

        return prob_ap_unnorm


def compute_ap(logits_llm, model_arr, input_ids_i, org_left_len_i, pos_left_len_i, device_st, topk_th, target_label, log_model_size):
    #device_st = 'cpu'
    logits_llm_ans = logits_llm[:, org_left_len_i:pos_left_len_i, :].to(device_st)
    sorted_logits, sorted_indices = torch.sort(logits_llm_ans, dim=-1, descending=True) #(1,ans_len, vocab)
    sorted_logits_topk = sorted_logits[:, :, :topk_th ]

    #logits_arr = [sorted_logits_topk.to(device_st)]
    logits_arr = [sorted_logits_topk]
    for model_i in model_arr:
        outputs_i = model_i( input_ids_i.to(model_i.device), return_dict = True )
        logit_topk_i = torch.gather(outputs_i.logits[:,org_left_len_i: pos_left_len_i,:].to(device_st), dim=-1, index=sorted_indices[:,:,:topk_th] )
        logits_arr.append( logit_topk_i )

    logits_all = torch.stack(logits_arr, dim=-1) #(1, ans_len, topk_th, num_models)
    prob_all = logits_all.softmax(dim=-2)[0,:,:,:]
    prob_topk_sorted_i = prob_all[:, :, 0]
    #prob_topk_sorted_i = prob_topk_sorted_i / prob_topk_sorted_i.sum(dim=-1, keepdim=True)

    #reverse prob_all
    need_to_reversed_bool = prob_all[:,:,0] > prob_all[:,:,-1]
    prob_all_rev = prob_all.clone()
    prob_all_rev[need_to_reversed_bool] = 1 - prob_all_rev[need_to_reversed_bool]

    prob_ap_rev_unnorm = estimate_para(prob_all_rev, log_model_size)

    #reverse back
    prob_ap_unnorm = prob_ap_rev_unnorm.clone()
    prob_ap_unnorm[need_to_reversed_bool] = 1 - prob_ap_rev_unnorm[need_to_reversed_bool]
    prob_ap_raw_i = prob_ap_unnorm / (1e-16+prob_ap_unnorm.sum(dim=-1, keepdim=True))

    #if torch.isnan(prob_ap_raw_i).any() or torch.isnan(prob_all).any():
    #print(prob_all, prob_ap_raw_i)

    target_label_small_idx = (sorted_indices == target_label).nonzero(as_tuple=False)[:,-1].unsqueeze(dim=-1)
    assert target_label_small_idx.size(0) == pos_left_len_i - org_left_len_i, print(target_label_small_idx, sorted_indices, target_label)

    return prob_ap_raw_i, prob_topk_sorted_i, target_label_small_idx


def merge_prob(prob_ap_raw_i, prob_topk_i, target_label_small_idx, inv_temp):
    prob_ap = (1- inv_temp) * prob_topk_i + inv_temp * prob_ap_raw_i
    #print(prob_ap)
    #print(prob_ap.size(), target_label_small_idx.size())
    gt_prob_ap = torch.gather(prob_ap, dim=-1, index=target_label_small_idx)
    gt_rank_ap = (gt_prob_ap <= prob_ap).to(torch.int32).sum(dim=-1)
    mrr_ap_raw = 1 / gt_rank_ap
    return prob_ap, mrr_ap_raw


