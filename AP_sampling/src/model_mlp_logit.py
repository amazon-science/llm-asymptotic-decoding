import torch.nn as nn
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel
from transformers.models.opt.modeling_opt import OPTPreTrainedModel, OPTModel

can_load_Qwen = True
try:
    from transformers.models.qwen2.modeling_qwen2 import  Qwen2PreTrainedModel, Qwen2Model
except ImportError:
    can_load_Qwen = False
    print('cannot import Qwen')

#from typing import Optional
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import TokenClassifierOutput, CausalLMOutputWithPast

class MLP_para(nn.Module):
    def __init__(self, log_model_size, decay_function='exp', poly_degree=10, emphasize_last_w=10, rev_curves=True):
        super().__init__()
        self.positive_mode = 'exp'
        self.log_model_size = torch.tensor(log_model_size)
        #self.use_a4 = use_a4
        #self.use_a56 = use_a56
        #self.use_a10 = use_a10
        #self.emphasize_last_w = -1
        self.emphasize_last_w = emphasize_last_w
        self.penalize_top_err = True
        #self.num_labels = 15
        if decay_function == 'poly':
            self.num_labels = 5 + poly_degree
        elif decay_function == 'scaled_poly':
            self.num_labels = 2 + (1 + poly_degree) * 3
        elif decay_function=='exp' or decay_function=='logistic':
            self.num_labels = 4
        self.decay_function = decay_function
        self.poly_degree = poly_degree
        #if self.use_a4:
        #    self.num_labels += 1
        #if self.use_a56:
        #    assert self.use_a4 
        #    self.num_labels += 2
        #if self.use_a10:
        #    assert self.use_a4 
        #    assert self.use_a56 
        #    self.num_labels += 4
        self.input_dim = len(log_model_size) + 1
        input_dropout_rate = 0.5
        hidden_state_dim = 100
        self.rev_curves = rev_curves

        self.dropout = nn.Dropout(input_dropout_rate)
        self.layer1 = nn.Linear(self.input_dim, hidden_state_dim)
        self.act1 = nn.GELU()
        self.layer2 = nn.Linear(hidden_state_dim, hidden_state_dim)
        self.act2 = nn.GELU()
        self.layer3 = nn.Linear(hidden_state_dim, hidden_state_dim)
        self.act3 = nn.GELU()
        self.output = nn.Linear(hidden_state_dim, self.num_labels)

    
    def compute_prob_prediction(self, logits, log_model_size, c):
        num_models = log_model_size.size(0)
        c = c.unsqueeze(-1).expand(-1,-1,-1,num_models) #-> (bsz, seq_len, num_k, num_models)
        logits = logits.unsqueeze(-1).expand(-1,-1,-1,-1,num_models) #-> (bsz, seq_len, num_k, num_labels, num_models)
        bsz, seq_len, num_k, num_para, num_models = logits.size()
        
        if self.positive_mode == 'exp':
            logits_pos = torch.exp(logits)
        elif self.positive_mode == 'ReLU':
            very_small_num = 1e-20
            logits_pos = torch.maximum(logits , torch.tensor(very_small_num,device=logits.device) )
        #c = logits_pos[:,:,:,0,:]
        b = logits_pos[:,:,:,1,:]
        f, g = logits_pos[:,:,:,2,:], logits_pos[:,:,:,3,:]
        #print(bsz, seq_len, num_k, num_para, num_models)
        #print(log_model_size.size())
        #print(g.size())
        if self.decay_function == 'poly':
            small_num = 1
            model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_k,num_models) - g), torch.tensor(small_num, device=logits.device) )
            a05 = logits_pos[:,:,:,4,:]
            prob_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5))
            for i in range(self.poly_degree):
                prob_pred = prob_pred + b*(logits_pos[:,:,:,5+i,:] / torch.pow(model_log_size_norm,i+1))
        elif self.decay_function == 'scaled_poly':
            small_num = 1
            model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_k,num_models) - g), torch.tensor(small_num, device=logits.device) )
            a05 = logits_pos[:,:,:,4,:]
            prob_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5))
            for i in range(self.poly_degree):
                f, g = logits_pos[:,:,:,5+3*i,:], logits_pos[:,:,:,6+3*i,:]
                model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_k,num_models) - g), torch.tensor(small_num, device=logits.device) )
                prob_pred = prob_pred + b*(logits_pos[:,:,:,7+3*i,:] / torch.pow(model_log_size_norm,i+1))
        elif self.decay_function == 'exp':
            small_num = 0
            model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_k,num_models) - g), torch.tensor(small_num, device=logits.device) )
            if self.rev_curves:
                prob_pred = c + b*( torch.exp(- model_log_size_norm ) )
            else:
                b = logits[:,:,:,1,:]
                prob_pred = c + b*( torch.exp(- model_log_size_norm ) )
        elif self.decay_function == 'logistic':
            small_num = 0
            model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_k,num_models) - g), torch.tensor(small_num, device=logits.device) )
            prob_pred = c + b / ( 1 + torch.exp(model_log_size_norm ) )
        
        #a05, a1, a2, a3 = logits_pos[:,:,:,4,:], logits_pos[:,:,:,5,:], logits_pos[:,:,:,6,:], logits_pos[:,:,:,7,:]
        #model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_k,num_models) - g), torch.tensor(small_num, device=logits.device) )
        #if self.use_a4:
        #    a4 = logits_pos[:,:,:,8,:]
        #    if self.use_a56:
        #        a5 = logits_pos[:,:,:,9,:]
        #        a6 = logits_pos[:,:,:,10,:]
        #        if self.use_a10:
        #            a7 = logits_pos[:,:,:,11,:]
        #            a8 = logits_pos[:,:,:,12,:]
        #            a9 = logits_pos[:,:,:,13,:]
        #            a10 = logits_pos[:,:,:,14,:]
        #            prob_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) + a4 / torch.pow(model_log_size_norm,4) + a5 / torch.pow(model_log_size_norm,5) + a6 / torch.pow(model_log_size_norm,6) + a7 / torch.pow(model_log_size_norm,7) + a8 / torch.pow(model_log_size_norm,8) + a9 / torch.pow(model_log_size_norm,9) + a10 / torch.pow(model_log_size_norm,10) )
        #        else:
        #            prob_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) + a4 / torch.pow(model_log_size_norm,4) + a5 / torch.pow(model_log_size_norm,5) + a6 / torch.pow(model_log_size_norm,6) )
        #    else:
        #        prob_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) + a4 / torch.pow(model_log_size_norm,4) )
        #else:
        #    prob_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) )

        return prob_pred, logits_pos

    def forward(self, input_prob, inf_LLM_prob):
        x = input_prob.clone()
        #if self.emphasize_last_w == -1:
        #    x = self.dropout(x) #(bsz, seq_len, num_k, num_models)
        #else: #make sure the largest LLM's probabilities are not masked
        if x.size(3) > 3:
            x[:,:,:,2:-1] = self.dropout(x[:,:,:,2:-1]) #(bsz, seq_len, num_k, num_models)
        x = torch.cat( (x, inf_LLM_prob.unsqueeze(-1)), dim=-1 )

        #print(x[0,0,0,:])
        #print('x', x[torch.isnan(x)])
        x = self.act1(self.layer1(x))
        #print(x.view(-1)[:10])
        x = self.act2(self.layer2(x))
        x = self.act3(self.layer3(x))
        logits = self.output(x) #(bsz, seq_len, num_k, num_labels)
        #print(logits.view(-1)[:10])
        #print(logits[0,0,0,:])
        #print('logits', logits[torch.isnan(logits)])

        prob_pred, logits_pos = self.compute_prob_prediction(logits, self.log_model_size, inf_LLM_prob)
        #print('prob_pred', prob_pred[torch.isnan(prob_pred)])
        #print(prob_pred[0,0,0,:])

        labels = input_prob
        #print(labels.view(-1)[:10])
        #print(prob_pred.view(-1)[:10])
        #labels = labels.to(logits.device)
        loss_fct = nn.MSELoss()
        if self.emphasize_last_w == -1:
            loss = torch.pow( loss_fct(prob_pred, labels), 0.5)
        else:
            if self.penalize_top_err:
                loss_rest = torch.pow( loss_fct(prob_pred[:,:,:,:-1], labels[:,:,:,:-1]), 0.5)
                top_err = torch.maximum(prob_pred[:,:,:,-1] - labels[:,:,:,-1], torch.tensor(0,device=labels.device) )
                #loss_last = torch.pow( (top_err*top_err).mean() , 0.5)
                loss_last = torch.pow( (top_err).mean() , 0.5)
            else:
                loss_rest = torch.pow( loss_fct(prob_pred[:,:,:,:-1], labels[:,:,:,:-1]), 0.5)
                loss_last = torch.pow( loss_fct(prob_pred[:,:,:,-1], labels[:,:,:,-1]), 0.5)
            loss = loss_rest + self.emphasize_last_w * loss_last



        return loss, logits_pos

def reverse_increasing_prob(all_prob_curves_norm, lm_top_prob, shift_const=1):
    need_to_reversed_bool = all_prob_curves_norm[:,:,:,0] < all_prob_curves_norm[:,:,:,-1]
    lm_top_prob_rev = lm_top_prob.clone() #(bsz, seq_len, num_k)
    all_prob_curves_norm_rev = all_prob_curves_norm.clone() #(bsz, seq_len, num_k, num_models)
    lm_top_prob_rev[need_to_reversed_bool] = shift_const - lm_top_prob[need_to_reversed_bool]
    all_prob_curves_norm_rev[need_to_reversed_bool] = shift_const - all_prob_curves_norm[need_to_reversed_bool]

    return all_prob_curves_norm_rev, lm_top_prob_rev


def uncompress_label_tensor(labels_tensor):
    prob_decay_tesnor = labels_tensor[:,:,:,3:]
    small_logits_tesnor = labels_tensor[:,:,:,2]
    LLM_logit_tensor = labels_tensor[:,:,:,1]
    index_tensor = labels_tensor[:,:,:,0].type(torch.LongTensor)
    return prob_decay_tesnor, index_tensor, LLM_logit_tensor, small_logits_tesnor



class GPTNeoXForLogitCorrection(GPTNeoXPreTrainedModel):
    def __init__(self, config, log_model_size, decay_function='exp', poly_degree=10, model_logit_decay = False, logit_reg_w=0.8, emphasize_last_w=10, rev_curves=True):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlp = MLP_para(log_model_size, decay_function, poly_degree, emphasize_last_w)
        #self.logit_reg_w = 0.4
        #self.logit_reg_w = 0.8
        self.logit_reg_w = logit_reg_w
        print('logit_reg_w', self.logit_reg_w)
        #self.org_prob_reg_w = 5
        #self.org_prob_reg_w = 7.5
        #self.org_prob_reg_w = 10
        self.org_prob_reg_w = 0
        #self.org_prob_reg_w = 20
        self.reg_inv_temp = 0.5
        # Initialize weights and apply final processing
        self.post_init()
        self.model_logit_decay = model_logit_decay
        assert rev_curves
    

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        #LLM_top_logit: Optional[torch.FloatTensor] = None,
        #LLM_top_w: Optional[torch.LongTensor] = None,
        #all_prob_curves : Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states) #could save the computation a little if we only compute dot product with the top word index

        loss = None
        lm_top_prob = None
        if labels is not None:
            #labels = labels.to(lm_logits.device)
            all_prob_curves, LLM_top_w, LLM_top_logit, small_logit = uncompress_label_tensor(labels)
            all_prob_curves = all_prob_curves.to(lm_logits.device)
            LLM_top_logit = LLM_top_logit.to(lm_logits.device)
            LLM_top_w = LLM_top_w.to(lm_logits.device)
            lm_logits_top = torch.gather(lm_logits, dim=-1, index=LLM_top_w) #(bsz, seq_len, vocab_size) -> (bsz, seq_len, top_k)
            if self.model_logit_decay:
                all_logit_curves = torch.log(1e-10 + all_prob_curves)
                all_logit_curves_norm = all_logit_curves - all_logit_curves.mean(dim=-2, keepdim=True)
                top_logit_out = LLM_top_logit - lm_logits_top
                top_logit_out_norm = top_logit_out - top_logit_out.mean(dim=-2, keepdim=True)
                all_logit_curves_norm_rev, top_logit_out_norm_rev = reverse_increasing_prob(all_logit_curves_norm, top_logit_out_norm, 0)
                loss, decay_para = self.mlp(all_logit_curves_norm_rev, top_logit_out_norm_rev)
                
                #lm_top_prob = torch.softmax(top_logit_out, dim=-1) #just for output
            else:
                lm_top_prob = torch.softmax(LLM_top_logit - lm_logits_top, dim=-1).clone()
                #shifted_logits = LLM_top_logit - lm_logits_top 
                #lm_top_exp = torch.exp( shifted_logits - shifted_logits.mean() )
                #lm_top_prob = lm_top_exp / (1e-23 + lm_top_exp.sum(dim=-1, keepdim=True) )  #(bsz, seq_len, num_k)
                #lm_top_prob = lm_top_exp / ( lm_top_exp.sum(dim=-1, keepdim=True) )  #(bsz, seq_len, num_k)

                all_prob_curves_norm = all_prob_curves / (1e-23 + all_prob_curves.sum(dim=-2, keepdim=True)) #(bsz, seq_len, num_k, num_models)

                all_prob_curves_norm_rev, lm_top_prob_rev = reverse_increasing_prob(all_prob_curves_norm, lm_top_prob)
                loss, decay_para = self.mlp(all_prob_curves_norm_rev, lm_top_prob_rev)
            if self.logit_reg_w > 0:
                loss_fct = nn.MSELoss()
                loss_small = torch.pow( loss_fct(lm_logits_top, small_logit), 0.5)
                loss = loss + self.logit_reg_w * loss_small

            elif self.org_prob_reg_w > 0:
                loss_fct = nn.MSELoss()
                lm_small_top_prob = torch.softmax(lm_logits_top, dim=-1)
                #print(lm_small_top_prob.size())
                #print(all_prob_curves_norm[:,:,:,0].size())
                if self.reg_inv_temp != 1:
                    reg_prob = torch.softmax(small_logit * self.reg_inv_temp, dim=-1)
                else:
                    reg_prob = all_prob_curves_norm[:,:,:,0]
                loss_small = torch.pow( loss_fct(lm_small_top_prob, reg_prob), 0.5)
                loss = loss + self.org_prob_reg_w * loss_small
            #print(loss, loss_small)

        
        if not return_dict:
            output = (lm_logits, lm_top_prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #return TokenClassifierOutput(
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values

        )


class OPTForLogitCorrection(OPTPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config, log_model_size, decay_function='exp', poly_degree=10, model_logit_decay = False, logit_reg_w=0.8, emphasize_last_w=10, rev_curves=True):
        super().__init__(config)
        self.model = OPTModel(config)

        # the lm_head weight is automatically tied to the embed tokens weight
        self.lm_head = nn.Linear(config.word_embed_proj_dim, config.vocab_size, bias=False)

        self.mlp = MLP_para(log_model_size, decay_function, poly_degree, emphasize_last_w)
        #self.logit_reg_w = 1.0
        self.logit_reg_w = logit_reg_w
        print('logit_reg_w', self.logit_reg_w)
        #self.logit_reg_w = 0.8
        #self.org_prob_reg_w = 5
        #self.org_prob_reg_w = 7.5
        self.org_prob_reg_w = 0
        self.reg_inv_temp = 0.5
        #self.org_prob_reg_w = 20
        # Initialize weights and apply final processing
        self.post_init()
        assert rev_curves
    
    def get_input_embeddings(self):
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.decoder = decoder

    def get_decoder(self):
        return self.model.decoder

    #def reverse_increasing_prob(self, all_prob_curves_norm, lm_top_prob):
    #    need_to_reversed_bool = all_prob_curves_norm[:,:,:,0] < all_prob_curves_norm[:,:,:,-1]
    #    lm_top_prob_rev = lm_top_prob.clone() #(bsz, seq_len, num_k)
    #    all_prob_curves_norm_rev = all_prob_curves_norm.clone() #(bsz, seq_len, num_k, num_models)
    #    lm_top_prob_rev[need_to_reversed_bool] = 1 - lm_top_prob[need_to_reversed_bool]
    #    all_prob_curves_norm_rev[need_to_reversed_bool] = 1 - all_prob_curves_norm[need_to_reversed_bool]

    #    return all_prob_curves_norm_rev, lm_top_prob_rev

    #def uncompress_label_tensor(self, labels_tensor):
    #    prob_decay_tesnor = labels_tensor[:,:,:,2:]
    #    LLM_logit_tensor = labels_tensor[:,:,:,1]
    #    index_tensor = labels_tensor[:,:,:,0].type(torch.LongTensor)
    #    return prob_decay_tesnor, index_tensor, LLM_logit_tensor

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        #LLM_top_logit: Optional[torch.FloatTensor] = None,
        #LLM_top_w: Optional[torch.LongTensor] = None,
        #all_prob_curves : Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.lm_head(hidden_states) #could save the computation a little if we only compute dot product with the top word index

        loss = None
        lm_top_prob = None
        if labels is not None:
            #labels = labels.to(lm_logits.device)
            all_prob_curves, LLM_top_w, LLM_top_logit, small_logit = uncompress_label_tensor(labels)
            all_prob_curves = all_prob_curves.to(lm_logits.device)
            LLM_top_logit = LLM_top_logit.to(lm_logits.device)
            LLM_top_w = LLM_top_w.to(lm_logits.device)
            lm_logits_top = torch.gather(lm_logits, dim=-1, index=LLM_top_w) #(bsz, seq_len, vocab_size) -> (bsz, seq_len, top_k)
            lm_top_prob = torch.softmax(LLM_top_logit - lm_logits_top, dim=-1).clone()
            #shifted_logits = LLM_top_logit - lm_logits_top 
            #lm_top_exp = torch.exp( shifted_logits - shifted_logits.mean() )
            #lm_top_prob = lm_top_exp / (1e-23 + lm_top_exp.sum(dim=-1, keepdim=True) )  #(bsz, seq_len, num_k)
            #lm_top_prob = lm_top_exp / ( lm_top_exp.sum(dim=-1, keepdim=True) )  #(bsz, seq_len, num_k)

            all_prob_curves_norm = all_prob_curves / (1e-23 + all_prob_curves.sum(dim=-2, keepdim=True)) #(bsz, seq_len, num_k, num_models)

            all_prob_curves_norm_rev, lm_top_prob_rev = reverse_increasing_prob(all_prob_curves_norm, lm_top_prob)
            loss, decay_para = self.mlp(all_prob_curves_norm_rev, lm_top_prob_rev)
            if self.logit_reg_w > 0:
                loss_fct = nn.MSELoss()
                loss_small = torch.pow( loss_fct(lm_logits_top, small_logit), 0.5)
                loss = loss + self.logit_reg_w * loss_small
                #print('pred', lm_logits_top)
                #print('org', small_logit)

            elif self.org_prob_reg_w > 0:
                loss_fct = nn.MSELoss()
                lm_small_top_prob = torch.softmax(lm_logits_top, dim=-1)
                #print(lm_small_top_prob.size())
                #print(all_prob_curves_norm[:,:,:,0].size())
                if self.reg_inv_temp != 1:
                    reg_prob = torch.softmax(small_logit * self.reg_inv_temp, dim=-1)
                else:
                    reg_prob = all_prob_curves_norm[:,:,:,0]
                loss_small = torch.pow( loss_fct(lm_small_top_prob, reg_prob), 0.5)
                loss = loss + self.org_prob_reg_w * loss_small
            #print(loss, loss_small)
        
        if not return_dict:
            output = (lm_logits, lm_top_prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

class GPTNeoXForLogitCorrectionSimple(GPTNeoXPreTrainedModel):
    def __init__(self, config, log_model_size, decay_function='exp', poly_degree=10, model_logit_decay = False, logit_reg_w=0.8, emphasize_last_w=10, rev_curves=True):
        super().__init__(config)
        self.gpt_neox = GPTNeoXModel(config)
        self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.mlp = MLP_para(log_model_size, decay_function, poly_degree, emphasize_last_w, rev_curves)
        self.logit_reg_w = logit_reg_w
        print('logit_reg_w', self.logit_reg_w)
        self.post_init()
        self.model_logit_decay = model_logit_decay
        self.rev_curves = rev_curves
        if not rev_curves:
            assert decay_function == 'exp'

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.gpt_neox(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states) #could save the computation a little if we only compute dot product with the top word index

        loss = None
        lm_top_prob = None
        if labels is not None:
            all_prob_curves, LLM_top_w, LLM_top_logit, small_logit = uncompress_label_tensor(labels)
            all_prob_curves = all_prob_curves.to(lm_logits.device)
            LLM_top_logit = LLM_top_logit.to(lm_logits.device)
            LLM_top_w = LLM_top_w.to(lm_logits.device)
            lm_logits_top = torch.gather(lm_logits, dim=-1, index=LLM_top_w) #(bsz, seq_len, vocab_size) -> (bsz, seq_len, top_k)
            if self.model_logit_decay:
                all_logit_curves = torch.log(1e-10 + all_prob_curves)
                all_logit_curves_norm = all_logit_curves - all_logit_curves.mean(dim=-2, keepdim=True)
                top_logit_out = LLM_top_logit - lm_logits_top
                top_logit_out_norm = top_logit_out - top_logit_out.mean(dim=-2, keepdim=True)
                if self.rev_curves:
                    all_logit_curves_norm_rev, top_logit_out_norm_rev = reverse_increasing_prob(all_logit_curves_norm, top_logit_out_norm, 0)
                else:
                    all_logit_curves_norm_rev = all_logit_curves_norm
                    top_logit_out_norm_rev = top_logit_out_norm
                loss, decay_para = self.mlp(all_logit_curves_norm_rev, top_logit_out_norm_rev)
                
            else:
                lm_top_prob = torch.softmax(LLM_top_logit - lm_logits_top, dim=-1).clone()
                all_prob_curves_norm = all_prob_curves / (1e-23 + all_prob_curves.sum(dim=-2, keepdim=True)) #(bsz, seq_len, num_k, num_models)

                if self.rev_curves:
                    all_prob_curves_norm_rev, lm_top_prob_rev = reverse_increasing_prob(all_prob_curves_norm, lm_top_prob)
                else:
                    all_prob_curves_norm_rev = all_prob_curves_norm
                    lm_top_prob_rev = lm_top_prob
                loss, decay_para = self.mlp(all_prob_curves_norm_rev, lm_top_prob_rev)
            if self.logit_reg_w > 0:
                loss_fct = nn.MSELoss()
                loss_small = torch.pow( loss_fct(lm_logits_top, small_logit), 0.5)
                loss = loss + self.logit_reg_w * loss_small


        
        if not return_dict:
            output = (lm_logits, lm_top_prob) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        #return TokenClassifierOutput(
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values

        )

if can_load_Qwen:
    class Qwen2ForLogitCorrection(Qwen2PreTrainedModel):
        _tied_weights_keys = ["lm_head.weight"]
        
        def __init__(self, config, log_model_size, decay_function='exp', poly_degree=10, model_logit_decay = False, logit_reg_w=0.8, emphasize_last_w=10, rev_curves=True):
            super().__init__(config)
            self.model = Qwen2Model(config)

            # the lm_head weight is automatically tied to the embed tokens weight
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

            self.mlp = MLP_para(log_model_size, decay_function, poly_degree, emphasize_last_w)
            #self.logit_reg_w = 1.0
            self.logit_reg_w = logit_reg_w
            print('logit_reg_w', self.logit_reg_w)
            #self.logit_reg_w = 0.8
            #self.org_prob_reg_w = 5
            #self.org_prob_reg_w = 7.5
            self.org_prob_reg_w = 0
            self.reg_inv_temp = 0.5
            #self.org_prob_reg_w = 20
            # Initialize weights and apply final processing
            self.post_init()
            assert rev_curves


        def get_input_embeddings(self):
            return self.model.embed_tokens

        def set_input_embeddings(self, value):
            self.model.embed_tokens = value

        def get_output_embeddings(self):
            return self.lm_head

        def set_output_embeddings(self, new_embeddings):
            self.lm_head = new_embeddings

        def set_decoder(self, decoder):
            self.model = decoder

        def get_decoder(self):
            return self.model

        def forward(
            self,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
        ) -> Union[Tuple, CausalLMOutputWithPast]:

            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = outputs[0]
            lm_logits = self.lm_head(hidden_states)
            lm_logits = lm_logits.float()
            
            loss = None
            lm_top_prob = None
            if labels is not None:
                #labels = labels.to(lm_logits.device)
                all_prob_curves, LLM_top_w, LLM_top_logit, small_logit = uncompress_label_tensor(labels)
                all_prob_curves = all_prob_curves.to(lm_logits.device)
                LLM_top_logit = LLM_top_logit.to(lm_logits.device)
                LLM_top_w = LLM_top_w.to(lm_logits.device)
                lm_logits_top = torch.gather(lm_logits, dim=-1, index=LLM_top_w) #(bsz, seq_len, vocab_size) -> (bsz, seq_len, top_k)
                lm_top_prob = torch.softmax(LLM_top_logit - lm_logits_top, dim=-1).clone()
                #shifted_logits = LLM_top_logit - lm_logits_top 
                #lm_top_exp = torch.exp( shifted_logits - shifted_logits.mean() )
                #lm_top_prob = lm_top_exp / (1e-23 + lm_top_exp.sum(dim=-1, keepdim=True) )  #(bsz, seq_len, num_k)
                #lm_top_prob = lm_top_exp / ( lm_top_exp.sum(dim=-1, keepdim=True) )  #(bsz, seq_len, num_k)

                all_prob_curves_norm = all_prob_curves / (1e-23 + all_prob_curves.sum(dim=-2, keepdim=True)) #(bsz, seq_len, num_k, num_models)

                all_prob_curves_norm_rev, lm_top_prob_rev = reverse_increasing_prob(all_prob_curves_norm, lm_top_prob)
                loss, decay_para = self.mlp(all_prob_curves_norm_rev, lm_top_prob_rev)
                if self.logit_reg_w > 0:
                    loss_fct = nn.MSELoss()
                    loss_small = torch.pow( loss_fct(lm_logits_top, small_logit), 0.5)
                    loss = loss + self.logit_reg_w * loss_small
                    #print('pred', lm_logits_top)
                    #print('org', small_logit)

                elif self.org_prob_reg_w > 0:
                    loss_fct = nn.MSELoss()
                    lm_small_top_prob = torch.softmax(lm_logits_top, dim=-1)
                    #print(lm_small_top_prob.size())
                    #print(all_prob_curves_norm[:,:,:,0].size())
                    if self.reg_inv_temp != 1:
                        reg_prob = torch.softmax(small_logit * self.reg_inv_temp, dim=-1)
                    else:
                        reg_prob = all_prob_curves_norm[:,:,:,0]
                    loss_small = torch.pow( loss_fct(lm_small_top_prob, reg_prob), 0.5)
                    loss = loss + self.org_prob_reg_w * loss_small
                #print(loss, loss_small)
            
            if not return_dict:
                output = (lm_logits, lm_top_prob) + outputs[2:]
                return ((loss,) + output) if loss is not None else output

            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                past_key_values=outputs.past_key_values
            )


        # Copied from transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation
        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            inputs_embeds=None,
            cache_position=None,
            position_ids=None,
            use_cache=True,
            **kwargs,
        ):
            # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
            # Exception 1: when passing input_embeds, input_ids may be missing entries
            # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
            if past_key_values is not None:
                if inputs_embeds is not None:  # Exception 1
                    input_ids = input_ids[:, -cache_position.shape[0] :]
                elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                    input_ids = input_ids[:, cache_position]

            if attention_mask is not None and position_ids is None:
                # create position_ids on the fly for batch generation
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]

                    # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                    position_ids = position_ids.clone(memory_format=torch.contiguous_format)

            # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
            if inputs_embeds is not None and cache_position[0] == 0:
                model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
            else:
                # The clone here is for the same reason as for `position_ids`.
                model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

            if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
                if model_inputs["inputs_embeds"] is not None:
                    batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                    device = model_inputs["inputs_embeds"].device
                else:
                    batch_size, sequence_length = model_inputs["input_ids"].shape
                    device = model_inputs["input_ids"].device

                dtype = self.lm_head.weight.dtype
                min_dtype = torch.finfo(dtype).min

                attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                    attention_mask,
                    sequence_length=sequence_length,
                    target_length=past_key_values.get_max_length(),
                    dtype=dtype,
                    device=device,
                    min_dtype=min_dtype,
                    cache_position=cache_position,
                    batch_size=batch_size,
                )

            model_inputs.update(
                {
                    "position_ids": position_ids,
                    "cache_position": cache_position,
                    "past_key_values": past_key_values,
                    "use_cache": use_cache,
                    "attention_mask": attention_mask,
                }
            )
            return model_inputs


