import torch.nn as nn
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXPreTrainedModel, GPTNeoXModel, GPT_NEOX_INPUTS_DOCSTRING
from transformers.models.opt.modeling_opt import OPTPreTrainedModel, OPTModel

#from typing import Optional
import torch
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.modeling_outputs import TokenClassifierOutput

class GPTNeoXForEXPEntropyClassification(GPTNeoXPreTrainedModel):
    #def __init__(self, config, log_model_size, use_a4=False, use_a56=False, use_a10=False):
    def __init__(self, config, log_model_size, decay_function='exp'):
        super().__init__(config)
        self.predict_last = True
        self.positive_mode = 'exp'
        weight_decay_ratio = 0.8 
        self.decay_function = decay_function

        self.log_model_size = torch.tensor(log_model_size)
        weight_list = []
        weight = 1
        for i in range(len(log_model_size)):
            weight_list.append(weight)
            weight *= weight_decay_ratio
        weight_list.reverse()
        self.weight_list = torch.tensor(weight_list)
        self.weight_list = self.weight_list / self.weight_list.mean()
        self.num_labels = 4

        if self.predict_last:
            self.num_labels += 1
            self.weight_list = torch.cat( [ self.weight_list, torch.ones(1) ] )

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def compute_entropy_prediction(self,logits, log_model_size):
        bsz, seq_len, must_eight, num_models = logits.size()
        
        small_num = 0
        if self.positive_mode == 'exp':
            logits_pos = torch.exp(logits)
        elif self.positive_mode == 'ReLU':
            very_small_num = 1e-20
            logits_pos = torch.maximum(logits , torch.tensor(very_small_num,device=logits.device) )
        c = logits_pos[:,:,0,:]
        #if not self.only_positive:
        #    logits_pos = logits
        b = logits_pos[:,:,1,:]
        f, g = logits_pos[:,:,2,:], logits_pos[:,:,3,:]
        model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_models) - g), torch.tensor(small_num, device=logits.device) )
        #a05 = logits_pos[:,:,4,:]
        if self.decay_function == 'exp':
            entropy_pred = c + b*( torch.exp(- model_log_size_norm ) )
        elif self.decay_function == 'logistic':
            entropy_pred = c + b / ( 1 + torch.exp(model_log_size_norm ) )

        if self.predict_last:
            last_ent = logits_pos[:,:,-1,0]
            entropy_pred = torch.cat( (entropy_pred, last_ent.unsqueeze(dim=-1)), dim=-1 )

        return entropy_pred, logits_pos


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
        create_very_large_size: Optional[bool] = False
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        hidden_states = self.dropout(hidden_states)
        if create_very_large_size:
            very_large_size = 2 * self.log_model_size[-1] - self.log_model_size[0]
            log_model_size = torch.cat( (self.log_model_size, torch.tensor([very_large_size]) ) , 0) 
        else:
            log_model_size = self.log_model_size
        num_models = log_model_size.size(0)
        logits = self.classifier(hidden_states).unsqueeze(-1).expand(-1,-1,-1,num_models) #(bsz, seq_len, num_labels, num_models)
        
        entropy_pred, logits_pos = self.compute_entropy_prediction(logits, log_model_size)

        loss = None
        if labels is not None:
            if self.predict_last:
                labels = torch.cat( ( labels, labels[:, :, -1].unsqueeze(-1) ), dim=-1 )
            labels = labels.to(logits.device)
            #assert labels.size(-1) == num_models
            loss_fct = nn.MSELoss()
            loss = torch.pow( loss_fct(entropy_pred, labels), 0.5)

        if not return_dict:
            output = (logits, entropy_pred, logits_pos) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class OPTForEntropyClassification(OPTPreTrainedModel):
    def __init__(self, config, log_model_size, poly_degree=10):
        super().__init__(config)
        self.model = OPTModel(config)
        self.poly_degree = poly_degree
        self.predict_last = True
        self.positive_mode = 'exp'
        weight_decay_ratio = 0.8 

        self.log_model_size = torch.tensor(log_model_size)
        weight_list = []
        weight = 1
        for i in range(len(log_model_size)):
            weight_list.append(weight)
            weight *= weight_decay_ratio
        weight_list.reverse()
        self.weight_list = torch.tensor(weight_list)
        self.weight_list = self.weight_list / self.weight_list.mean()
        self.num_labels = 5 + poly_degree

        if self.predict_last:
            self.num_labels += 1
            self.weight_list = torch.cat( [ self.weight_list, torch.ones(1) ] )

        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def compute_entropy_prediction(self,logits, log_model_size):
        bsz, seq_len, must_eight, num_models = logits.size()
        
        small_num = 1
        if self.positive_mode == 'exp':
            logits_pos = torch.exp(logits)
        elif self.positive_mode == 'ReLU':
            very_small_num = 1e-20
            logits_pos = torch.maximum(logits , torch.tensor(very_small_num,device=logits.device) )
        c = logits_pos[:,:,0,:]
        b = logits_pos[:,:,1,:]
        f, g = logits_pos[:,:,2,:], logits_pos[:,:,3,:]
        model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_models) - g), torch.tensor(small_num, device=logits.device) )
        a05 = logits_pos[:,:,4,:]
        entropy_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5))
        for i in range(self.poly_degree):
            entropy_pred = entropy_pred + b*(logits_pos[:,:,5+i,:] / torch.pow(model_log_size_norm,i+1))

        if self.predict_last:
            last_ent = logits_pos[:,:,-1,0]
            entropy_pred = torch.cat( (entropy_pred, last_ent.unsqueeze(dim=-1)), dim=-1 )

        return entropy_pred, logits_pos


    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        create_very_large_size: Optional[bool] = False
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model.decoder(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        if create_very_large_size:
            very_large_size = 2 * self.log_model_size[-1] - self.log_model_size[0]
            log_model_size = torch.cat( (self.log_model_size, torch.tensor([very_large_size]) ) , 0) 
        else:
            log_model_size = self.log_model_size
        num_models = log_model_size.size(0)
        logits = self.classifier(hidden_states).unsqueeze(-1).expand(-1,-1,-1,num_models) #(bsz, seq_len, num_labels, num_models)
        
        entropy_pred, logits_pos = self.compute_entropy_prediction(logits, log_model_size)

        loss = None
        if labels is not None:
            if self.predict_last:
                labels = torch.cat( ( labels, labels[:, :, -1].unsqueeze(-1) ), dim=-1 )
            labels = labels.to(logits.device)
            #assert labels.size(-1) == num_models
            loss_fct = nn.MSELoss()
            loss = torch.pow( loss_fct(entropy_pred, labels), 0.5)

        if not return_dict:
            output = (logits, entropy_pred, logits_pos) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class GPTNeoXForEntropyClassification(GPTNeoXPreTrainedModel):
    #def __init__(self, config, log_model_size, use_a4=False, use_a56=False, use_a10=False):
    def __init__(self, config, log_model_size, poly_degree=10):
        super().__init__(config)
        #self.predict_last = False
        #self.only_positive = True
        self.poly_degree = poly_degree
        #self.use_a4 = use_a4
        #self.use_a56 = use_a56
        #self.use_a10 = use_a10
        self.predict_last = True
        self.positive_mode = 'exp'
        weight_decay_ratio = 0.8 
        #self.positive_mode = 'ReLU'
        #self.only_positive = False

        #self.num_labels = config.num_labels
        self.log_model_size = torch.tensor(log_model_size)
        weight_list = []
        weight = 1
        for i in range(len(log_model_size)):
            weight_list.append(weight)
            weight *= weight_decay_ratio
        weight_list.reverse()
        self.weight_list = torch.tensor(weight_list)
        self.weight_list = self.weight_list / self.weight_list.mean()
        self.num_labels = 5 + poly_degree
        #if self.use_a4:
        #    self.num_labels += 1
        #if self.use_a56:
        #    assert self.use_a4 
        #    self.num_labels += 2
        #if self.use_a10:
        #    assert self.use_a4 
        #    assert self.use_a56 
        #    self.num_labels += 4

        if self.predict_last:
            self.num_labels += 1
            self.weight_list = torch.cat( [ self.weight_list, torch.ones(1) ] )

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        #self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        # Initialize weights and apply final processing
        self.post_init()
        #self.classifier.weight.data[:] = 0
        #self.classifier.bias.data[:] = 0 

    def compute_entropy_prediction(self,logits, log_model_size):
        bsz, seq_len, must_eight, num_models = logits.size()
        
        small_num = 1
        if self.positive_mode == 'exp':
            logits_pos = torch.exp(logits)
        elif self.positive_mode == 'ReLU':
            very_small_num = 1e-20
            logits_pos = torch.maximum(logits , torch.tensor(very_small_num,device=logits.device) )
        c = logits_pos[:,:,0,:]
        #if not self.only_positive:
        #    logits_pos = logits
        b = logits_pos[:,:,1,:]
        f, g = logits_pos[:,:,2,:], logits_pos[:,:,3,:]
        model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_models) - g), torch.tensor(small_num, device=logits.device) )
        a05 = logits_pos[:,:,4,:]
        entropy_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5))
        for i in range(self.poly_degree):
            entropy_pred = entropy_pred + b*(logits_pos[:,:,5+i,:] / torch.pow(model_log_size_norm,i+1))

        #a05, a1, a2, a3 = logits_pos[:,:,4,:], logits_pos[:,:,5,:], logits_pos[:,:,6,:], logits_pos[:,:,7,:]
        #if self.use_a4:
        #    a4 = logits_pos[:,:,8,:]
        #    if self.use_a56:
        #        a5 = logits_pos[:,:,9,:]
        #        a6 = logits_pos[:,:,10,:]
        #        if self.use_a10:
        #            a7 = logits_pos[:,:,11,:]
        #            a8 = logits_pos[:,:,12,:]
        #            a9 = logits_pos[:,:,13,:]
        #            a10 = logits_pos[:,:,14,:]
        #            entropy_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) + a4 / torch.pow(model_log_size_norm,4) + a5 / torch.pow(model_log_size_norm,5) + a6 / torch.pow(model_log_size_norm,6) + a7 / torch.pow(model_log_size_norm,7) + a8 / torch.pow(model_log_size_norm,8) + a9 / torch.pow(model_log_size_norm,9) + a10 / torch.pow(model_log_size_norm,10) )
        #        else:
        #            entropy_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) + a4 / torch.pow(model_log_size_norm,4) + a5 / torch.pow(model_log_size_norm,5) + a6 / torch.pow(model_log_size_norm,6) )
        #    else:
        #        entropy_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) + a4 / torch.pow(model_log_size_norm,4) )
        #else:
        #    entropy_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5)  + a1 / model_log_size_norm + a2 / torch.pow(model_log_size_norm,2) + a3 / torch.pow(model_log_size_norm,3) )

        if self.predict_last:
            last_ent = logits_pos[:,:,-1,0]
            entropy_pred = torch.cat( (entropy_pred, last_ent.unsqueeze(dim=-1)), dim=-1 )

        return entropy_pred, logits_pos


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
        create_very_large_size: Optional[bool] = False
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        hidden_states = self.dropout(hidden_states)
        if create_very_large_size:
            very_large_size = 2 * self.log_model_size[-1] - self.log_model_size[0]
            log_model_size = torch.cat( (self.log_model_size, torch.tensor([very_large_size]) ) , 0) 
        else:
            log_model_size = self.log_model_size
        num_models = log_model_size.size(0)
        logits = self.classifier(hidden_states).unsqueeze(-1).expand(-1,-1,-1,num_models) #(bsz, seq_len, num_labels, num_models)
        
        entropy_pred, logits_pos = self.compute_entropy_prediction(logits, log_model_size)

        loss = None
        if labels is not None:
            if self.predict_last:
                labels = torch.cat( ( labels, labels[:, :, -1].unsqueeze(-1) ), dim=-1 )
            labels = labels.to(logits.device)
            #assert labels.size(-1) == num_models
            loss_fct = nn.MSELoss()
            loss = torch.pow( loss_fct(entropy_pred, labels), 0.5)

            #bsz, seq_len, num_models = entropy_pred.size()
            #weight_list_expand = self.weight_list.to(logits.device).expand(bsz,seq_len,num_models) 
            #loss_fct = nn.MSELoss(reduction='none')
            #loss = torch.pow( (loss_fct(entropy_pred, labels) * weight_list_expand).mean() , 0.5)
            
            #labels = labels.to(logits.device)
            #loss_fct = CrossEntropyLoss()
            #loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits, entropy_pred, logits_pos) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

class GPTNeoXForScaledEntropyClassification(GPTNeoXPreTrainedModel):
    #def __init__(self, config, log_model_size, use_a4=False, use_a56=False, use_a10=False):
    def __init__(self, config, log_model_size, poly_degree=5):
        super().__init__(config)
        self.poly_degree = poly_degree
        self.predict_last = True
        self.positive_mode = 'exp'
        weight_decay_ratio = 0.8 

        self.log_model_size = torch.tensor(log_model_size)
        weight_list = []
        weight = 1
        for i in range(len(log_model_size)):
            weight_list.append(weight)
            weight *= weight_decay_ratio
        weight_list.reverse()
        self.weight_list = torch.tensor(weight_list)
        self.weight_list = self.weight_list / self.weight_list.mean()
        self.num_labels = 2 + (1 + poly_degree) * 3

        if self.predict_last:
            self.num_labels += 1
            self.weight_list = torch.cat( [ self.weight_list, torch.ones(1) ] )

        self.gpt_neox = GPTNeoXModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)

        self.post_init()

    def compute_entropy_prediction(self,logits, log_model_size):
        bsz, seq_len, must_eight, num_models = logits.size()
        
        small_num = 1
        if self.positive_mode == 'exp':
            logits_pos = torch.exp(logits)
        elif self.positive_mode == 'ReLU':
            very_small_num = 1e-20
            logits_pos = torch.maximum(logits , torch.tensor(very_small_num,device=logits.device) )
        c = logits_pos[:,:,0,:]
        b = logits_pos[:,:,1,:]
        f, g = logits_pos[:,:,2,:], logits_pos[:,:,3,:]
        model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_models) - g), torch.tensor(small_num, device=logits.device) )
        a05 = logits_pos[:,:,4,:]
        entropy_pred = c + b*(a05 / torch.pow(model_log_size_norm,0.5))
        for i in range(self.poly_degree):
            f, g = logits_pos[:,:,5+3*i,:], logits_pos[:,:,6+3*i,:]
            model_log_size_norm = torch.maximum( f * (log_model_size.to(logits.device).expand(bsz,seq_len,num_models) - g), torch.tensor(small_num, device=logits.device) )
            entropy_pred = entropy_pred + b*(logits_pos[:,:,7+3*i,:] / torch.pow(model_log_size_norm,i+1))

        if self.predict_last:
            last_ent = logits_pos[:,:,-1,0]
            entropy_pred = torch.cat( (entropy_pred, last_ent.unsqueeze(dim=-1)), dim=-1 )

        return entropy_pred, logits_pos


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
        create_very_large_size: Optional[bool] = False
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
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
        hidden_states = self.dropout(hidden_states)
        if create_very_large_size:
            very_large_size = 2 * self.log_model_size[-1] - self.log_model_size[0]
            log_model_size = torch.cat( (self.log_model_size, torch.tensor([very_large_size]) ) , 0) 
        else:
            log_model_size = self.log_model_size
        num_models = log_model_size.size(0)
        logits = self.classifier(hidden_states).unsqueeze(-1).expand(-1,-1,-1,num_models) #(bsz, seq_len, num_labels, num_models)
        
        entropy_pred, logits_pos = self.compute_entropy_prediction(logits, log_model_size)

        loss = None
        if labels is not None:
            if self.predict_last:
                labels = torch.cat( ( labels, labels[:, :, -1].unsqueeze(-1) ), dim=-1 )
            labels = labels.to(logits.device)
            loss_fct = nn.MSELoss()
            loss = torch.pow( loss_fct(entropy_pred, labels), 0.5)

        if not return_dict:
            output = (logits, entropy_pred, logits_pos) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

