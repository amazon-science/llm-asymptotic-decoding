import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'REAL_sampling', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..')) 

from model import GPTNeoXForEntropyClassification
from model import OPTForEntropyClassification
from model_mlp_logit import GPTNeoXForLogitCorrection, OPTForLogitCorrection,  Qwen2ForLogitCorrection
from transformers import AutoTokenizer, AutoModelForCausalLM

import torch

def extract_param(raw_name, param_prefix, param_suffix):
    if param_prefix not in raw_name:
        return 3
    prefix_start = raw_name.index(param_prefix)
    start_idx = prefix_start+len(param_prefix)
    if param_suffix is None:
        return int(raw_name[start_idx:])
    end_idx = raw_name[start_idx:].index(param_suffix)
    #print(start_idx, end_idx)
    return int(raw_name[start_idx:start_idx+end_idx])

def compute_model_last(model_ent, org_left_word_tensor):
    org_left_word_tensor_ent = org_left_word_tensor.to(model_ent.device)
    get_pos = -1
    with torch.no_grad():
        #output = model_ent(org_left_word_tensor_ent, return_dict=False)
        output = model_ent(org_left_word_tensor_ent, return_dict=False, create_very_large_size=True)
        #output = model_ent(org_left_word_tensor_ent, return_dict=False, create_very_large_size=True, use_cache=True)
    ent_pred = output[1]
    logit_pos = output[2]
    c = logit_pos[:,get_pos,0,0]
    pred_last_ent = ent_pred[:,get_pos,-1]
    #curve_last_ent = ent_pred[:,get_pos,-2]
    curve_large_ent = ent_pred[:,get_pos,-2]
    curve_last_ent = ent_pred[:,get_pos,-3]
    uncertainty_score1 = (curve_last_ent - c)
    uncertainty_score2 = (pred_last_ent - c)
    return c, pred_last_ent, curve_last_ent, curve_large_ent, uncertainty_score1, uncertainty_score2


def update_log_size( log_model_size, student_model_name ):
    if '_sub' in student_model_name:
        sub_num = student_model_name.split('_sub')[0].split('_')[-1]
        assert sub_num.isnumeric(), print(sub_num)
        log_model_size = [ log_model_size[int(x)] for x in list(sub_num) ]
        return log_model_size
    return log_model_size

class LogitsWarper:
    """Abstract base class for all logit warpers that can be applied during generation with multinomial sampling."""

    #@add_start_docstrings(LOGITS_PROCESSOR_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

class LogitsProcessorList(list):
    """
    This class can be used to create a list of [`LogitsProcessor`] or [`LogitsWarper`] to subsequently process a
    `scores` input tensor. This class inherits from list and adds a specific *__call__* method to apply each
    [`LogitsProcessor`] or [`LogitsWarper`] to the inputs.
    """

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> torch.FloatTensor:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. [What are input IDs?](../glossary#input-ids)
            scores (`torch.FloatTensor` of shape `(batch_size, config.vocab_size)`):
                Prediction scores of a language modeling head. These can be logits for each vocabulary when not using
                beam search or log softmax for each vocabulary token when using beam search
            kwargs (`Dict[str, Any]`, *optional*):
                Additional kwargs that are specific to a logits processor.

        Return:
            `torch.FloatTensor` of shape `(batch_size, config.vocab_size)`:
                The processed prediction scores.

        """
        for processor in self:
            function_args = inspect.signature(processor.__call__).parameters
            if len(function_args) > 2:
                if not all(arg in kwargs for arg in list(function_args.keys())[2:]):
                    raise ValueError(
                        f"Make sure that all the required parameters: {list(function_args.keys())} for "
                        f"{processor.__class__} are passed to the logits processor."
                    )
                scores = processor(input_ids, scores, **kwargs)
            else:
                scores = processor(input_ids, scores)
        return scores


class FETopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs `final entropy` top-p. 
    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        final_entropy_model_path (`str`): 
            The path of the model that predicts the final entropy
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, decay_temperature: float, final_entropy_model_path: str, tokenizer, tokenizer_ent, sample_sub_method: str, window_size: int,  filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, sent_idx_decay_rate: float = 1, use_top_k: bool = False, student_model_name: str = None, use_CD_alpha: bool = False, use_AP: bool = False, device=None, use_log_softmax=True):
        #top_p = float(top_p)
        #if not use_top_k and ( top_p < 0 or top_p > 1.0 ):
        #    raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        #device = torch.device("cpu")

        self.window_size = window_size
        if self.window_size is not None:
            assert sample_sub_method[-4:] == '_win' 
        #self.top_p = top_p
        #log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
        #use_a4 = False
        #use_a56 = False
        #use_a10 = False
        #if '_a4_' in final_entropy_model_path:
        #    use_a4 = True
        #if '_a6_' in final_entropy_model_path:
        #    use_a4 = True
        #    use_a56 = True
        #if '_a10_' in final_entropy_model_path:
        #    use_a4 = True
        #    use_a56 = True
        #    use_a10 = True

        self.sample_sub_method = sample_sub_method
        self.decay_temperature = decay_temperature
        poly_degree = extract_param(final_entropy_model_path, '_a', '_')
        #self.model_ent = GPTNeoXForEntropyClassification.from_pretrained(final_entropy_model_path, log_model_size=log_model_size, use_a4=use_a4, use_a56=use_a56, use_a10=use_a10)
        if 'OPT_' in final_entropy_model_path:
            log_model_size = [18.2771614, 19.5373201, 20.9161984, 21.6486751, 22.5877428]
            model_class = OPTForEntropyClassification
        else:
            log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
            model_class = GPTNeoXForEntropyClassification
        self.model_ent = model_class.from_pretrained(final_entropy_model_path, log_model_size=log_model_size, poly_degree=poly_degree)
        self.model_ent.eval()
        self.model_ent = self.model_ent.to(device)

        #self.debug = True
        self.debug = False
        if self.debug:
            self.tokenizer = tokenizer
        #    self.tokenizer = AutoTokenizer.from_pretrained('EleutherAI/pythia-70m-deduped', padding_side='left')
        
        self.tokenizer_mismatch = False
        if tokenizer.name_or_path != tokenizer_ent.name_or_path:
            print('Tokenizer mismatch! Need to tokenize the input on the fly.')
            self.tokenizer_mismatch = True
            self.tokenizer = tokenizer
            self.tokenizer_ent = tokenizer_ent

        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

        self.sent_idx_decay_rate = sent_idx_decay_rate
        self.bsz = None
        self.input_len = None
        self.sent_decay_w = 1
        self.top_p_lower_cap = 0.3
        if self.sent_idx_decay_rate < 1:
            self.tokenizer_ent = tokenizer_ent

        self.use_top_k = use_top_k

        self.use_AP = use_AP
        self.use_CD = False
        if student_model_name is not None:
            assert use_top_k == False and sent_idx_decay_rate == 1 #Do not accidentally combine CD with other methods for now
            #assert 'pythia' in tokenizer.name_or_path and 'pythia' in student_model_name #Do not support LLM for now
            self.use_CD = True
            self.inv_temp = top_p
            if self.use_AP:
                #use_a4 = False
                #use_a56 = False
                #use_a10 = False
                #if '_a4_' in student_model_name:
                #    use_a4 = True
                #if '_a6_' in student_model_name:
                #    use_a4 = True
                #    use_a56 = True
                #if '_a10_' in student_model_name:
                #    use_a4 = True
                #    use_a56 = True
                #    use_a10 = True
                
                model_logit_decay = False
                if '_ld_' in student_model_name:
                    model_logit_decay = True

                poly_degree = 10
                if '_exp_decay_' in student_model_name:
                    decay_function='exp'
                elif '_logistic_decay_' in student_model_name:
                    decay_function='logistic'
                elif 'scaled_a' in student_model_name:
                    decay_function='scaled_poly'
                    poly_degree = extract_param(student_model_name, '_a', '_')
                else:
                    decay_function='poly'
                    poly_degree = extract_param(student_model_name, '_a', '_')

                if 'prob_opt' in student_model_name:
                    log_model_size = [18.2771614, 19.5373201, 20.9161984, 21.6486751, 22.5877428]
                    self.model_st = OPTForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
                elif 'prob_Qwen_4b' in student_model_name:
                    log_model_size = [19.5469252164,21.1456953943,21.9934232469]
                    self.model_st = Qwen2ForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
                else:
                    log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
                    log_model_size = update_log_size( log_model_size, student_model_name )
                    if '_sub' in student_model_name:
                        sub_num = student_model_name.split('_sub')[0].split('_')[-1]
                        assert sub_num.isnumeric(), print(sub_num)
                        log_model_size = [ log_model_size[int(x)] for x in list(sub_num) ]

                    self.model_st = GPTNeoXForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
            else:
                self.model_st = AutoModelForCausalLM.from_pretrained(student_model_name.replace('models/','')).to(device)
            self.model_st.eval()
            self.use_log_softmax = use_log_softmax
            #self.student_use_gen_tokenizer = student_use_gen_tokenizer
        
        self.use_CD_alpha = use_CD_alpha
        self.past_key_values = None
        self.past_input_len = None


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            
        input_ids_in = input_ids
        org_device = scores.device
        scores = scores.to(self.model_ent.device)
        
        if self.tokenizer_mismatch:
             current_str = self.tokenizer.batch_decode(input_ids_in, skip_special_tokens = True)
             input_ids_in = self.tokenizer_ent(current_str, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True)
             input_ids_in = input_ids_in['input_ids']
        
        if self.sent_idx_decay_rate < 1:
            bsz, seq_len = input_ids_in.size()
            if self.bsz is None or bsz != self.bsz or seq_len != self.input_len + 1: #the start of a new sequence
                self.sent_decay_w = torch.ones( (bsz) ).to(self.model_ent.device)
            self.bsz = bsz
            self.input_len = seq_len
            for i in range(self.bsz):
                if '.' in self.tokenizer_ent.convert_ids_to_tokens( [input_ids_in[i, -1]] )[0]:
                    self.sent_decay_w[i] = 1

        if self.window_size is not None and self.sample_sub_method[-4:] == '_win':
            input_ids_in = input_ids_in[:,-self.window_size:]
            #print(input_ids_in.size())

        c, pred_last_ent, curve_last_ent, curve_large_ent, uncertainty_score1, uncertainty_score2 = compute_model_last(self.model_ent, input_ids_in)
        #top_p_decay_rate = c / curve_last_ent # (bsz)
        #top_p_decay_rate = torch.exp(c) / torch.exp(curve_last_ent) # (bsz)
        probs = None
        if self.sample_sub_method == 'exp_1' or self.sample_sub_method == 'exp_1_win':
            top_p_decay_rate = torch.exp( (c - curve_last_ent) / self.decay_temperature ) # (bsz)
        elif self.sample_sub_method == 'exp_real_e_only_win':
            probs = scores.softmax(dim=-1)
            ent = - (probs * (1e-23+probs).log() ).sum(dim=-1)
            top_p_decay_rate = torch.exp( - ent / self.decay_temperature ) # (bsz)
        elif self.sample_sub_method == 'exp_large_win':
            top_p_decay_rate = torch.exp( (curve_large_ent - curve_last_ent) / self.decay_temperature ) # (bsz)
        elif self.sample_sub_method == 'exp_e_only_win':
            top_p_decay_rate = torch.exp( (- curve_last_ent) / self.decay_temperature ) # (bsz)
        elif self.sample_sub_method == 'exp_raw_e_only_win':
            top_p_decay_rate = torch.exp( (- pred_last_ent) / self.decay_temperature ) # (bsz)
        elif self.sample_sub_method == 'exp_1_norm':
            top_p_decay_rate = torch.exp( (c - curve_last_ent) / curve_last_ent / self.decay_temperature ) # (bsz)
        elif self.sample_sub_method == 'exp_2':
            probs = scores.softmax(dim=-1)
            ent = - (probs * (1e-23+probs).log() ).sum(dim=-1)
            neg_pred_ent = torch.minimum(pred_last_ent - ent, torch.zeros_like(pred_last_ent))
            top_p_decay_rate = torch.exp( (c - curve_last_ent + neg_pred_ent ) / self.decay_temperature ) # (bsz)
        #top_p = self.top_p * top_p_decay_rate * self.sent_decay_w
        top_p = top_p_decay_rate * self.sent_decay_w
        if self.debug:
            print(c)
            print(top_p)
            #input_text = self.tokenizer.batch_decode(input_ids)
            print(input_text)
            if self.window_size is not None and self.sample_sub_method[-4:] == '_win':
                input_text = self.tokenizer.batch_decode(input_ids_in)
                print(input_text)

        if self.use_CD_alpha:
            if probs is None:
                probs = scores.softmax(dim=-1)
            indices_to_remove = probs < (torch.max(probs, dim=-1)[0] * self.top_p * (1 - top_p_decay_rate ) / (1 + top_p_decay_rate))[..., None]

        elif self.use_top_k:
            top_p = torch.maximum(top_p.ceil() , torch.ones_like(top_p) )
            top_k = torch.minimum(top_p, scores.size(-1) * torch.ones_like(top_p))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = torch.zeros_like(scores, dtype=torch.bool)
            bsz, seq_len = scores.size()
            for i in range(bsz):
                indices_to_remove[i,:] = scores[i,:] < torch.topk(scores[i,:], int(top_k[i].item()))[0][-1, None]
        else: 
            top_p = torch.minimum(top_p, torch.ones_like(top_p_decay_rate))
            sorted_logits, sorted_indices = torch.sort(scores, descending=True)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            bsz, vocab_size = cumulative_probs.size()

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p.unsqueeze(dim=-1).expand(bsz, vocab_size)
            if self.min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        
        if self.use_CD:
            input_ids_CD = input_ids
            #input_ids_CD = input_ids_in
            #if args.student_use_gen_tokenizer:
            #    input_ids_CD = input_ids

            if self.past_key_values is None or self.past_input_len + 1 !=  input_ids_CD.size(1):
                outputs_students = self.model_st(input_ids_CD.to(self.model_st.device), return_dict = True)
            else:
                outputs_students = self.model_st(input_ids_CD[:,-1].unsqueeze(-1).to(self.model_st.device), past_key_values = self.past_key_values, return_dict = True)

            self.past_input_len = input_ids_CD.size(1)

            self.past_key_values = outputs_students.past_key_values

            #outputs_students = self.model_st(input_ids_CD.to(self.model_st.device), return_dict = True)
            scores_st = outputs_students.logits[:,-1,:]

            if self.use_log_softmax:
                scores = torch.log_softmax(scores,dim=-1)
                scores_st = torch.log_softmax(scores_st,dim=-1)

            #print(scores.size())
            #print(scores_st.size())
            scores[:,:scores_st.size(-1)] = scores[:,:scores_st.size(-1)] - self.inv_temp * scores_st
            scores[:,scores_st.size(-1):] = self.filter_value
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        
        if self.sent_idx_decay_rate < 1:
            self.sent_decay_w = torch.maximum(self.sent_decay_w * self.sent_idx_decay_rate, self.top_p_lower_cap* torch.ones_like(self.sent_decay_w) )
        scores = scores.to(org_device)

        return scores

class PeriodFactualTopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs `factual` top-p which decays p-value by top_p_decay_rate. 
    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        top_p_decay_rate (`float`): 
            If set to < 1, top_p value will be decayed every iteration as follows: top_p = top_p * top_p_decay_rate
        top_p_lower_cap (`float`, *optional*, default=0.0): 
            Sets the lower-bound for how far top_p value can be decayed down to.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, top_p_decay_rate: float, top_p_lower_cap: float = 0.0, tokenizer = None, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.og_top_p = top_p
        self.top_p = top_p

        self.top_p_decay_rate = top_p_decay_rate
        self.top_p_lower_cap = top_p_lower_cap

        self.tokenizer = tokenizer

        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

        self.bsz = None
        self.input_len = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        bsz, seq_len = input_ids.size()
        if self.bsz is None or bsz != self.bsz or seq_len != self.input_len + 1: #the start of a new sequence
            self.top_p = self.og_top_p * torch.ones( (bsz) )
        self.bsz = bsz
        self.input_len = seq_len
        for i in range(self.bsz):
            if '.' in self.tokenizer.convert_ids_to_tokens( [input_ids[i, -1]] )[0]:
                self.top_p[i] = self.og_top_p


        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        bsz, vocab_size = cumulative_probs.size()
        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p.unsqueeze(dim=-1).expand(bsz, vocab_size).to(cumulative_probs.device)
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        
        self.top_p = torch.maximum(self.top_p * self.top_p_decay_rate, self.top_p_lower_cap* torch.ones_like(self.top_p) )

        return scores

class FactualTopPLogitsWarper(LogitsWarper):
    """
    [`LogitsWarper`] that performs `factual` top-p which decays p-value by top_p_decay_rate. 
    Args:
        top_p (`float`):
            If set to < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept
            for generation.
        top_p_decay_rate (`float`): 
            If set to < 1, top_p value will be decayed every iteration as follows: top_p = top_p * top_p_decay_rate
        top_p_lower_cap (`float`, *optional*, default=0.0): 
            Sets the lower-bound for how far top_p value can be decayed down to.
        filter_value (`float`, *optional*, defaults to `-float("Inf")`):
            All filtered values will be set to this float value.
        min_tokens_to_keep (`int`, *optional*, defaults to 1):
            Minimum number of tokens that cannot be filtered.
    """

    def __init__(self, top_p: float, top_p_decay_rate: float, top_p_lower_cap: float = 0.0, reset_patience: int = 1, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1):
        top_p = float(top_p)
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")

        self.og_top_p = top_p
        self.top_p = top_p

        self.top_p_decay_rate = top_p_decay_rate
        self.top_p_lower_cap = top_p_lower_cap

        self.reset_patience = reset_patience
        self.p_reset_counter = 0

        self.filter_value = filter_value
        self.min_tokens_to_keep = min_tokens_to_keep

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.p_reset_counter +=1

        if self.p_reset_counter % self.reset_patience == 0:
            self.top_p, self.p_reset_counter = self.og_top_p, 0

        self.top_p = max(self.top_p * self.top_p_decay_rate, self.top_p_lower_cap)


        sorted_logits, sorted_indices = torch.sort(scores, descending=True)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

        # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        if self.min_tokens_to_keep > 1:
            # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
            sorted_indices_to_remove[..., : self.min_tokens_to_keep - 1] = 0
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, self.filter_value)

        return scores


class ContrastiveDecodingLogitsWarper(LogitsWarper):
    def __init__(self, alpha: float, temperature: float, student_model_name: str, tokenizer, tokenizer_st, filter_value: float = -float("Inf"), device=None, use_alpha=True, top_k = -1, use_log_softmax=True):

        #assert 'pythia' in tokenizer.name_or_path and 'pythia' in student_model_name

        #self.tokenizer_mismatch = False
        #if tokenizer_st is not None and tokenizer.name_or_path != tokenizer_st.name_or_path:
        #    print('Tokenizer mismatch! Need to tokenize the input on the fly.')
        #    self.tokenizer_mismatch = True
        #    self.tokenizer = tokenizer
        #    self.tokenizer_st = tokenizer_st

        self.model_st = AutoModelForCausalLM.from_pretrained(student_model_name.replace('models/','')).to(device)
        #if 'Qwen' in student_model_name:
        #    self.model_st.half()
        self.model_st.eval()
        self.alpha = alpha
        self.temperature = temperature
        self.filter_value = filter_value
        self.use_alpha = use_alpha
        self.top_k = top_k
        self.use_log_softmax = use_log_softmax

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        input_ids_in = input_ids
        #if self.tokenizer_mismatch:
        #     current_str = self.tokenizer.batch_decode(input_ids_in, skip_special_tokens = True)
        #     input_ids_in = self.tokenizer_st(current_str, padding=True, truncation=True, return_tensors='pt', add_special_tokens=True, return_length=True)
        #     input_ids_in = input_ids_in['input_ids']

        if self.use_alpha:
            probs = scores.softmax(dim=-1)
            indices_to_remove = probs < torch.max(probs, dim=-1)[0][..., None] * self.alpha
            if self.top_k > 0:
                indices_to_remove_topk = scores < torch.topk(scores, self.top_k)[0][..., -1, None]
                indices_to_remove = torch.logical_or(indices_to_remove,indices_to_remove_topk)
        else:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - self.alpha)
            # Keep at least min_tokens_to_keep
            self.min_tokens_to_keep = 1
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0
            if self.top_k > 0:
                sorted_indices_to_remove[..., :(sorted_indices_to_remove.size(-1)-self.top_k ) ] = 1
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)


        outputs_students = self.model_st(input_ids_in.to(self.model_st.device), return_dict = True)
        scores_st = outputs_students.logits[:,-1,:]
        
        #print(scores.size())
        #print(scores_st.size())
        if self.use_log_softmax:
            scores = torch.log_softmax(scores,dim=-1)
            scores_st = torch.log_softmax(scores_st,dim=-1)
        scores[:,scores_st.size(-1):] = self.filter_value
        scores[:,:scores_st.size(-1)] = scores[:,:scores_st.size(-1)] - self.temperature * scores_st
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class CDTopKLogitsWarper(LogitsWarper):
    def __init__(self, top_k: int, temperature: float, student_model_name: str, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, device=None):

        #assert 'pythia' in tokenizer.name_or_path and 'pythia' in student_model_name

        #self.tokenizer_mismatch = False
        #if tokenizer_st is not None and tokenizer.name_or_path != tokenizer_st.name_or_path:
        #    print('Tokenizer mismatch! Need to tokenize the input on the fly.')
        #    self.tokenizer_mismatch = True
        #    self.tokenizer = tokenizer
        #    self.tokenizer_st = tokenizer_st

        self.model_st = AutoModelForCausalLM.from_pretrained(student_model_name.replace('models/','')).to(device)
        self.model_st.eval()
        self.temperature = temperature
        self.filter_value = filter_value
        self.top_k = max(int(top_k), min_tokens_to_keep)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        top_k = min(self.top_k, scores.size(-1))  # Safety check
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        
        outputs_students = self.model_st(input_ids.to(self.model_st.device), return_dict = True)
        scores_st = outputs_students.logits[:,-1,:]
        scores[:,scores_st.size(-1):] = self.filter_value
        scores[:,:scores_st.size(-1)] = scores[:,:scores_st.size(-1)] - self.temperature * scores_st

        # Remove all tokens with a probability less than the last token of the top-k
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class APTopKLogitsWarper(LogitsWarper):
    def __init__(self, top_k: int, student_model_name: str, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, device=None):
        #if not isinstance(top_k, int) or top_k <= 0:
        #    raise ValueError(f"`top_k` has to be a strictly positive integer, but is {top_k}")

        #use_a4 = False
        #use_a56 = False
        #use_a10 = False
        #if '_a4_' in student_model_name:
        #    use_a4 = True
        #if '_a6_' in student_model_name:
        #    use_a4 = True
        #    use_a56 = True
        #if '_a10_' in student_model_name:
        #    use_a4 = True
        #    use_a56 = True
        #    use_a10 = True
        model_logit_decay = False
        if '_ld_' in model_name_or_path:
            model_logit_decay = True

        poly_degree = 10
        if '_exp_decay_' in student_model_name:
            decay_function='exp'
        elif '_logistic_decay_' in student_model_name:
            decay_function='logistic'
        elif 'scaled_a' in student_model_name:
            decay_function='scaled_poly'
            poly_degree = extract_param(student_model_name, '_a', '_')
        else:
            decay_function='poly'
            poly_degree = extract_param(student_model_name, '_a', '_')
        if 'prob_opt' in student_model_name:
            log_model_size = [18.2771614, 19.5373201, 20.9161984, 21.6486751, 22.5877428]
            self.model_ap = OPTForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
        else:
            log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
            log_model_size = update_log_size( log_model_size, student_model_name )
            self.model_ap = GPTNeoXForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
        self.model_ap.eval()
        self.top_k = max(int(top_k), min_tokens_to_keep)
        self.filter_value = filter_value

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        top_k = min(self.top_k, scores.size(-1))  # Safety check
        indices_to_remove = scores < torch.topk(scores, top_k)[0][..., -1, None]
        
        outputs_students = self.model_ap(input_ids.to(self.model_ap.device), return_dict = True)
        scores_st = outputs_students.logits[:,-1,:]
        scores[:,scores_st.size(-1):] = self.filter_value
        scores[:,:scores_st.size(-1)] = scores[:,:scores_st.size(-1)] - scores_st

        # Remove all tokens with a probability less than the last token of the top-k
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

class APCDTopPALogitsWarper(LogitsWarper):
    def __init__(self, top_p: float, student_model_name: str, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, device=None, use_alpha=True, temperature = 1, top_k = -1, CD_model_name = '', CD_inv_temp = 0.5):
        
        self.model_st = AutoModelForCausalLM.from_pretrained(CD_model_name).to(device)
        self.model_st.eval()

        self.CD_inv_temp = CD_inv_temp

        model_logit_decay = False
        if '_ld_' in student_model_name:
            model_logit_decay = True
        
        poly_degree = 10
        if '_exp_decay_' in student_model_name:
            decay_function='exp'
        elif '_logistic_decay_' in student_model_name:
            decay_function='logistic'
        elif 'scaled_a' in student_model_name:
            decay_function='scaled_poly'
            poly_degree = extract_param(student_model_name, '_a', '_')
        else:
            decay_function='poly'
            poly_degree = extract_param(student_model_name, '_a', '_')

        if 'prob_opt' in student_model_name:
            log_model_size = [18.2771614, 19.5373201, 20.9161984, 21.6486751, 22.5877428]
            self.model_ap = OPTForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
        else:
            log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
            log_model_size = update_log_size( log_model_size, student_model_name )
            self.model_ap = GPTNeoXForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
        self.model_ap.eval()
        self.inv_temperature = temperature
        
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.min_tokens_to_keep = min_tokens_to_keep
        self.top_p = float(top_p)
        self.top_k = top_k
        self.filter_value = filter_value
        self.use_alpha = use_alpha
        self.past_key_values = None
        self.past_input_len = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.use_alpha:
            probs = scores.softmax(dim=-1)
            indices_to_remove = probs < torch.max(probs, dim=-1)[0][..., None] * self.top_p
            if self.top_k > 0:
                #sorted_logits, sorted_indices = torch.sort(scores, descending=False)
                #sorted_indices_to_remove = torch.zeros_like(sorted_logits)
                #sorted_indices_to_remove[..., :(sorted_indices_to_remove.size(-1)-self.top_k ) ] = 1
                #indices_to_remove_topk = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                indices_to_remove_topk = scores < torch.topk(scores, self.top_k)[0][..., -1, None]
                indices_to_remove = torch.logical_or(indices_to_remove,indices_to_remove_topk)
        else:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0
            if self.top_k > 0:
                sorted_indices_to_remove[..., :(sorted_indices_to_remove.size(-1)-self.top_k ) ] = 1
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)


        outputs_CD = self.model_st(input_ids.to(self.model_st.device), return_dict = True)
        scores_CD = outputs_CD.logits[:,-1,:]

        if self.past_key_values is None or self.past_input_len + 1 !=  input_ids.size(1):
            outputs_students = self.model_ap(input_ids.to(self.model_ap.device), return_dict = True)
        else:
            outputs_students = self.model_ap(input_ids[:,-1].unsqueeze(-1).to(self.model_ap.device), past_key_values = self.past_key_values, return_dict = True)

        self.past_input_len = input_ids.size(1)

        self.past_key_values = outputs_students.past_key_values
        scores_st = outputs_students.logits[:,-1,:]
        
        scores[:,scores_st.size(-1):] = self.filter_value
        scores[:,:scores_st.size(-1)] = scores[:,:scores_st.size(-1)] - self.inv_temperature * scores_st - self.CD_inv_temp * scores_CD
        
        # scatter sorted tensors to original indexing
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores


class APTopPALogitsWarper(LogitsWarper):
    def __init__(self, top_p: float, student_model_name: str, filter_value: float = -float("Inf"), min_tokens_to_keep: int = 1, device=None, use_alpha=True, temperature = 1, top_k = -1, use_log_softmax=True):
        
        #use_a4 = False
        #use_a56 = False
        #use_a10 = False
        #if '_a4_' in student_model_name:
        #    use_a4 = True
        #if '_a6_' in student_model_name:
        #    use_a4 = True
        #    use_a56 = True
        #if '_a10_' in student_model_name:
        #    use_a4 = True
        #    use_a56 = True
        #    use_a10 = True
        model_logit_decay = False
        if '_ld_' in student_model_name:
            model_logit_decay = True
        
        poly_degree = 10
        if '_exp_decay_' in student_model_name:
            decay_function='exp'
        elif '_logistic_decay_' in student_model_name:
            decay_function='logistic'
        elif 'scaled_a' in student_model_name:
            decay_function='scaled_poly'
            poly_degree = extract_param(student_model_name, '_a', '_')
        else:
            decay_function='poly'
            poly_degree = extract_param(student_model_name, '_a', '_')

        if 'prob_opt' in student_model_name:
            log_model_size = [18.2771614, 19.5373201, 20.9161984, 21.6486751, 22.5877428]
            self.model_ap = OPTForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
        elif 'prob_Qwen_4b' in student_model_name:
            log_model_size = [19.5469252164,21.1456953943,21.9934232469]
            self.model_ap = Qwen2ForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
        else:
            log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
            log_model_size = update_log_size( log_model_size, student_model_name )
            self.model_ap = GPTNeoXForLogitCorrection.from_pretrained(student_model_name, log_model_size=log_model_size, decay_function=decay_function,poly_degree=poly_degree,model_logit_decay=model_logit_decay).to(device)
        self.model_ap.eval()
        self.inv_temperature = temperature
        
        if top_p < 0 or top_p > 1.0:
            raise ValueError(f"`top_p` has to be a float > 0 and < 1, but is {top_p}")
        if not isinstance(min_tokens_to_keep, int) or (min_tokens_to_keep < 1):
            raise ValueError(f"`min_tokens_to_keep` has to be a positive integer, but is {min_tokens_to_keep}")

        self.min_tokens_to_keep = min_tokens_to_keep
        self.top_p = float(top_p)
        self.top_k = top_k
        self.filter_value = filter_value
        self.use_alpha = use_alpha
        self.past_key_values = None
        self.past_input_len = None
        self.use_log_softmax = use_log_softmax

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.use_alpha:
            probs = scores.softmax(dim=-1)
            indices_to_remove = probs < torch.max(probs, dim=-1)[0][..., None] * self.top_p
            if self.top_k > 0:
                #sorted_logits, sorted_indices = torch.sort(scores, descending=False)
                #sorted_indices_to_remove = torch.zeros_like(sorted_logits)
                #sorted_indices_to_remove[..., :(sorted_indices_to_remove.size(-1)-self.top_k ) ] = 1
                #indices_to_remove_topk = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                indices_to_remove_topk = scores < torch.topk(scores, self.top_k)[0][..., -1, None]
                indices_to_remove = torch.logical_or(indices_to_remove,indices_to_remove_topk)
        else:
            sorted_logits, sorted_indices = torch.sort(scores, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

            # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
            # Keep at least min_tokens_to_keep
            sorted_indices_to_remove[..., -self.min_tokens_to_keep :] = 0
            if self.top_k > 0:
                sorted_indices_to_remove[..., :(sorted_indices_to_remove.size(-1)-self.top_k ) ] = 1
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)

        if self.past_key_values is None or self.past_input_len + 1 !=  input_ids.size(1):
            outputs_students = self.model_ap(input_ids.to(self.model_ap.device), return_dict = True)
        else:
            outputs_students = self.model_ap(input_ids[:,-1].unsqueeze(-1).to(self.model_ap.device), past_key_values = self.past_key_values, return_dict = True)

        self.past_input_len = input_ids.size(1)

        self.past_key_values = outputs_students.past_key_values
        scores_st = outputs_students.logits[:,-1,:]
        if self.use_log_softmax:
            scores = torch.log_softmax(scores,dim=-1)
            scores_st = torch.log_softmax(scores_st,dim=-1)
        
        scores[:,scores_st.size(-1):] = self.filter_value
        scores[:,:scores_st.size(-1)] = scores[:,:scores_st.size(-1)] - self.inv_temperature * scores_st
        
        # scatter sorted tensors to original indexing
        scores = scores.masked_fill(indices_to_remove, self.filter_value)
        return scores

