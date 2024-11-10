#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for causal language modeling (GPT, GPT-2, CTRL, ...) on a text file or a dataset.

Here is the full list of checkpoints on the hub that can be fine-tuned by this script:
https://huggingface.co/models?filter=text-generation
"""
# You can also adapt this script on your own causal language modeling task. Pointers for this are left as comments.

import logging
import math
import os
import sys
import warnings
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional

import datasets
import evaluate
import torch
from datasets import Dataset, load_from_disk

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_CAUSAL_LM_MAPPING,
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    is_torch_tpu_available,
    set_seed,
)
#from transformers.testing_utils import CaptureLogger
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version
from model import GPTNeoXForEntropyClassification, GPTNeoXForEXPEntropyClassification, GPTNeoXForScaledEntropyClassification
from model import OPTForEntropyClassification, GPT2ForEntropyClassification

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
#check_min_version("4.32.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/language-modeling/requirements.txt")

logger = logging.getLogger(__name__)


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default='EleutherAI/pythia-410m-deduped',
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    model_type: Optional[str] = field(
        default=None,
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    #tokenizer_name: Optional[str] = field(
    #    default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    #)
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    #use_fast_tokenizer: bool = field(
    #    default=True,
    #    metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    #)
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    #token: str = field(
    #    default=None,
    #    metadata={
    #        "help": (
    #            "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
    #            "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
    #        )
    #    },
    #)
    use_auth_token: bool = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token`."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. This option"
                "should only be set to `True` for repositories you trust and in which you have read the code, as it will"
                "execute code present on the Hub on your local machine."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    def __post_init__(self):
        if self.config_overrides is not None and (self.config_name is not None or self.model_name_or_path is not None):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_text_file: Optional[str] = field(default="data/processed/openwebtext17-18_1e6_Pythia/tensors_all/train.pt", metadata={"help": "The input training data file (a text file)."})
    validation_text_file: Optional[str] = field(default="data/processed/openwebtext17-18_1e6_Pythia/tensors_all/val_org.pt",metadata={"help": "An input evaluation data file (a text file)."},)
    train_label_folder: Optional[str] = field(default="data/processed/openwebtext17-18_1e6_Pythia/entropy_tensor/train", metadata={"help": "The input training label folder."})
    validation_label_folder: Optional[str] = field(default="data/processed/openwebtext17-18_1e6_Pythia/entropy_tensor/val",metadata={"help": "An input evaluation label file."},)
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    #block_size: Optional[int] = field(
    #    default=None,
    #    metadata={
    #        "help": (
    #            "Optional input sequence length after tokenization. "
    #            "The training dataset will be truncated in block of this size for training. "
    #            "Default to the model max input length for single sentence inputs (take into account special tokens)."
    #        )
    #    },
    #)
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    #validation_split_percentage: Optional[int] = field(
    #    default=5,
    #    metadata={
    #        "help": "The percentage of the train set used as validation set in case there's no validation split"
    #    },
    #)
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    #keep_linebreaks: bool = field(
    #    default=True, metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    #)

    def __post_init__(self):
        if self.streaming:
            require_version("datasets>=2.0.0", "The streaming feature requires `datasets>=2.0.0`")

        #if self.dataset_name is None and self.train_file is None and self.validation_file is None:
        #    raise ValueError("Need either a dataset name or a training/validation file.")
        #else:
        #    if self.train_file is not None:
        #        extension = self.train_file.split(".")[-1]
        #        assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, a json or a txt file."
        #    if self.validation_file is not None:
        #        extension = self.validation_file.split(".")[-1]
        #        assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, a json or a txt file."


#file_suffix = '_bptt_256.pt'
file_suffix = '_bptt_1024.pt'

def load_label_data(label_folder, file_prefix, model_names):
    label_data_arr = []
    for i in range(len(model_names)):
        file_name = label_folder + '/' + file_prefix + model_names[i] + file_suffix
        print('loading ', file_name)
        with open(file_name, 'rb') as f_in:
            entropy_tensor = torch.load(f_in, map_location='cpu')
            label_data_arr.append(entropy_tensor)
    
    entropy_decay_tesnor = torch.stack(label_data_arr, dim=-1)
    del label_data_arr
    return entropy_decay_tesnor


def compose_dataset(text_file, label_folder, file_prefix, model_names):
    print('loading', text_file)
    with open(text_file,'rb') as f_in:
        w_ind_tensor = torch.load(f_in, map_location='cpu')
    entropy_decay_tesnor = load_label_data(label_folder, file_prefix, model_names)
    num_seq, bptt, num_models = entropy_decay_tesnor.size()
    
    cut_tok_num = w_ind_tensor.numel() - num_seq*bptt
    assert cut_tok_num < bptt, print('cut_tok_num', cut_tok_num)
    w_ind_tensor = w_ind_tensor[:num_seq*bptt].view(num_seq, bptt)
    output_dict = {"input_ids": w_ind_tensor.tolist(), "labels": entropy_decay_tesnor.tolist()}
    assert len(output_dict["input_ids"]) == len(output_dict["labels"])
    return Dataset.from_dict(output_dict)


def extract_param(raw_name, param_prefix, param_suffix):
	prefix_start = raw_name.index(param_prefix)
	start_idx = prefix_start+len(param_prefix)
	if param_suffix is None:
		return int(raw_name[start_idx:])
	end_idx = raw_name[start_idx:].index(param_suffix)
	#print(start_idx, end_idx)
	return int(raw_name[start_idx:start_idx+end_idx])


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn("The `use_auth_token` argument is deprecated and will be removed in v4.34.", FutureWarning)
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    if 'pythia' in model_args.model_name_or_path or 'Pythia' in data_args.train_label_folder:
        model_prefix = 'EleutherAI_pythia-'
        log_model_size = [16.75548316, 18.25882042, 19.52696825, 20.50726726, 20.91273067, 21.64659275, 22.58644061]
        model_names = ["70m-deduped", "160m-deduped", "410m-deduped", "1b-deduped", "1.4b-deduped", "2.8b-deduped", "6.9b-deduped"]
    elif 'opt' in model_args.model_name_or_path or 'OPT' in data_args.train_label_folder:
        model_prefix = 'facebook_opt-'
        model_names = ["125m", "350m", "1.3b", "2.7b", "6.7b"]
        log_model_size = [18.2771614, 19.5373201, 20.9161984, 21.6486751, 22.5877428]

    w_idx_model_name = model_prefix + model_names[-1]
    cache_file_name = 'cache_logit_ext_new_' + w_idx_model_name + file_suffix.replace('.pt','.hf')
    file_prefix = 'ent_' + model_prefix
    #file_prefix = 'per_' + model_prefix
    #file_prefix = 'ent_EleutherAI_pythia-'
    #file_prefix = 'per_EleutherAI_pythia-'



    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_clm", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    
    cache_train_file_path = data_args.train_label_folder + '/' + cache_file_name
    cache_eval_file_path = data_args.validation_label_folder + '/' + cache_file_name

    if os.path.exists(cache_train_file_path):
        print('load cache: ', cache_train_file_path)
        print('load cache: ', cache_eval_file_path)
        train_dataset = load_from_disk(cache_train_file_path)
        eval_dataset = load_from_disk(cache_eval_file_path)
    else:
        train_dataset = compose_dataset(data_args.train_text_file, data_args.train_label_folder, file_prefix, model_names)
        eval_dataset = compose_dataset(data_args.validation_text_file, data_args.validation_label_folder, file_prefix, model_names)
        train_dataset.save_to_disk(cache_train_file_path)
        eval_dataset.save_to_disk(cache_eval_file_path)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        #'classifier_dropout': 0.1,
        #"token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
        #'log_model_size': log_model_size
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    #config.log_model_size = log_model_size
    config.classifier_dropout = 0.1

    if model_args.model_name_or_path:
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)
        )
        if '_exp_decay_' in training_args.output_dir or '_logistic_decay_' in training_args.output_dir:
            if '_exp_decay_' in training_args.output_dir:
                print('use exponential decay')
                decay_function='exp'
            elif '_logistic_decay_' in training_args.output_dir:
                print('use logistic decay')
                decay_function='logistic'

            model = GPTNeoXForEXPEntropyClassification.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                log_model_size=log_model_size,
                decay_function=decay_function
            )
        else:

            if 'scaled_a' in training_args.output_dir:
                print('predict scaled s')
                model_class = GPTNeoXForScaledEntropyClassification
            else:
                if 'OPT_' in training_args.output_dir:
                    model_class = OPTForEntropyClassification
                elif 'GPT2_' in training_args.output_dir:
                    model_class = GPT2ForEntropyClassification
                else:
                    model_class = GPTNeoXForEntropyClassification

            poly_degree = extract_param(training_args.output_dir, '_a', '_e')
            #use_a4 = False
            #use_a56 = False
            #use_a10 = False
            #if '_a4_' in training_args.output_dir:
            #    use_a4 = True
            #if '_a6_' in training_args.output_dir:
            #    use_a4 = True
            #    use_a56 = True
            #if '_a10_' in training_args.output_dir:
            #    use_a4 = True
            #    use_a56 = True
            #    use_a10 = True
            #print(use_a4, use_a56, use_a10)

            model = model_class.from_pretrained(
                model_args.model_name_or_path,
                from_tf=bool(".ckpt" in model_args.model_name_or_path),
                config=config,
                cache_dir=model_args.cache_dir,
                revision=model_args.model_revision,
                trust_remote_code=model_args.trust_remote_code,
                torch_dtype=torch_dtype,
                low_cpu_mem_usage=model_args.low_cpu_mem_usage,
                log_model_size=log_model_size,
                poly_degree=poly_degree
                #use_a4=use_a4, use_a56=use_a56, use_a10=use_a10
           )
        model.classifier.weight.data[:] = 0
        model.classifier.bias.data[:] = 0
    #else:
    #    model = AutoModelForCausalLM.from_config(config, trust_remote_code=model_args.trust_remote_code)
    #    n_params = sum({p.data_ptr(): p.numel() for p in model.parameters()}.values())
    #    logger.info(f"Training new model from scratch - Total size={n_params/2**20:.2f}M params")

    # since this will be pickled to avoid _LazyModule error in Hasher force logger loading before tokenize_function
    tok_logger = transformers.utils.logging.get_logger("transformers.tokenization_utils_base")


    if training_args.do_train:
        #if "train" not in tokenized_datasets:
        #    raise ValueError("--do_train requires a train dataset")
        #train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        #if "validation" not in tokenized_datasets:
        #    raise ValueError("--do_eval requires a validation dataset")
        #eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))

        #def preprocess_logits_for_metrics(logits, labels):
        #    if isinstance(logits, tuple):
        #        # Depending on the model and config, logits may contain extra tensors,
        #        # like past_key_values, but logits always come first
        #        logits = logits[0]
        #    return logits.argmax(dim=-1)

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        #tokenizer=tokenizer,
        # Data collator will default to DataCollatorWithPadding, so we change it.
        data_collator=default_data_collator,
        compute_metrics=None,
        #preprocess_logits_for_metrics=preprocess_logits_for_metrics
        #if training_args.do_eval and not is_torch_tpu_available()
        #else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        #try:
        #    perplexity = math.exp(metrics["eval_loss"])
        #except OverflowError:
        #    perplexity = float("inf")
        #metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)



def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
