# Explaining and Improving Contrastive Decoding by Extrapolating the Probabilities of a Huge and Hypothetical LM

<p align="center"><img src="https://github.com/amazon-science/llm-asymptotic-decoding/blob/main/AP_sampling/imgs/APD_first_figure.png?raw=true" width="540" height="477"></p>

## Introduction

To overcome the limitation of contrastive decoding (CD), we propose a new unsupervised decoding method called **A**symptotic **P**robability **D**ecoding (APD). APD explicitly extrapolates the probability curves from the LMs of different sizes to infer the asymptotic probabilities from an infinitely large LM without inducing more inference costs than CD. In FactualityPrompts, an open-ended text generation benchmark, sampling using APD significantly boosts factuality in comparison to the CD sampling and its variants, and achieves state-of-the-art results for Pythia 6.9B and OPT 6.7B. Furthermore, in five commonsense QA datasets, APD is often significantly better than CD and achieves a similar effect of using a larger LLM. For example, the perplexity of APD on top of Pythia 6.9B is even lower than the perplexity of Pythia 12B in CommonsenseQA and LAMBADA.


## Computational Environment

You can reproduce our python enviroment using
```
conda create --name <env> --file requirement.txt
```
Most of the codes could also be run using older versions (e.g., the version in the REAL_sampling/requirement.txt) of huggingface except for running the Qwen LLM

## How to run APD

To learn how to use APD and/or REAL sampling in huggingface, please see the following example code

```
./src/example_APD_REAL.py
```

### Run FactualityPrompts

To evaluate the generation results, first follow ../FactualityPrompt/README.md to download the data, change ../FactualityPrompt/src/const.py and run the following script.

If you have >7 GPUs in your machine, you can just run the following file to generate the contiunations.
```
./bin/continue_wiki_prompt_loop_eval.sh
```

### Run Question Answering Datasets

Step 1: Run the dataset download codes at src/QA/dataset_preparation (For ARC, we concatenate the easy and challenge json output).

Step 2: Test APD models on the datasets. For datasets with only positive answers (e.g., LAMBADA, SQuAD, and MultiRC), use src/QA/dataset_preparation/test_squad_dataset.py. For the datasets with negative answers (e.g., QASC, ARC, SocialIQA, and CommonsenceQA), use src/QA/dataset_preparation/test_neg_dataset.py . If you want to also test the APD on the fly baseline, use test_squad_dataset_online_all.py and test_neg_dataset_online_all.py instead. Remember to change the paths in each file accordingly.

Step 3: Run analyze_results.py or analyze_results_online_all.py to collect results. For datasets that have negative answers and accuracy metrics, set have_acc to be 1.


## How to Train ALM' (in order to use APD)

Put your text file into "data/raw/".

Change the INPUT_FILE, data_folder_name, and OUTPUT_MODEL_FOLDER in bin/finetune_ALM.sh and run it (Assuming you have more than 7 GPUs in your machine).

Notice that our current implementation will first save lots of probabilities and logits from the top tokens of various LLMs into a cache, which will take lot of disk space. 
And we also need lots of CPU memory to load these probabilities. For example, after process ~270M Wikipedia text using 5 OPT models, we store 70G tensor and 52G dataset cache and our server has around 750G cpu memory. 
