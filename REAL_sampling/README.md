# REAL Sampling: Boosting Factuality and Diversity of Open-Ended Generation by Extrapolating the Entropy of an Infinitely Large LM

<p align="center"><img src="https://github.com/amazon-science/llm-asymptotic-decoding/blob/main/REAL_sampling/imgs/REAL_second_figure.png?raw=true" width="700" height="150"></p>

## Introduction

REAL (**R**esidual **E**ntropy from **A**symptotic **L**ine) sampling is a decoding method that achieves improved factuality and diversity over nucleus sampling by predicting an adaptive threshold of p. Specifically, REAL sampling predicts the step-wise likelihood of an LLM to hallucinate, and lowers the p threshold when an LLM is likely to hallucinate. Otherwise, REAL sampling increases the p threshold to boost the diversity. To predict the step-wise hallucination likelihood without supervision, we construct a Token-level Hallucination Forecasting (THF) model to predict the asymptotic entropy (i.e., inherent uncertainty) of the next token by extrapolating the next-token entropies from a series of LLMs with different sizes. If a LLM's entropy is higher than the asymptotic entropy (i.e., the LLM is more uncertain than it should be), the THF model predicts a high hallucination hazard, which leads to a lower p threshold in REAL sampling. In the FactualityPrompts benchmark, we demonstrate that REAL sampling based on a 70M THF model can substantially improve the factuality and diversity of 7B LLMs simultaneously, judged by both retrieval-based metrics and human evaluation. 

## Computational Environment

You can reproduce our python enviroment using
```
conda create --name <env> --file requirement.txt
```
## How to run REAL sampling

To learn how to use REAL sampling in huggingface, please see the following example code 

```
./src/example.py
```

### Run FactualityPrompts

To evaluate the generation results, first follow ../FactualityPrompt/README.md to download the data, change ../FactualityPrompt/src/const.py and run the following script.

If you have >7 GPUs in your machine, you can just run the following file to generate the contiunations.
```
./bin/continue_wiki_prompt_loop.sh
```

To evaluate the generation results, first follow ../FactualityPrompt/README.md to download the data, change ./FactualityPrompt/src/const.py and run the following script.
```
../FactualityPrompt/bin/eval_loop.sh
```


## How to Train THF


Put your text file into "data/raw/". 

Change the INPUT_FILE in bin/train_THF_model.sh and run it (Assuming you have more than 7 GPUs in your machine).


## How to use THF to produce unsupervised features for hallucination detection tasks

Please check src/process_hallucination_dataset/get_entropy_all.py and analyze_datasets/feature_clf_all.py



