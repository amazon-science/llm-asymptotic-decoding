# Extrapolating an Infinite LLM♾🤖

## Introduction

Assuming you have a series of LLMs with different sizes that are trained on the same data and you want to increase the factuality and diversity of the text sampled from your largest LLM. Then, consider to use our proposed REAL sampling and/or APD sampling. In FactualityPrompt, we show that APD + REAL sampling outperforms 13 state-of-the-art sampling methods. Our baselines include typical ([Meister et al., 2022](https://arxiv.org/abs/2202.00666)), eta ([Hewitt et al., 2022](https://arxiv.org/pdf/2210.15191)), EDT ([Zhang et al., 2024](https://arxiv.org/abs/2403.14541)), adaptive ([Zhu et al., 2024](https://arxiv.org/abs/2402.18223)), microstat ([Basu et al., 2021](https://arxiv.org/abs/2007.14966)), EAD w/o ELI ([Arora et al., 2023](https://arxiv.org/abs/2302.06784)) factual ([Lee et al., 2022](https://arxiv.org/abs/2206.04624)) top-p ([Holtzman et al., 2020](https://arxiv.org/pdf/1904.09751)), top-k ([Fan et al., 2018](https://arxiv.org/pdf/1805.04833)), and temperature sampling; contrastive search ([Su and Collier, 2022](https://arxiv.org/pdf/2210.14140)) , contrastive decoding (CD) ([Li et al., 2022](https://arxiv.org/pdf/2210.15097)), and DoLa ([Chuang et al., 2023](https://arxiv.org/pdf/2309.03883)). We show that APD + REAL sampling makes Pythia 6.9B simultaneously achieve the factuality of greedy sampling and diversity of top-p with p=0.5.

<p align="center"><img src="https://github.com/amazon-science/llm-asymptotic-decoding/blob/main/AP_sampling/imgs/Results.png?raw=true" width="540" height="195"></p>

## Usage

To run our code, please follow the instructions in the README.md of each folder.

We first write the REAL sampling code in the REAL_sampling folder and revise the code for APD sampling in the AP_sampling folder. As a result, AP_sampling also includes the inference code of REAL sampling. We also slightly modify the code of FactualityPrompt (https://github.com/nayeon7lee/FactualityPrompt) to make it easier to run.

## Computational Resources

Our code assumes that your machine has 8 GPUs and each GPU has 32G memory. If you have less GPU or your GPU has less memory, you can try to reduce your generation model sizes.

## Questions

If you have any questions or find any bugs, please send an email to Haw-Shiuan Chang (hschang@cs.umass.edu).

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This library is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International](https://creativecommons.org/licenses/by-nc/4.0/) License.

## Citation

If you use our code for THF model or REAL sampling in your work, consider to cite https://arxiv.org/abs/2406.07735 .
```
@misc{chang2024realsamplingboostingfactuality,
      title={REAL Sampling: Boosting Factuality and Diversity of Open-Ended Generation via Asymptotic Entropy},
      author={Haw-Shiuan Chang and Nanyun Peng and Mohit Bansal and Anil Ramakrishna and Tagyoung Chung},
      year={2024},
      eprint={2406.07735},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.07735},
}
```

If you use our code for APD sampling in your work, consider to cite https://arxiv.org/abs/2411.01610 (see the example reference and bib information below).
```
@inproceedings{chang2024explaining,
  title={Explaining and Improving Contrastive Decoding by Extrapolating the Probabilities of a Huge and Hypothetical LM},
  author={Chang, Haw-Shiuan and Peng, Nanyun and Bansal, Mohit and Ramakrishna, Anil and Chung, Tagyoung},
  booktitle={Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing},
  year={2024},
}
```

If you use FactualityPrompt, cite their paper (https://arxiv.org/abs/2206.04624).
