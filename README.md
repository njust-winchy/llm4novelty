# NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment

## Overview

**Dataset and source code for paper "NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment".**

The pipeline for constructing NovBench as follow.<br>
<img width="2700" height="851" alt="Figure1" src="https://github.com/user-attachments/assets/71399c44-0998-4029-a430-08d77bb1b743" />


## Dataset
The raw data and calculated data can be obtained from here (We will provide it after the peer review is completed).<br>
## Directory structure

<pre>
NovBench                                    Root directory
├── code                                    Code for processing and experiment
│   ├── data_process                        Code for data processing
│   ├── LLM_novelty_generate                Code for novelty generate experiment
│   │   ├── API_model_zero.py                  Code for 
│   ├── Novelty_sentence_review             Code for experiment
│   │   ├── LLM4nov_sentence.py                Code for
│   ├── comment_evaluate.py                    Code for evaluate the novelty assessment generation
│   ├── human_sample.py                        Code for human evaluation sample selection
├── data                                    Dataset for experiment
│
└── README.md

</pre>


## Dependency packages
System environment is set up according to the following configuration:
- transformers==4.16.2
- nltk==3.6.7
- matplotlib==3.5.1
- scikit-learn==1.1.3
- pytorch==2.0.1
- tqdm==4.65.0
- numpy==1.24.1
- pandas==2.2.3
- openai==1.53.0
- sentence-transformers==3.4.1
- ai_researcher
## Acknowledgement

The datasets we use come from Dycke et al.（2023）(https://github.com/UKPLab/nlpeer)

>Nils Dycke, Ilia Kuznetsov, and Iryna Gurevych. 2023. NLPeer: A Unified Resource for the Computational Study of Peer Review. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5049–5073, Toronto, Canada. Association for Computational Linguistics.<br>

## Citation
Please cite the following paper if you use this code and dataset in your work.




