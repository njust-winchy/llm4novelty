# NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment
<img width="2700" height="851" alt="Figure1" src="https://github.com/user-attachments/assets/71399c44-0998-4029-a430-08d77bb1b743" />
This is the official repository for the dataset and code of the paper: "NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment", accepted at ACL 2026 (Findings).<br>

## Overview
- We introducing a structured benchmark for novelty evaluation. This resource, which incorporates the' textual evaluations of the novelty dimension of the reviewers alongside the novelty descriptions of the paper introductions, constitutes a critical resource for future research. <br>
- We propose a four-dimensional, interpretable, and semantics-aware framework for evaluating LLM-generated novelty review text. This metric suite surpasses the limitations of traditional lexical overlap metrics (e.g., ROUGE/BLEU) and more effectively captures the quality of "novelty" evaluation in free-form text. <br>
- We conduct systematic evaluation and benchmarking of current general LLMs (e.g., GPT-5, Gemini-2.5-flash) and specialized LLMs on the novelty evaluation task. Based on this benchmark, we further analyze the gap between LLM-generated evaluations and human judgments, providing an in-depth understanding of these models' advantages and limitations in identifying and articulating novelty. <br>
- We conduct a comprehensive empirical analysis that identifies key factors for generating high-quality, highly interpretable novelty review text, and reveals important behavioral patterns of LLMs in novelty evaluation, thus directing the future development of more reliable AI-assisted peer review.

## Dataset
The raw data and calculated data can be obtained from here (https://drive.google.com/drive/folders/1VZTCcUngoBa4jC3skKgbcefyotuHY9Zc?usp=drive_link).<br>
## Directory structure

<pre>
NovBench                                      Root directory
├── code                                      Code for processing and experiment
│   ├── data_process                          Code for data processing
│   │   ├── COLING_Paper_process.py           Code for processing COLING data
│   │   ├── data2exps.py                      Code for processing final data
│   │   ├── EMNLP_data_process.py             Code for processing EMNLP data to get introduction
│   │   ├── EMNLP_introduction_data_save.py   Code for spliting EMNLP' introduction and review text to sentences
│   │   ├── output_format.py                  Code for processing EMNLP data to format based on sentiment distribution
│   │   ├── rev_inrto_nov_identification.py   Code for processing EMNLP data to obtain novelty descriptions in the introduction and novelty evaluation in the review text
│   ├── LLM_novelty_generate                  Code for novelty generate experiment
│   │   ├── API_model_zero.py                 Code for Closed-source LLMs to generate novelty assessment
│   │   ├── API_model_few.py                  Code for Closed-source LLMs to generate novelty assessment
│   │   ├── API_model_rag.py                  Code for Closed-source LLMs to generate novelty assessment
│   │   ├── CycleReviewer_zero.py             Code for CycleReviewer to generate novelty assessment
│   │   ├── CycleReviewer_few.py              Code for CycleReviewer LLMs to generate novelty assessment
│   │   ├── CycleReviewer_rag.py              Code for CycleReviewer LLMs to generate novelty assessment
│   │   ├── Deep_review_zero.py               Code for Deepreviewer to generate novelty assessment
│   │   ├── Deep_review_few.py                Code for Deepreviewer LLMs to generate novelty assessment
│   │   ├── Deep_review_rag.py                Code for Deepreviewer LLMs to generate novelty assessment
│   │   ├── SEA_zero.py                       Code for SEA-S and SEA-E to generate novelty assessment
│   │   ├── SEA_few.py                        Code for SEA-S and SEA-E to generate novelty assessment
│   │   ├── SEA_rag.py                        Code for SEA-S and SEA-E to generate novelty assessment
│   │   ├── zero_shot.py                      Code for Open-source LLMs to generate novelty assessment
│   │   ├── few_shot.py                       Code for Open-source LLMs to generate novelty assessment
│   │   ├── RAG_prompt.py                     Code for Open-source LLMs to generate novelty assessment
│   ├── Novelty_sentence_review               Code for experiment
│   │   ├── LLM4nov_sentence.py               Testing LLMs for identifying novelty descriptions in the introduction
│   │   ├── LLM4Novelty.py                    Testing LLM for identifying novelty evaluation in the review text
│   │   ├── RAG.py                            Code for rag prompt in LLM4Novelty
│   │   ├── RAG_retrieve.py                   Code for rag prompt in LLM4Novelty
│   ├── comment_evaluate.py                   Code for evaluate the novelty assessment generation
│   ├── human_sample.py                       Code for human evaluation sample selection
├── data                                      Dataset for experiment
│   ├── COLING_labeled_data.csv               Novelty description in the CONLING's introduction annotated by humans
│   ├── Final_data.json                       Dataset for LLMs generate novelty assessment
│   ├── non_novelty_data.json                 Dataset for testing LLM for identifying novelty evaluation in the review text
│   ├── novelty_data.json                     Dataset for testing LLM for identifying novelty evaluation in the review text
│   ├── novelty_Yuan.json                     Data for RAG in LLM4Novelty
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
- scientific-information-change
## Acknowledgement

The datasets we used come from Yuan et al.（2022）(https://github.com/neulab/ReviewAdvisor), Dycke et al.（2023）(https://github.com/UKPLab/nlpeer) and Lu et al. (2025) (https://github.com/UKPLab/emnlp2025-aspects-in-reviews)

>Yuan, W., Liu, P., & Neubig, G. (2022). Can we automate scientific reviewing?. Journal of Artificial Intelligence Research, 75, 171-212.<br>

>Nils Dycke, Ilia Kuznetsov, and Iryna Gurevych. 2023. NLPeer: A Unified Resource for the Computational Study of Peer Review. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 5049–5073, Toronto, Canada. Association for Computational Linguistics.<br>

>Sheng Lu, Ilia Kuznetsov, and Iryna Gurevych. 2025. Identifying Aspects in Peer Reviews. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 6145–6167, Suzhou, China. Association for Computational Linguistics.<br>
## Citation
Please cite the following paper if you use this code and dataset in your work.





