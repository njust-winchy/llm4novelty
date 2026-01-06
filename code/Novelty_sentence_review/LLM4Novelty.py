import json
from openai import OpenAI
from RAG import NoveltyRetriever
from tqdm import tqdm
import csv
import os
prompt = '''You are analyzing whether the following peer review sentence evaluates the **novelty** of a paper.

Sentence:
[REVIEW_SENTENCE]

Answer with only "Yes" or "No": Does this sentence evaluate novelty?'''


def build_prompt(sentence):
    prompt = f"""You are analyzing whether the following peer review sentence evaluates the **novelty** of a paper.

Sentence:

\"\"\"{sentence}\"\"\"

"""

    prompt += "\nAnswer with only 'Yes' or 'No': Does this sentence evaluate novelty?"
    return prompt


def build_rag_prompt(sentence, retrieved_examples):
    prompt = f"""You are analyzing whether the following peer review sentence evaluates the **novelty** of a paper.

Sentence:

\"\"\"{sentence}\"\"\"

Here are examples of novelty-related evaluation sentences:
"""
    for ex in retrieved_examples:
        prompt += f"- {ex}\n"

    prompt += "\nAnswer with only 'Yes' or 'No': Does this sentence evaluate novelty?"
    return prompt

def LLM4task(prompt, model):
    client = OpenAI(
        base_url='http://localhost:11434/v1/',
        # required but ignored
        api_key='ollama',
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ],
        model=model,
        temperature=0,
    )
    return chat_completion.choices[0].message.content


def gpt4task(prompt, model):
    client = OpenAI(
        base_url='https://api.kksj.org/v1',
        # required but ignored
        api_key='sk-d1ywZ0OuvHU3aHYzGFxe888FaIuJe9uYsfkJVKngnrd0tkQe',
    )
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'user',
                'content': prompt,

            }
        ],
        model=model,
        temperature=0,
    )
    return chat_completion.choices[0].message.content


def zeroshot_pipeline(review_sentences, model):

    results = []
    for sentence in tqdm(review_sentences):
        prompt = build_prompt(sentence)
        #answer = LLM4task(prompt, model=model)
        answer = gpt4task(prompt, model=model)

        # 规范输出为Yes/No，简单处理

        results.append((sentence, answer))
    return results


def rag_pipeline(review_sentences, novelty_kb, model, top_k=3):
    retriever = NoveltyRetriever(novelty_kb)
    retrieved_lists = retriever.batch_retrieve(review_sentences, top_k=top_k)

    results = []
    for sentence, retrieved in tqdm(zip(review_sentences, retrieved_lists)):
        prompt = build_rag_prompt(sentence, retrieved)
        #prompt = build_prompt(sentence)
        #answer = LLM4task(prompt, model=model)
        answer = gpt4task(prompt, model=model)

        # 规范输出为Yes/No，简单处理

        results.append((sentence, answer))
    return results

def save_results_to_csv(results, file_path):
    with open(file_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sentence", "Novelty_Evaluation"])  # header
        for sentence, label in results:
            writer.writerow([sentence, label])

with open("novelty_Yuan.json", encoding='utf-8') as f:
     database = json.load(f)
f.close()
novelty_knowledge_base = [d[0] for d in database]
with open("non_novelty_data.json", encoding='utf-8') as f:
    novelty_data = json.load(f)
f.close()
with open("novelty_data.json", encoding='utf-8') as f:
    non_novelty_data = json.load(f)
f.close()
test_data = []
for n in non_novelty_data:
    save_dic = {}
    save_dic['text'] = n[0]
    save_dic['label'] = 1
    test_data.append(save_dic)
for n in novelty_data:
    save_dic = {}
    save_dic['text'] = n['text']
    save_dic['label'] = 0
    test_data.append(save_dic)
#review_sentences = [d[0] for d in test_data]
review_sentences = [d['text'] for d in test_data]
#llm_list = ['gemma3:4b', 'deepseek-r1:8b', 'qwen3:8b']
#llm_list = ['deepseek-r1:8b', 'qwen3:8b']
llm_list = ['gpt-4o', 'gpt-4o-mini', 'gpt-5', 'gpt-5-mini', 'grok-4']

for llm in llm_list:
    prior = llm#.split(':')[0]
    #file_name = prior + "_novelty_no_rag_results.csv"
    file_name = 'nov_review/' + prior + "_non_novelty_rag_results.csv"
    file_path = 'nov_review/' + prior + "_non_novelty_no_rag_results.csv"
    if os.path.exists(file_name):
        print(llm)
        continue
    outputs = rag_pipeline(review_sentences, novelty_knowledge_base, llm, top_k=3)
    zero_outputs = zeroshot_pipeline(review_sentences, llm)
    save_results_to_csv(outputs, file_path=file_name)
    save_results_to_csv(zero_outputs, file_path=file_path)