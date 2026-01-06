from openai import OpenAI
from RAG import NoveltyRetriever
from tqdm import tqdm
import json
import pandas as pd
import ast
import os


def build_prompt(sentence):
    prompt = f"""You are analyzing whether the following peer review sentence evaluates the **novelty** of a paper.

Sentence:

\"\"\"{sentence}\"\"\"

"""

    prompt += "\nAnswer with only 'Yes' or 'No': Does this sentence evaluate novelty?"
    return prompt


# For paper novelty sentence
def gpt4task(prompt, model):
    client = OpenAI(
        base_url='https://api.kksj.org/v1',
        # required but ignored
        api_key='sk-yqwpNJSFxMqyEDjaJockxdXNQBkCMiihZdL2DPlCF9kCGWGY',
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


def build_novelty_prompt(sentence, context):
    return f"""
You are a senior academic reviewer familiar with how research papers describe their novelty.

You will be given a **target sentence** from the Introduction section of a research paper, along with its **context** (several surrounding sentences).

Your task is to determine whether the **target sentence** describes the novelty or contribution of **this paper**.
- Novelty typically includes introducing new methods, models, tasks, datasets, perspectives, or achieving new improvements or combinations that have not been explored before.
- Only consider whether the target sentence refers to **this paper’s novelty**—not the novelty of prior work.
- Use the provided context to assist your judgment, but base your decision specifically on the target sentence.

Answer with:
- Yes — if the target sentence describes the novelty of this paper.
- No — otherwise.

Context:
\"\"\"  
{context}  
\"\"\"

Target sentence:
\"\"\"  
{sentence}  
\"\"\"
"""

def add_context_and_prompts(sentences, window_size=2):
    """
    输入: sentences (list[str]) → 一个包含几十条句子的列表
    window_size: 表示前后取多少句
    输出: DataFrame，包含 sentence, context, prompt 三列
    """
    results = []
    n = len(sentences)

    for i, sentence in enumerate(sentences):
        if i == 0:  # 第一句
            context = sentence + " " + " ".join(sentences[i+1:i+1+window_size])
        elif i == n - 1:  # 最后一句
            context = " ".join(sentences[max(0, i-window_size):i]) + " " + sentence
        else:  # 中间句
            context = " ".join(
                sentences[max(0, i-window_size):i] + [sentence] + sentences[i+1:min(n, i+1+window_size)]
            )

        prompt = build_novelty_prompt(sentence, context)  # 复用你之前的 prompt 生成器
        results.append({
            "sentence": sentence,
            "context": context,
            "prompt": prompt
        })

    return pd.DataFrame(results)


#For review novelty sentence
def rev_pipeline(review_sentences, model):
    results = []
    for sentence in tqdm(review_sentences):
        prompt = build_prompt(sentence)
        answer = gpt4task(prompt, model=model)
        # 规范输出为Yes/No，简单处理

        results.append((sentence, answer))
    return results


def intro_pipeline(intro_sentences, model):
    sentence = list(intro_sentences['sentence'])
    prompt = list(intro_sentences['prompt'])
    results = []
    for i in tqdm(range(len(sentence))):
        answer = gpt4task(prompt[i], model=model)
        # 规范输出为Yes/No，简单处理

        results.append((sentence[i], answer))
    return results



# 1. 读取数据库
#with open("novelty_Yuan.json", encoding='utf-8') as f:
#    database = json.load(f)
#novelty_knowledge_base = [d[0] for d in database]
#retriever = NoveltyRetriever(novelty_knowledge_base)

# 2. 读取输入数据
df = pd.read_csv('emnlp_23_sen.csv')
introduction_sentence = df['introduction_sentence'].apply(ast.literal_eval)
review_sentence = df['review_sentence'].apply(ast.literal_eval)
reviews = list(df['reviews'])
# 3. 检查是否有中间结果
save_path = 'emnlp23_novelty_partial.csv'
if os.path.exists(save_path):
    df_partial = pd.read_csv(save_path)
    if 'review_novelty' in df_partial.columns:
        novelty_reviews = df_partial['review_novelty'].apply(ast.literal_eval).tolist()
        novelty_papers = df_partial['paper_novelty'].apply(ast.literal_eval).tolist()
        start_idx = len(novelty_reviews)
    else:
        novelty_reviews, novelty_papers = [], []
        start_idx = 0
else:
    novelty_reviews, novelty_papers = [], []
    start_idx = 0

list_length = len(review_sentence)

# 4. 循环处理
for i in range(start_idx, list_length):
    review_sentences = review_sentence[i]
    intro_sentences = introduction_sentence[i]
    result_list = add_context_and_prompts(intro_sentences, window_size=2)
    # 调用模型

    outputs = rev_pipeline(review_sentences, 'gpt-5')
    result = intro_pipeline(result_list, 'gpt-5')


    novelty_review = [idx for idx, ans in outputs if 'yes' in ans.strip().lower()]

    # 提取novelty索引
    novelty_paper = [idx for idx, ans in result if 'yes' in ans.strip().lower()]

    novelty_reviews.append(novelty_review)
    novelty_papers.append(novelty_paper)

    # 每100条保存一次
    if (i + 1) % 10 == 0 or (i + 1) == list_length:
        df_out = pd.DataFrame({
            'introduction_sentence': introduction_sentence[:i+1],
            'review_sentence': review_sentence[:i+1],
            'review_novelty': novelty_reviews,
            'paper_novelty': novelty_papers
        })
        df_out.to_csv(save_path, index=False)
        print(f"Progress saved at {i+1}/{list_length}")

# 5. 最后输出完整文件
df['review_novelty'] = novelty_reviews
df['paper_novelty'] = novelty_papers
df.to_csv('emnlp23_novelty.csv', index=False)
print("Processing complete, results saved to emnlp23_novelty.csv")