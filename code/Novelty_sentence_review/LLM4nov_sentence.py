import json
from openai import OpenAI
from tqdm import tqdm
import pandas as pd
import os


def build_prompt(sentence):
    prompt = f"""You are a senior academic reviewer familiar with how research papers describe their novelty.

    Given a sentence from the Introduction section of a paper, determine whether it describes the novelty or contribution of this paper. Novelty typically includes introducing new methods, models, tasks, datasets, perspectives, or achieving new improvements or combinations that have not been explored before.

    Only consider whether the sentence refers to this paper's novelty—not the novelty of prior work.

    Answer with:

    Yes — if the sentence describes the novelty of this paper

    No — otherwise

    Sentence:

    \"\"\"{sentence}\"\"\"

    """
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

def check_novelty(sentence, model="gemma3:4b"):
    """
    使用Prompt Chaining判断一个句子是否描述论文的新颖性。
    参数：
        sentence: str，学术论文引言中的一句话
        model: str，使用的模型名称（默认为gpt-4）
    返回：
        result: dict，包含step1、step2和最终判断
    """
    # Step 1 + Step 2: 获取主旨 + 判断是否描述本论文
    step1_prompt = f"""You are a careful academic language analyst.

Given the sentence below, perform two tasks:

1. Briefly summarize what the sentence is mainly saying.
2. Indicate whether it refers to this paper's own work, or to prior work or general background.

Sentence:
"{sentence}"

Answer format:
Main idea: ...
Refers to: [This paper / Previous work / General background]"""

    response1 = LLM4task(step1_prompt, model)
    answer1 = response1

    # 提取Step1输出内容
    lines = answer1.strip().split("\n")
    main_idea = ""
    refers_to = ""
    for line in lines:
        if line.lower().startswith("main idea:"):
            main_idea = line.split(":", 1)[1].strip()
        elif line.lower().startswith("refers to:"):
            refers_to = line.split(":", 1)[1].strip()

    # Step 3: 判断是否描述新颖性
    step3_prompt = f"""Given the following context:

Main idea: {main_idea}  
Refers to: {refers_to}

Does this sentence describe the novelty or original contribution of this paper?

Novelty may include introducing a new method, dataset, model, task, theory, or a new combination/improvement.  
Do not rely only on keywords like “new” or “novel” — base your judgment on meaning.

Answer with: Yes or No."""

    response2 = LLM4task(step3_prompt, model)
    final_judgment = response2.strip()

    return {
        "input_sentence": sentence,
        "step1_main_idea": main_idea,
        "step2_refers_to": refers_to,
        "novelty": final_judgment  # "Yes" or "No"
    }


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


def gpt4check_novelty(sentence, model):
    """
    使用Prompt Chaining判断一个句子是否描述论文的新颖性。
    参数：
        sentence: str，学术论文引言中的一句话
        model: str，使用的模型名称（默认为gpt-4）
    返回：
        result: dict，包含step1、step2和最终判断
    """
    # Step 1 + Step 2: 获取主旨 + 判断是否描述本论文
    step1_prompt = f"""You are a careful academic language analyst.

Given the sentence below, perform two tasks:

1. Briefly summarize what the sentence is mainly saying.
2. Indicate whether it refers to this paper's own work, or to prior work or general background.

Sentence:
"{sentence}"

Answer format:
Main idea: ...
Refers to: [This paper / Previous work / General background]"""

    response1 = gpt4task(step1_prompt, model)
    answer1 = response1

    # 提取Step1输出内容
    lines = answer1.strip().split("\n")
    main_idea = ""
    refers_to = ""
    for line in lines:
        if line.lower().startswith("main idea:"):
            main_idea = line.split(":", 1)[1].strip()
        elif line.lower().startswith("refers to:"):
            refers_to = line.split(":", 1)[1].strip()

    # Step 3: 判断是否描述新颖性
    step3_prompt = f"""Given the following context:

Main idea: {main_idea}  
Refers to: {refers_to}

Does this sentence describe the novelty or original contribution of this paper?

Novelty may include introducing a new method, dataset, model, task, theory, or a new combination/improvement.  
Do not rely only on keywords like “new” or “novel” — base your judgment on meaning.

Answer with: Yes or No."""

    response2 = gpt4task(step3_prompt, model)
    final_judgment = response2.strip()

    return {
        "input_sentence": sentence,
        "step1_main_idea": main_idea,
        "step2_refers_to": refers_to,
        "novelty": final_judgment  # "Yes" or "No"
    }

def build_context_prompt(sentence, context):
    prompt = f"""
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
    return prompt

#llm_list = ['gemma3:4b', 'deepseek-r1:8b', 'qwen3:8b']
llm_list = ['gemma3:12b', 'gpt-oss:20b']

#llm_list = ['gpt-4o-mini', 'gpt-4o-2024-08-06', 'gemma3:12b', 'deepseek-r1:8b', 'qwen3:8b']
#llm_list = ['deepseek-r1:14b', 'qwen3:14b', 'gpt-oss:latest']


#
for llm in llm_list:
    print(llm)
    df = pd.read_csv('labeling_data.csv', encoding='gbk')
    paper_id = list(df.paper_id)
    sentence = list(df.sentence)
    label = list(df.label)
    step_prediction = []
    prediction = []
    if 'gpt-4' in llm:
        file_name = llm + "4nov_sentence.csv"
        if os.path.exists(file_name):
            continue
        for sen in tqdm(sentence):
            # step prompt
            result = gpt4check_novelty(sen, llm)
            step_prediction.append(result['novelty'])
            # zero shot
            input_prompt = build_prompt(sen)
            result = gpt4task(input_prompt, llm)
            prediction.append(result)
    else:
        prior = llm.split(':')[0]
        file_name = prior + "4nov_sentence.csv"
        if os.path.exists(file_name):
            continue
        for sen in tqdm(sentence):
            # step prompt
            result = check_novelty(sen, llm)
            step_prediction.append(result['novelty'])
            # zero shot
            input_prompt = build_prompt(sen)
            result = LLM4task(input_prompt, llm)
            prediction.append(result)


    df['zero_result'] = prediction
    df['step_result'] = step_prediction
    df.to_csv(file_name, index=False)




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


def add_context_and_prompts(df, window_size=2):
    """
    df 必须有两列: paper_id, sentence
    window_size: 表示前后取多少句
    返回的 df 会多一列 context 和 prompt
    """
    results = []

    for paper_id, group in df.groupby("paper_id"):
        sentences = list(group["sentence"])
        label = list(group['label'])
        n = len(sentences)
        for i, sentence in enumerate(sentences):
            if i == 0:  # 第一个句子 → 只取后 window_size 句
                context = sentence + ' ' + " ".join(sentences[i + 1:i + 1 + window_size])
            elif i == n - 1:  # 最后一个句子 → 只取前 window_size 句
                context = " ".join(sentences[max(0, i - window_size):i]) + ' ' + sentence
            else:  # 中间句子 → 前 window_size 句 + 后 window_size 句
                context = " ".join(
                    sentences[max(0, i - window_size):i] + [sentence]
                    + sentences[i + 1:min(n, i + 1 + window_size)]
                )

            prompt = build_novelty_prompt(sentence, context)
            results.append({
                "paper_id": paper_id,
                "sentence": sentence,
                "label": label[i],
                "context": context,
                "prompt": prompt
            })

    return pd.DataFrame(results)



#llm_list = ['gpt-4o-mini', 'gpt-4o', 'gemma3:12b', 'deepseek-r1:8b', 'qwen3:8b']
llm_list = ['gemma3:12b', 'deepseek-r1:8b', 'qwen3:8b', 'gpt-oss:20b']

#llm_list = ['gpt-4o-mini', 'gpt-4o']

df = pd.read_csv('labeling_data.csv', encoding='gbk')
# 先生成上下文数据
result_df = add_context_and_prompts(df, window_size=2)

paper_id = list(result_df.paper_id)
sentence = list(result_df.sentence)
label = list(result_df.label)
prompt = list(result_df.prompt)

for llm in llm_list:
    print(llm)
    prior = llm.split(':')[0]
    file_name = prior + "_context4nov_sentence.csv"


    context_prediction = []
    batch_size = 100

    # 如果之前已经有部分结果，就从文件里加载已完成的数量，接着跑
    start_idx = 0
    if os.path.exists(file_name):
        done_df = pd.read_csv(file_name)
        start_idx = len(done_df)
        print(f"Resuming {llm} from index {start_idx}")

    for idx, prom in enumerate(tqdm(prompt[start_idx:], initial=start_idx, total=len(prompt))):
        global_idx = start_idx + idx
        if 'gpt' in llm:
            result = gpt4task(prom, llm)
        else:
            result = LLM4task(prom, llm)

        context_prediction.append(result)

        # 每 batch_size 条保存一次
        if (global_idx + 1) % batch_size == 0 or (global_idx + 1) == len(prompt):
            batch_start = global_idx + 1 - len(context_prediction)  # 当前 batch 的起始索引
            batch_end = global_idx + 1  # 当前 batch 的结束索引

            batch_df = pd.DataFrame({
                'paper_id': paper_id[batch_start:batch_end],
                'sentence': sentence[batch_start:batch_end],
                'label': label[batch_start:batch_end],
                'prompt': prompt[batch_start:batch_end],
                'result': context_prediction
            })

            # 追加写入
            header = not os.path.exists(file_name) or (start_idx == 0 and batch_start == 0)
            batch_df.to_csv(file_name, index=False, mode='a', header=header)

            context_prediction = []  # 清空，避免内存堆积

    print(f"Saved results for {llm} to {file_name}")




# 定义 few-shot prompt 模板
novelty_prompt_template = """
You are a senior academic reviewer familiar with how research papers describe their novelty.
Your task is to determine whether a given sentence from the Introduction section describes the novelty or contribution of THIS paper.

Definition of Novelty:
Novelty includes introducing new methods, models, tasks, datasets, perspectives, or achieving new improvements or combinations that have not been explored before.
Do NOT mark sentences as novel if they only discuss prior work’s novelty or general background.

Answer strictly with:
- Yes — if the sentence describes this paper’s novelty.
- No — otherwise.

Examples:

# Simple clear cases
Sentence: "In this paper, we propose a novel transformer-based architecture that integrates syntactic information into language modeling."
Answer: Yes

Sentence: "Previous studies have explored the use of graph neural networks for text classification."
Answer: No

Sentence: "Our method achieves state-of-the-art performance on the benchmark dataset, surpassing previous approaches by a significant margin."
Answer: Yes

Sentence: "There has been growing interest in applying deep learning methods to natural language processing tasks."
Answer: No

Sentence: "We introduce a new large-scale dataset for evaluating dialogue systems in multi-turn conversations."
Answer: Yes

Sentence: "Several authors have studied this problem from both theoretical and empirical perspectives."
Answer: No

# Borderline cases
Sentence: "We show that applying existing attention mechanisms to longer documents leads to performance improvements."
Answer: No

Sentence: "This paper is the first to conduct a comprehensive comparison of previously proposed algorithms for neural parsing."
Answer: Yes

Sentence: "Our experiments confirm prior findings that model size correlates strongly with accuracy."
Answer: No

Sentence: "We extend the existing framework by incorporating domain adaptation techniques, enabling application to low-resource languages."
Answer: Yes

Sentence: "The results indicate that combining convolutional and recurrent networks is effective for text classification."
Answer: No

Sentence: "Unlike previous approaches, our model integrates symbolic reasoning with neural networks in a unified framework."
Answer: Yes

Sentence: "To the best of our knowledge, this is the first work that studies multilingual dialogue summarization."
Answer: Yes

Sentence: "Many prior works have focused on English datasets, whereas our study evaluates methods on multilingual benchmarks."
Answer: Yes

Sentence: "Our study builds upon Smith et al. (2020) and does not introduce new methods."
Answer: No

Now classify the following sentence:

Sentence: \"\"\"{sentence}\"\"\"
"""


def build_prompts(sentences):
    """
    输入: sentences (list[str]) - 句子列表
    输出: prompts (list[str]) - 带 few-shot 示例的完整 prompt 列表
    """
    prompts = [novelty_prompt_template.format(sentence=s) for s in sentences]
    return prompts


# few shot

#llm_list = ['gpt-4o-mini', 'gpt-4o', 'gemma3:12b', 'deepseek-r1:8b', 'qwen3:8b']
llm_list = ['gemma3:12b', 'deepseek-r1:8b', 'qwen3:8b', 'gpt-oss:20b']

#llm_list = ['gpt-4o-mini', 'gpt-4o']
df = pd.read_csv('labeling_data.csv', encoding='gbk')
# 先生成上下文数据

paper_id = list(df.paper_id)
sentence = list(df.sentence)
label = list(df.label)
prompts = build_prompts(sentence)
print('few shot')
for llm in llm_list:
    print(llm)
    prior = llm.split(':')[0]
    file_name = prior + "_few4nov_sentence.csv"


    context_prediction = []
    batch_size = 100

    # 如果之前已经有部分结果，就从文件里加载已完成的数量，接着跑
    start_idx = 0
    if os.path.exists(file_name):
        done_df = pd.read_csv(file_name)
        start_idx = len(done_df)
        print(f"Resuming {llm} from index {start_idx}")

    for idx, prom in enumerate(tqdm(prompts[start_idx:], initial=start_idx, total=len(prompts))):
        global_idx = start_idx + idx
        if 'gpt' in llm:
            result = gpt4task(prom, llm)
        else:
            result = LLM4task(prom, llm)

        context_prediction.append(result)

        # 每 batch_size 条保存一次
        if (global_idx + 1) % batch_size == 0 or (global_idx + 1) == len(prompt):
            batch_start = global_idx + 1 - len(context_prediction)  # 当前 batch 的起始索引
            batch_end = global_idx + 1  # 当前 batch 的结束索引

            batch_df = pd.DataFrame({
                'paper_id': paper_id[batch_start:batch_end],
                'sentence': sentence[batch_start:batch_end],
                'label': label[batch_start:batch_end],
                'prompt': prompt[batch_start:batch_end],
                'result': context_prediction
            })

            # 追加写入
            header = not os.path.exists(file_name) or (start_idx == 0 and batch_start == 0)
            batch_df.to_csv(file_name, index=False, mode='a', header=header)

            context_prediction = []  # 清空，避免内存堆积

    print(f"Saved results for {llm} to {file_name}")
