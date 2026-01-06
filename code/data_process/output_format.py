from openai import OpenAI
import os
import pandas as pd
from tqdm import tqdm
import ast
def build_prompt(reviews):
    prompt = f"""
    You are a review analysis assistant. I am providing a set of reviewer comments regarding the novelty of a paper. 

    Your task:
    1. Deduplicate and consolidate comments that are semantically identical or very similar into a single, concise statement.
    2. Categorize each consolidated comment into one of the following classes:
       - Positive Novelty Evaluations
       - Neutral Novelty Evaluations
       - Negative Novelty Evaluations
    3. Use the exact output format:

    Positive Novelty Evaluations
    - [Consolidated positive comment 1]
    - [Consolidated positive comment 2]

    Neutral Novelty Evaluations
    - [Consolidated neutral comment 1]

    Negative Novelty Evaluations
    - [Consolidated negative comment 1]
    - [Consolidated negative comment 2]

    Here are the evaluations:
    {reviews}
    """
    return prompt


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


# df = pd.read_csv('emnlp23_novelty.csv', encoding='utf-8')
# review_novelty = list(df['review_novelty'])
# paper_novelty = list(df['paper_novelty'])
#
# output_format = []
# for idx, sen in tqdm(enumerate(review_novelty)):
#     rev = ast.literal_eval(sen)
#     nov_intro = ast.literal_eval(paper_novelty[idx])
#
#     if len(sen) == 0 or len(nov_intro) == 0:
#         result = ''
#     else:
#         prompt = build_prompt(rev)
#         result = gpt4task(prompt, 'gpt-4o-mini')
#     output_format.append(result)
# df['output_format'] = output_format
# df.to_csv('emnlp23_output-format.csv', encoding='utf-8')

input_file = 'emnlp23_novelty.csv'
output_file = 'emnlp23_output-format.csv'
save_every = 50

df = pd.read_csv(input_file, encoding='utf-8')
review_novelty = list(df['review_novelty'])
paper_novelty = list(df['paper_novelty'])

if os.path.exists(output_file):
    df_out = pd.read_csv(output_file, encoding='utf-8')

    processed = df_out['output_format'].notna().sum()
    print(f"have processed {processed}")
else:
    df['output_format'] = ""
    df_out = df.copy()
    processed = 0
    print("new task")

for idx in tqdm(range(processed, len(review_novelty))):
    sen = review_novelty[idx]
    nov_intro = paper_novelty[idx]

    try:
        rev = ast.literal_eval(sen)
        nov_intro_eval = ast.literal_eval(nov_intro)
    except Exception as e:
        print(f"??  {idx} th error: {e}")
        df_out.loc[idx, 'output_format'] = ""
        continue

    if len(sen) == 0 or len(nov_intro) == 0:
        result = ''
    else:
        prompt = build_prompt(rev)
        result = gpt4task(prompt, 'gpt-4o')

    df_out.loc[idx, 'output_format'] = result


    if (idx + 1) % save_every == 0:
        df_out.to_csv(output_file, encoding='utf-8', index=False)
        print(f"? have saved {output_file}��now {idx+1}/{len(df)}")


df_out.to_csv(output_file, encoding='utf-8', index=False)
print("All saved")