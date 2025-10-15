import os
from tqdm import tqdm
import time  
import json
import re
from nltk.tokenize import sent_tokenize
import jieba
from openai import OpenAI
import Levenshtein as lev  
import requests
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name_or_path="moc_metachunker" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
sampling_params = SamplingParams(temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=1280) 
model = LLM(model=model_name_or_path,tensor_parallel_size=1, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 




def get_wenxinyiyan(prompt): 
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/ernie-3.5-128k?access_token=" + ""
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "temperature": 0.7,
        "top_p": 0.8,
        "penalty_score": 1,
        "disable_search": True,
        "enable_citation": False,
        "response_format": "text"
    }, ensure_ascii=False)
    headers = {
        'Content-Type': 'application/json'
    }
    
    response = requests.request("POST", url, headers=headers, data=payload.encode("utf-8"))
    
    return json.loads(response.text)['result']

def get_yiyan(prompt):
    client = OpenAI(
        api_key='', 
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    completion = client.chat.completions.create(
        model="qwen-max",  
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        top_p=0.8
        )
    
    return json.loads(completion.model_dump_json())["choices"][0]['message']['content']

def prompt_llm(user_prompt):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": user_prompt}
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,  # Set to False to strictly disable thinking
    )
    outputs = model.generate([text], sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
    response = generated_text
    return response

def chunking_by_llmsft(text):
    user_prompt='''这是一个文本分块任务，你是一名文本分割方面的专家，负责将给定的文本分割成文本块。你必须遵守以下四个条件：
1. 将内容相关的若干个连续句子组合成文本块，使得每个文本块都具有完整的逻辑表达。
2. 避免文本块过短，在识别内容转换与分块长度之间取得良好平衡。
3. 输出的分块结果是一个列表格式，其中的每个元素表示文档中的一个文本块。
4. 其中输出的每个文本块由文本块开头和结尾几个字符组成，中间内容由“[MASK]”来代替，输出格式如下：
[
    "文本块1的开头几个字符[MASK]文本块1的结尾几个字符",
    ......
]

如果你理解，请将以下文本分割成文本块，并通过上述要求输出列表格式。

文档内容：'''
    prompt=user_prompt+text+'\n\n分割好的文本块的列表格式输出：'
    response=prompt_llm(prompt)
    return response

def is_valid_json(json_str):
    try:
        json.loads(json_str)
        return True
    except json.JSONDecodeError:
        return False

def split_text_by_punctuation(text,language,target_size): 
    if language=='en': 
        full_segments = sent_tokenize(text)
        merged_paragraphs = []  
        current_paragraph = "" 
        for paragraph in full_segments:  
            if len(current_paragraph.split()) + len(paragraph.split()) <= target_size:  
                current_paragraph +=' '+paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  
                current_paragraph = paragraph  
        if current_paragraph:  
            merged_paragraphs.append(current_paragraph)  
        return merged_paragraphs
    elif language=='zh':
        sentences = jieba.cut(text, cut_all=False)  
        sentences_list = list(sentences)  
        sentences = []  
        temp_sentence = ""  
        for word in sentences_list:  
            if word in ["。", "！", "？","；"]:  
                sentences.append(temp_sentence.strip()+word)  
                temp_sentence = ""  
            else:  
                temp_sentence += word  
        if temp_sentence:   
            sentences.append(temp_sentence.strip())  
        
        merged_paragraphs = []  
        current_paragraph = "" 
        for paragraph in sentences:  
            # 检查如果当前段落加上新段落是否超过目标大小  
            if len(current_paragraph) + len(paragraph) <= target_size:  
                current_paragraph +=paragraph  
            else:  
                merged_paragraphs.append(current_paragraph)  # 添加当前合并的段落到结果列表  
                current_paragraph = paragraph  # 重置当前段落为新段落  
        if current_paragraph:  
            merged_paragraphs.append(current_paragraph)  
        return merged_paragraphs


def chunking_by_llm_list(content):
    system_prompt='''这是一个文本分块任务，你是一名文本分割方面的专家，负责将给定的文本分割成文本块。你必须遵守以下五个条件：
1. 根据文本的逻辑和语义结构对文本进行分割，使得每个文本块都具有完整的逻辑表达。
2. 避免文本块过短，在识别内容转换与分块长度之间取得良好平衡。
3. 不要改变文本的原始词汇或内容。
4. 不要添加任何新词或符号。
5. 输出完整的分块结果，不能够省略。

如果你理解，请将以下文档内容分割成文本块，直接回复分割好的文本块，不要输出其他任何解释信息。每个文本块使用<chunk>和</chunk>来包含。


文档内容：{}


分割好的文本块为：'''.format(content)
    raw_gpt_output=get_yiyan(system_prompt)
    
    save = {}
    pattern = r"<chunk>(.*?)</chunk>"
    matches = re.findall(pattern, raw_gpt_output,re.DOTALL)
    gpt_output=[]
    # 输出提取到的内容
    for match in matches:
        gpt_output.append(match.strip())
    
    save['raw_corpus'] = content
    save['gpt_output'] = ''
    save['final_chunks'] = gpt_output
    
    return save



def find_most_similar_text(reference_text, original_text):
    window_size=len(reference_text)
    min_distance = float('inf')
    most_similar_paragraph = None
    
    for i in range(len(original_text) - window_size + 1):
        window = original_text[i:i + window_size]
        
        distance = lev.distance(window, reference_text)

        if distance < min_distance:
            min_distance = distance
            most_similar_paragraph = window

    return most_similar_paragraph, min_distance


filename ="OmniEval/chunk_1.5all3_sft_3M.json"  

with open('', 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)
    
start_time=time.time()

all_data_chunk=[]          
for text in tqdm(qa_data):
    try:
        chunks_json=chunking_by_llmsft(text)   
        
        if not is_valid_json(chunks_json):
            save=chunking_by_llm_list(text)  
        else:
            keys_list = json.loads(chunks_json)
            new_final_chunks=[]
            for regex_pattern in keys_list:
                if regex_pattern.strip() !='':
                    zzz=regex_pattern.strip()
                    regex_pattern=regex_pattern.replace('[MASK]','.*?')
                    regex_pattern=regex_pattern.replace('|','\|').strip()
                    try:
                        matches = re.findall(regex_pattern, text, re.DOTALL)
                        new_final_chunks.append(matches[0])
                    except:
                  
                        try:
                            my_regex=zzz
                            if '[MASK]' in my_regex:
                                left_right = re.split(r'\[MASK\]', my_regex)
                                tmp_left=left_right[0]
                                tmp_right=left_right[-1]
                                left, distance1 = find_most_similar_text(tmp_left, text)
                                right, distance2 = find_most_similar_text(tmp_right, text)
                                my_regex=left+".*?"+right
                                my_regex=my_regex.replace('[','\[')
                                my_regex=my_regex.replace(']','\]')
                                my_regex=my_regex.replace('|','\|')
                                matches = re.findall(my_regex, text, re.DOTALL)
                                # print(matches)
                                new_final_chunks.append(matches[0])
                            else:
                                new_final_chunks.append(my_regex)
                        except:
                            print([regex_pattern],flush=True)
                            extract_prompt='''正则表达式：{}
    您是一名文本提取专家。请根据上述正则表达式从以下文档中提取出相应文本块，其中正则表达式由文本块开头和结尾几个字符和中间的省略符号组成，尽可能完整的识别并提取，保证文本块开头和结尾与正则表达式的开头和结尾匹配。只输出与文档内容匹配的原文本，不输出任何解释。

    文档内容：{}'''.format(regex_pattern,text)

                            match_output=get_wenxinyiyan(extract_prompt)
                            match_output.replace(regex_pattern,'')
                            new_final_chunks.append(match_output.strip())
            save = {}
            save['raw_corpus'] = text
            save['gpt_output'] = chunks_json
            save['final_chunks'] = new_final_chunks
        
        all_data_chunk.append(save)
    except:
        json_str = json.dumps({'raw_corpus':text}, ensure_ascii=False)
        with open('OmniEval/error.jsonl', 'a',encoding='utf-8') as file:
            file.write(json_str + '\n')

    with open(filename, 'w', encoding='utf-8') as sfile:
        json.dump(all_data_chunk, sfile, ensure_ascii=False, indent=4)

end_time = time.time()  
# Calculate and print execution time
execution_time = end_time - start_time  
print(f"The program execution time is: {execution_time} s.")

# CUDA_VISIBLE_DEVICES=5 nohup python chunk_moc.py >> OmniEval/chunk_1.5all3_sft_3M.log 2>&1 &