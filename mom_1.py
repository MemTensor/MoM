import time
import json
from tqdm import tqdm
import re

from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer
model_name_or_path="scenario_ratio_7B" 
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
sampling_params = SamplingParams(temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=2048) 
model = LLM(model=model_name_or_path,tensor_parallel_size=2, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 



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

def chunking_by_llm(content):
    system_prompt='''这是一个文档记忆提取任务，你是一名记忆提取方面的专家。认真分析下面提供的内容，首先生成一个记忆提取大纲，然后依据大纲为给定的文档生成一组对应数量的场景记忆。
记忆提取大纲生成需要从领域专家的视角出发，利用好原始文档的全局信息，并且其中每条内容代表了对应场景记忆中文本块的作用及其摘要内容。
根据生成的大纲对文档进行记忆提取时，每个场景记忆包含两部分内容：
1. 一个具有完整逻辑表达的文本块，按照逻辑和语义结构从文档中分割得到。要求：避免文本块过短，在识别内容转换与分块长度之间取得良好平衡。输出的每个文本块由文本块开头和结尾几个字符组成，中间内容由“[MASK]”来代替。
2. 描述对应文本块中的核心内容。
整体输出格式如下：
<outline>
文档的记忆提取大纲
</outline>

<scenario>
<chunk>文本块1的开头几个字符[MASK]文本块1的结尾几个字符</chunk>
文本块核心内容描述
</scenario>
.......


如果你理解，按照格式直接回复内容，不同场景记忆之间使用换行来区别。不要输出其他任何解释内容，也不要用引号或其他分隔符括住你的回复。


文档内容：
<document>{}</document>'''.format(content)
    try:
        str_result=prompt_llm(system_prompt)
        return str_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"



with open('tmp_cleandata/crud.json', 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)
save_path='mom/crud_ratio_7B.json'

start_time = time.time()
save_list=[]
for content in tqdm(qa_data):
    raw_gpt_output=chunking_by_llm(content)
    save = {}
    if raw_gpt_output == "GPT thinks prompt is unsafe":
        json_str = json.dumps({'raw_corpus':content}, ensure_ascii=False)
        print('111',json_str,flush=True)
    else: 
        save['raw_corpus'] = content
        save['gpt_output'] = raw_gpt_output
                
        save_list.append(save)    
        
        if len(save_list) % 100 == 0:
            with open(save_path, 'w', encoding='utf-8') as sfile:
                json.dump(save_list, sfile, ensure_ascii=False, indent=4)
with open(save_path, 'w', encoding='utf-8') as sfile:
    json.dump(save_list, sfile, ensure_ascii=False, indent=4)
    
end_time = time.time()  
# Calculate and print execution time
execution_time = end_time - start_time  
print(f"The program execution time is: {execution_time} s.")

# CUDA_VISIBLE_DEVICES=2,3 nohup python mom_1.py >> mom/crud_ratio_7B.log 2>&1 &
