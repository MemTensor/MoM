from openai import OpenAI
import time
import json
from tqdm import tqdm
import re
model_type='Qwen-vllm' 

if model_type=='Qwen-vllm':
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_name_or_path="model/Qwen3-14B" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    sampling_params = SamplingParams(temperature=0.1, top_p=0.1, repetition_penalty=1.05, max_tokens=8192) 
    model = LLM(model=model_name_or_path,tensor_parallel_size=1, pipeline_parallel_size=1, dtype='float16',gpu_memory_utilization=0.9) 



def prompt_llm(model_type, user_prompt):
    if model_type == "GPT":
        try:
            client = OpenAI(
                api_key='',
                base_url="",
            )
            completion = client.chat.completions.create(
                model="gpt-4o",
                max_tokens=8192,
                temperature=0.5,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_prompt}
                ])
            return completion.choices[0].message.content
        except Exception as e:
            print('111',flush=True)
            print(f"An error occurred: {e}.")
            return "GPT thinks prompt is unsafe"
    elif model_type == "GLM":
        from zhipuai import ZhipuAI
        client = ZhipuAI(api_key='')
        response_glm = client.chat.completions.create(
                model='glm-4-plus',  
                messages= [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": user_prompt},
                    ],
            )
        ans_glm=response_glm.choices[0].message.content
        return ans_glm
    elif model_type=='Qwen-vllm':
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        outputs = model.generate([text], sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
        response = generated_text
        return response

def chunking_by_llm(model_type, content):
    system_prompt='''这是一个文本分块任务，你是一名文本分割方面的专家，负责将给定的文本分割成文本块。你必须遵守以下五个条件：
1. 根据文本的逻辑和语义结构对文本进行分割，使得每个文本块都具有完整的逻辑表达。
2. 避免文本块过短，在识别内容转换与分块长度之间取得良好平衡。
3. 不要改变文本的原始词汇或内容。
4. 不要添加任何新词或符号。
5. 输出完整的分块结果，不能够省略。

如果你理解，请将以下文档内容分割成文本块，直接回复分割好的文本块，不要输出其他任何解释信息。每个文本块使用<chunk>和</chunk>来包含。


文档内容：{}


分割好的文本块为：'''.format(content)
    try:
        str_result=prompt_llm(model_type, system_prompt)
        return str_result
    except Exception as e:
        print('111',flush=True)
        print(f"An error occurred: {e}.")
        return "GPT thinks prompt is unsafe"



with open('tmp_data/multifield.json', 'r', encoding='utf-8') as file:  
    qa_data = json.load(file)
start_time = time.time()
save_list=[]
for content in tqdm(qa_data):
    raw_gpt_output=chunking_by_llm(model_type, content.strip())
    save = {}
    if raw_gpt_output == "GPT thinks prompt is unsafe":
        json_str = json.dumps({'raw_corpus':content}, ensure_ascii=False)
        with open('multifiled/nochunk_qwen3_14B.jsonl', 'a',encoding='utf-8') as file:
            file.write(json_str + '\n')
    else: 
        pattern = r"<chunk>(.*?)</chunk>"
        matches = re.findall(pattern, raw_gpt_output,re.DOTALL)
        gpt_output=[]

        for match in matches:
            gpt_output.append(match.strip())
        
        save['raw_corpus'] = content
        save['gpt_output'] = gpt_output
        save['raw_gpt_output'] = raw_gpt_output
        
        save_list.append(save)    
        with open('multifiled/qwen3_14B_set.json', 'w', encoding='utf-8') as sfile:
            json.dump(save_list, sfile, ensure_ascii=False, indent=4)

end_time = time.time()  
execution_time = end_time - start_time  
print(f"The program execution time is: {execution_time} s.")

# CUDA_VISIBLE_DEVICES=2 nohup python chunk_gpt.py >> multifiled/qwen3_14B_set.log 2>&1 &
