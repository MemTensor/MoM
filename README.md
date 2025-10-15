<h1 align="center">
    MoM: Mixtures of Scenario-Aware Document Memories for Retrieval-Augmented Generation Systems
</h1>
<p align="center">
    <a href="">
        <img alt="arXiv Paper" src="https://img.shields.io/badge/arXiv-Paper-b31b1b.svg?logo=arxiv">
    </a>
    <a href="">
        <img src="https://img.shields.io/badge/Huggingface-Paper-yellow?style=flat-square&logo=huggingface">
    </a>
    <a href="https://opensource.org/license/apache-2-0">
        <img alt="Apache 2.0 License" src="https://img.shields.io/badge/License-Apache_2.0-green.svg?logo=apache">
    </a>
    <br>
    <a href="https://huggingface.co/datasets/Robot2050/MoM">
        <img src="https://img.shields.io/badge/Huggingface-Dataset-FF6F00?style=flat-square&logo=huggingface">
    </a>
    <a href="https://huggingface.co/Robot2050/MoM/tree/main/scenario_cot_ratio_1.5B">
        <img src="https://img.shields.io/badge/Model-MemReader 1.5B-FF6F00?style=flat-square&logo=huggingface">
    </a>
    <a href="https://huggingface.co/Robot2050/MoM/tree/main/scenario_cot_ratio_3B">
        <img src="https://img.shields.io/badge/Model-MemReader 3B-FF6F00?style=flat-square&logo=huggingface">
    </a>
    <a href="https://huggingface.co/Robot2050/MoM/tree/main/scenario_ratio_7B">
        <img src="https://img.shields.io/badge/Model-MemReader 7B-FF6F00?style=flat-square&logo=huggingface">
    </a>
</p>


### 🎯 Who Should Pay Attention to Our Work?

This study proposes an innovative framework aimed at breaking through the cognitive bottlenecks of traditional RAG systems, offering significant reference value for researchers and engineers committed to enhancing the depth and breadth of information processing in LLMs. Specifically, professionals in the following fields will benefit from our work:

**Researchers in NLP and Information Retrieval**: The active memory extraction paradigm proposed in this paper challenges the traditional text processing workflow of "chunk first, then understand", providing a novel research perspective for fields such as document understanding, semantic segmentation, and knowledge representation.

**Developers of LLM Applications**: Our work directly addresses the core challenges faced by RAG systems in constructing knowledge-intensive applications, such as semantic incompleteness and logical fragmentation of text chunks. It offers a systematic approach to generating high-quality, structured document memories.

**Researchers in SLMs**: Facing the limitations of SLMs in complex cognitive tasks, we demonstrate, through the reverse construction strategy of the **C**hain reasoning **o**f **M**emory extraction (CoM), how to efficiently transfer the deep reasoning capabilities of LLMs to SLMs, opening up new pathways for building lightweight, high-performance intelligent systems.

**Scholars in the Interdisciplinary Field of Cognitive Science and AI**: The core of this study lies in simulating the cognitive processes of human experts by transforming unstructured text into hierarchical memories. This provides robust support for exploring human-like cognition, knowledge construction, and reasoning mechanisms in machines.

### ✨ Core Contributions

**Proposing Active Memory Extraction**: We advocate transforming text processing in RAG from passive text chunking to active memory extraction. By simulating domain experts, we first achieve a holistic and macroscopic understanding of documents and then construct structured document memories.

**Defining Structured Document Memories**: We formally define document memories as a triplet composed of a macroscopic logical outline, highly condensed core content, and semantically coherent atomic chunks.

**Constructing the MoM Framework and CoM**: We design the MoM framework, which generates high-quality memories through a multi-path sampling and multi-dimensional evaluation mechanism. Furthermore, we employ a reverse reasoning strategy to construct the CoM, thereby endowing SLMs with complex cognitive capabilities.

**Designing a Three-Layer Retrieval Mechanism and Providing Theoretical Proof**: We develop a three-layer document memory retrieval mechanism encompassing logical outlines, core content, and original text. From a probabilistic modeling perspective, we theoretically demonstrate that this strategy can more effectively reduce information loss and achieve more precise knowledge localization compared to fusing information before retrieval.

## **🛠️ Quick Start**

- Install dependency packages

```bash
pip install -r requirements.txt
```

- Start the milvus-lite service (vector database)

```bash
milvus-server --data /Storage/path/of/the/database
```

- Download models to corresponding directories.
- Modify various configurations  according to your need.
- Run `chunk_*.py` and `mom_*.py`  to accomplish the text chunking task for domain documents.

```bash
CUDA_VISIBLE_DEVICES=0 nohup python chunk_gpt.py >> multifiled/qwen3_14B_set.log 2>&1 &
```

- Subsequently, execute  `quick_start.py` and `retrieval.py` to carry out the retrieval and question-answering processes.

```bash
CUDA_VISIBLE_DEVICES=1 nohup python quick_start.py 
--docs_path 'crud_qwen3_14B_set.json' 
--collection_name 'crud_qwen3_14B_set' 
--retrieve_top_k 8 
--task 'quest_answer' 
--construct_index 
>> log/mom_crud_qwen3_14B_set.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 nohup python retrieval.py 
--data_path 'evaldata/multifieldqa_zh.json'
--save_file 'eval/mom_multifieldqa_zh_qwen3_14B_set.json'
--docs_path 'multifieldqa_zh_qwen3_14B_set.json' 
--collection_name 'multifieldqa_zh_qwen3_14B_set' 
--retrieve_top_k 8 
--construct_index 
>> log/mom_multifieldqa_zh_huagong_qwen3_14B_set.log 2>&1 &
```

- Open and run `chunk.ipynb`, which will conduct a comprehensive quality assessment of the results generated by different chunking strategies.

### 📊 Results

We conduct extensive experiments on three QA datasets across different domains, including news, finance and so on. 

**Performance Across Domains**: Our MemReader demonstrates outstanding performance in handling pure text QA tasks.

**Effectiveness of Evaluation Metrics**: The memory evaluation metrics we proposed are proven to effectively assess the quality of memory chunks, providing a reliable basis for the automatic screening of high-quality document memories.

**Information Supportiveness of Retrieved Content**: The results indicate that the memories extracted and organized by MoM can provide more comprehensive information for downstream tasks.



![Experimental Results](image\experimental_results.png)















