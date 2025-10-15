from abc import ABC

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, get_response_synthesizer
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.postprocessor import SimilarityPostprocessor
from llama_index.node_parser import SimpleNodeParser
from llama_index import download_loader

from llama_index.embeddings import LangchainEmbedding
from llama_index import ServiceContext, StorageContext
from langchain.schema.embeddings import Embeddings
from llama_index.vector_stores import MilvusVectorStore
import json
from llama_index.data_structs import Node
from FlagEmbedding import FlagReranker

import Levenshtein as lev  
import copy

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

def remove_duplicates(input_list):
    result = []   
    for item1 in input_list:
        if item1==None:
            continue
        result.append(item1)
        input_list[0] = None
        tmp=[]
        for i,item2 in enumerate(input_list):
            if item2==None:
                continue
            if len(item1) > len(item2):
                most_similar_paragraph, min_distance=find_most_similar_text(item2, item1)
            else:
                most_similar_paragraph, min_distance=find_most_similar_text(item1, item2)
            # print(most_similar_paragraph, min_distance)
            if min_distance < min(len(item1), len(item2)) / 2:
                tmp.append(i)
        for i,item in enumerate(input_list):
            if i in tmp:
                input_list[i] = None
    return result



class BaseRetriever(ABC):
    def __init__(
            self, 
            docs_directory: str, 
            embed_model: Embeddings,
            embed_dim: int = 768,
            chunk_size: int = 128,
            chunk_overlap: int = 0,
            collection_name: str = "docs",
            construct_index: bool = False,
            add_index: bool = False,
            similarity_top_k: int=2,
            reranker_model:FlagReranker=None
        ):
        self.docs_directory = docs_directory
        self.embed_model = embed_model
        self.embed_dim = embed_dim
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self.reranker_model=reranker_model

        if construct_index:
            self.construct_index()
        else:
            self.load_index_from_milvus()
        
        if add_index:
            self.add_index()

        # self.query_engine = self.vector_index.as_query_engine()
        retriever = VectorIndexRetriever(
            index=self.vector_index,
            similarity_top_k=self.similarity_top_k,
        )

        # assemble query engine
        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
        )

    def construct_index(self):
        folder_path = self.docs_directory  
        nodes=[]
        with open(folder_path, 'r', encoding='utf-8') as file:  
            qa_data = json.load(file)
        tmp_qa_data=[]
        for item in qa_data:
            for i,iit in enumerate(item["final_chunk"]):
                tmp_qa_data.append(item["outline"][i].replace('\n',' ').strip()+'\n'+iit.replace('\n\n','\n').strip())
            tmp_qa_data.append('文档大纲：\n'+'\n'.join(item["outline"]))
            tmp_qa_data.append('文档关键信息：\n'+'\n'.join(item["core_content"]))
        
        for i in tmp_qa_data:
            if not isinstance(i, str):
                continue
            if len(i)<10:
                continue
            node1 = Node(text=i)
            nodes.append(node1)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            dim=self.embed_dim, overwrite=True,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        
        # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):  
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context, 
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

            vector_store = MilvusVectorStore(
                overwrite=False,
                collection_name=self.collection_name
            )
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

        print("Indexing finished!")

    def add_index(self):
        if self.docs_type == 'json':
            JSONReader = download_loader("JSONReader")
            documents = JSONReader().load_data(self.docs_directory)
        else:
            documents = SimpleDirectoryReader(self.docs_directory).load_data()
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        nodes = node_parser.get_nodes_from_documents(documents, show_progress=True)
        
        self.embed_model = LangchainEmbedding(self.embed_model)
        service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model,llm=None,
        )
        vector_store = MilvusVectorStore(
            overwrite=False,
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

         # Process and index nodes in chunks due to Milvus limitations
        for spilt_ids in range(0, len(nodes), 8000):  
            self.vector_index = GPTVectorStoreIndex(
                nodes[spilt_ids:spilt_ids+8000], service_context=service_context, 
                storage_context=storage_context, show_progress=True
            )
            print(f"Indexing of part {spilt_ids} finished!")

        print("Indexing finished!")

    def load_index_from_milvus(self):
        vector_store =  MilvusVectorStore(
            overwrite=False, dim=self.embed_dim, 
            collection_name=self.collection_name
        )
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        service_context = ServiceContext.from_defaults(embed_model=self.embed_model, llm=None)
        self.vector_index = GPTVectorStoreIndex(
            [], storage_context=storage_context, 
            service_context=service_context,
        )

    def search_docs(self, query_text: str):
        response_vector = self.query_engine.query(query_text)

        response_text_list = response_vector.response.split('\n---------------------\n')
        response_text = response_text_list[1].split("\n\n")
        print(response_text)
        with open('', 'r', encoding='utf-8') as file:  
            all_outline_core = json.load(file)
        final_data=[]
        for item in response_text:
            item=item.strip()
            if item.startswith('文档大纲：'):
                try:
                    final_data+=all_outline_core[0][item]
                except:
                    pass
            elif item.startswith('文档关键信息：'):
                try:
                    final_data+=all_outline_core[1][item]
                except:
                    pass
            else:
                final_data.append(item.split("\n", 1)[-1])

        final_data = remove_duplicates(final_data)
        final_data = [item.strip() for item in final_data if item.strip()]
        smi_rerank=[]
        for index, paragraph in enumerate(final_data):
            sim_qc=self.reranker_model.compute_score([query_text, paragraph])
            smi_rerank.append(float(sim_qc[0]))

        sorted_lst = sorted(enumerate(smi_rerank), key=lambda x: x[1],reverse=True)  
        rerank_result=[]
        for i in range(len(sorted_lst)):
            rerank_result.append(final_data[sorted_lst[i][0]])
            
        rerank_result=rerank_result[:4]
        
        response_text = "\n\n".join([text for text in rerank_result if not text.startswith("file_path: ")])
        final_response_text=response_text_list[0]+'\n---------------------\n'+response_text+'\n---------------------\n'+response_text_list[2]
        return final_response_text
