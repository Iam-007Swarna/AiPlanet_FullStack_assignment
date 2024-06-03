#############################Imports##############################################
## General imports
import os
import sys
import zipfile
import logging
import IPython
from IPython.display import display
from pyvis.network import Network

HF_TOKEN = os.environ.get("HF_TOKEN", None)
MISTRAL_API = os.environ.get("MISTRAL_API", None)

## logger
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().handlers = []
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Knowledge graph imports
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    KnowledgeGraphIndex,
    load_index_from_storage,
    Settings,
)
from llama_index.core.graph_stores import SimpleGraphStore
from llama_index.embeddings.mistralai import MistralAIEmbedding
from llama_index.llms.mistralai import MistralAI

## BM25 imports
from llama_index.core import (
    VectorStoreIndex,
    QueryBundle,
)
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.retrievers import (
    BaseRetriever,
    VectorIndexRetriever,
    QueryFusionRetriever,
    )
from llama_index.core.schema import NodeWithScore
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.response.notebook_utils import (
    display_response,
    display_source_node,
)

# Chat engine
from llama_index.core import PromptTemplate
from llama_index.core import chat_engine
from llama_index.core import memory

################### Loading the LLM via Mistral API
llm = MistralAI(api_key=MISTRAL_API, model="open-mixtral-8x7b")
### Loading the Embedding via Mistral API
embed_model = MistralAIEmbedding(api_key=MISTRAL_API, model = "mistral-embed")

################### Knowledge Graph Index#################
########### While loading from persist only ###############
Settings.llm = llm
Settings.embed_model = embed_model
kg_storage_context = StorageContext.from_defaults(persist_dir="/storage/kg")
kg_index = load_index_from_storage(kg_storage_context)

kg_retriever = kg_index.as_retriever(include_text=True,
                                     response_mode ="tree_summarize",
                                     embedding_mode="hybrid",
                                     similarity_top_k=10)

################### BM25 Index #################
####################### While load
# ing Indices from persist #######################
# Settings.llm = llm
# Settings.embed_model = embed_model
storage_context_v = StorageContext.from_defaults(persist_dir="/storage/bm25")
index_v = load_index_from_storage(storage_context_v)

vector_retriever = index_v.as_retriever(similarity_top_k=10)
bm25_retriever = BM25Retriever.from_defaults(index=index_v, similarity_top_k=10, verbose=False) ## loading after persist


###################### Reranker from HuggingFace 🤗
reranker = SentenceTransformerRerank(top_n=5, model="mixedbread-ai/mxbai-rerank-base-v1", keep_retrieval_score=False)

################ Query Fusion Retriever ##################
QUERY_GEN_PROMPT = (
    "You are a helpful assistant that generates multiple search queries on astronomy based on a "
    "single input query. Generate {num_queries} search queries for astronomy, one on each line, "
    "related to the following input query:\n"
    "Query: {query}\n"
    "Queries:\n"
)

#### Hybrid (Dense + BM25 + KG) Retriever ####
hybrid_retriever = QueryFusionRetriever(
    retrievers = [vector_retriever, bm25_retriever, kg_retriever],
    retriever_weights = [0.25, 0.25, 0.50],
    similarity_top_k=5,
    llm=llm,
    num_queries=3,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=False,
    query_gen_prompt=QUERY_GEN_PROMPT,
)

#### KG Retriever ####
kg_retriever = QueryFusionRetriever(
    retrievers = [kg_retriever],
    # retriever_weights = [0.25, 0.25, 0.50],
    similarity_top_k=5,
    llm=llm,
    num_queries=3,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=False,
    query_gen_prompt=QUERY_GEN_PROMPT,
)

#### Hybrid BM25 (Dense + BM25) Retriever ####
hybrid_bm25_retriever = QueryFusionRetriever(
    retrievers = [vector_retriever, bm25_retriever],
    # retriever_weights = [0.25, 0.25, 0.50],
    similarity_top_k=5,
    llm=llm,
    num_queries=3,  # set this to 1 to disable query generation
    mode="reciprocal_rerank",
    use_async=True,
    verbose=False,
    query_gen_prompt=QUERY_GEN_PROMPT,
)

################# Hybrid Chat Engine ###################
system_prompt_template = """You are a helpful AI assistant for reaearcher or enthusiast in the domain of astronomy.
Please check if the following pieces of context has any mention of the keywords provided in the Question. If not then don't know the answer, just say that you don't know.Stop there. Please donot try to make up an answer."""

from llama_index.core import chat_engine
from llama_index.core import memory

######## Hybrid (Dense + BM25 + KG) chat engine #########
hybrid_memory = memory.ChatMemoryBuffer.from_defaults(token_limit=30000)
Hybrid_chat_engine = chat_engine.CondensePlusContextChatEngine(retriever=hybrid_retriever,
                                          llm=llm,
                                          memory=hybrid_memory,
                                          # context_prompt=,
                                          # condense_prompt=,
                                          system_prompt=system_prompt_template,
                                          node_postprocessors=[reranker],
                                          verbose=False,
                                          )

######## Hybrid (Dense + BM25 + KG) chat engine #########
kg_memory = memory.ChatMemoryBuffer.from_defaults(token_limit=30000)
kg_chat_engine = chat_engine.CondensePlusContextChatEngine(retriever=kg_retriever,
                                          llm=llm,
                                          memory=kg_memory,
                                          # context_prompt=,
                                          # condense_prompt=,
                                          system_prompt=system_prompt_template,
                                          node_postprocessors=[reranker],
                                          verbose=False,
                                          )

######## Hybrid (Dense + BM25 + KG) chat engine #########
hybrid_bm25_memory = memory.ChatMemoryBuffer.from_defaults(token_limit=30000)
hybrid_bm25_chat_engine = chat_engine.CondensePlusContextChatEngine(retriever=hybrid_bm25_retriever,
                                          llm=llm,
                                          memory=hybrid_bm25_memory,
                                          # context_prompt=,
                                          # condense_prompt=,
                                          system_prompt=system_prompt_template,
                                          node_postprocessors=[reranker],
                                          verbose=False,
                                          )

################## Post Processors ###################
def get_documents(response):
    """Get the reference documents for a chat engine response.
    
    Args:
        response (ChatResponse): The chat engine response object.
    
    Returns:
        str: A formatted string containing the reference document paths and page numbers.
    """
    plist = []
    reference_docs={}
    for idx, content in enumerate(response.sources[0].content.split("\n\n")):
        if "page_label" in content:
            plist.append(content)
    for i, entry in enumerate(plist, start=1):
        lines = entry.split('\n')
        page_label = int(lines[0].split(': ')[1])
        file_path = lines[1].split(': ')[1]

        reference_docs[f"doc{i}"] = {
            "document_path": file_path,
            "page": page_label
        }
    reference = f"""
    \n Reference Docs (document paths, page number):
        ...{reference_docs['doc1']['document_path']}, page {reference_docs['doc1']['page']}
        ...{reference_docs['doc2']['document_path']}, page {reference_docs['doc2']['page']}
        ...{reference_docs['doc3']['document_path']}, page {reference_docs['doc3']['page']}
        ...{reference_docs['doc4']['document_path']}, page {reference_docs['doc4']['page']}
        ...{reference_docs['doc5']['document_path']}, page {reference_docs['doc5']['page']}
        """
    return reference

##################### Query Function #####################
def get_query(query=""):
    """Get a response from the Hybrid chat engine and format the result with reference documents.
    
    Args:
        query (str): The query to send to the chat engine.
    
    Returns:
        str: The chat engine response with the reference documents appended.
    """
    response_1 = Hybrid_chat_engine.chat(query)
    response_2 = kg_chat_engine.chat(query)
    response_3 = hybrid_bm25_chat_engine.chat(query)
    reference_1 = get_documents(response_1)
    reference_2 = get_documents(response_2)
    reference_3 = get_documents(response_3)
    reply_1 = response_1.response
    reply_2 = response_2.response
    reply_3 = response_3.response
    result_hybrid = reply_1 + reference_1
    result_kg = reply_2 + reference_2
    result_hybrid_bm25 = reply_3 + reference_3
    return result_hybrid, result_kg, result_hybrid_bm25



# get_query("what is difference between class I and class II ?")

# Based on the provided documents, Class I and Class II are two distinct evolutionary stages of young stellar objects (YSOs) in the process of forming stars. The documents describe the evolutionary stages of YSOs as follows:

# * Class 0: The formation of a YSO in the central region of a protostellar core with an envelope mass that is much in excess of the YSO mass.
# * Class I: The collapse of the envelope onto the central object, with the transition between Class 0 and Class I being the point in time at which the envelope mass and the mass of the protostar are nearly equal.
# * Class II: The emergence of a disk around the central star.
# * Class III: The dissipation of the disk by various processes such as the formation of planets, photo-evaporation, and tidal stripping.

# The documents also mention that an intermediate class between Class 0 and Class I has been proposed, but it is considered to be close to Class I in terms of the evolutionary status of the YSOs.

# Regarding the difference between Class I and Class II, the documents do not provide a detailed explanation, but it is suggested that the main difference lies in the presence of a disk around the central star in Class II, which is not present in Class I. The emergence of a disk around the central star is a sign of a more evolved stage in the star formation process. The documents also mention that the exact duration of each evolutionary stage for YSOs is relatively uncertain and depends on the number of objects found in each class, which may be affected by misclassifications due to YSOs being seen edge-on.
#   Reference Docs (document paths, page number):
#     .../content/documents/2405.00095v1.pdf, page 8
#     .../content/documents/2405.00095v1.pdf, page 3
#     .../content/documents/2405.00095v1.pdf, page 9
#     .../content/documents/2405.00095v1.pdf, page 2
#     .../content/documents/2405.00095v1.pdf, page 2
    