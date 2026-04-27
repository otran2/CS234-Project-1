#Get API Key
from pathlib import Path
TRITON_API_KEY = Path("~/api-key.txt").expanduser().read_text(encoding="utf-8").splitlines()[0].strip()

#RapidFireAI Imports
from rapidfireai.automl import (
    List,
    RFLangChainRagSpec,
    RFOpenAIAPIModelConfig,
    RFPromptManager,
    RFGridSearch,
)
###################### Import before Experiment
# from rapidfireai_datahub_compat import (
#     patch_rapidfireai_for_datahub
# )
# print(patch_rapidfireai_for_datahub())
######################
from rapidfireai import Experiment

import re, json
from typing import List as listtype, Dict, Any

import pandas as pd
from datasets import Dataset

#Create Experiment
experiment = Experiment(experiment_name="exp1-sourcedocs-full-evaluation", mode="evals")

#Knobs for langchain part of RAG pipeline
from langchain_community.document_loaders import DirectoryLoader, JSONLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from typing import Dict

batch_size = 32

# CPU-based RAG
rag_cpu = RFLangChainRagSpec(
    # Replace document_loader in rag_cpu
    document_loader=DirectoryLoader(
        path="sourcedocs/sourcedocs/",
        glob="**/*.rst",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        sample_seed=1337,
    ),
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2", chunk_size=512, chunk_overlap=32
    ),
    embedding_cfg=List([
        # {
        #     "class": HuggingFaceEmbeddings,
        #     "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        #     "model_kwargs": {"device": "cpu"},
        #     "encode_kwargs": {"normalize_embeddings": True, "batch_size": batch_size},
        # },
        { # For quicker demo in class
            "class": OpenAIEmbeddings,
            "model": "api-tgpt-embeddings",
            "api_key": TRITON_API_KEY,
            "base_url": "https://tritonai-api.ucsd.edu",
            "check_embedding_ctx_length": False,
        },
    ]),
    # FAISS is an in-memory store and only works in create mode. 
    vector_store_cfg={
        "type": "faiss", 
        "batch_size": batch_size
    }, # if not set, uses FAISS by default
    search_cfg=List([{"type": "similarity", "k": 10}, {"type": "mmr", "k": 10}]), # 2 different search types
    reranker_cfg={
        "class": CrossEncoderReranker,
        "model_name": "cross-encoder/ms-marco-MiniLM-L6-v2",
        "model_kwargs": {"device": "cpu"},
        "top_n": 5,
    },
    enable_gpu_search=False,
)