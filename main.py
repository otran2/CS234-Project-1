import argparse

#Parse command line arguments for the input/output file paths
parser = argparse.ArgumentParser(description='Command line input for the automated RAG pipleline')
parser.add_argument('--input', required=True, help="Input json file path")
parser.add_argument('--output', required=True, help="Desired output json file path")

args = parser.parse_args()
input = open(args.input, "r")
output = open(args.output, "w")

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

#Instructions for data processing
INSTRUCTIONS = """You are a precise technical assistant for the RapidFire AI documentation.
You will be given a user question and relevant context chunks retrieved from the RapidFire AI docs.

Rules:
- Answer using ONLY information present in the provided context. Do not use outside knowledge.
- Be specific and complete — include parameter names, types, defaults, and exact values when present.
- For procedural questions, list the steps in order.
- For comparative questions, clearly distinguish between the two things being compared.
- For factual/lookup questions, give the exact answer directly.
- If the context does not contain enough information to answer, say: "The provided context does not contain enough information to answer this question."
- Do not add caveats, filler phrases, or unnecessary preamble. Get to the answer immediately.
- Stay within 2000 tokens total context budget.

Respond with your answer only. No reasoning prefix needed.
"""

#Data processing/post-processing functions for RAG pipeline
def openai_sample_preprocess_fn(
    batch: Dict[str, listtype], rag: RFLangChainRagSpec, prompt_manager: RFPromptManager
) -> Dict[str, listtype]:
    """Function to prepare the final inputs given to the generator model"""
    
    all_context = rag.get_context(batch_queries=batch["query"], serialize=False)
    serialized_context = rag.serialize_documents(all_context)
    batch["query_id"] = [int(query_id) for query_id in batch["query_id"]]

    return {
        "prompts": [
            [
                {"role": "system", "content": INSTRUCTIONS},
                {
                    "role": "user", 
                    "content": f"\nQuestion:\n{question}\n\nContext:\n{context}\n\nAnswer:"
                },
            ]
            for question, context in zip(batch["query"], serialized_context)
        ],
        "retrieved_context": serialized_context,
        "sources": [
            [{"file": Path(doc.metadata["source"]).name, "lines": [0, 0]} for doc in docs]
            for docs in all_context
        ],
        **batch,
    }

def sample_postprocess_fn(batch: Dict[str, listtype]) -> Dict[str, listtype]:
    """No regex extraction needed — LLM judge scores raw prose answers"""
    batch["answer"] = batch["generated_text"]
    return batch

#Custom evaluation metrics for RAG pipeline
def sample_compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    """Function to compute all eval metrics based on retrievals and/or generations"""

    true_positives, precisions, recalls, f1_scores, ndcgs, rrs, acc = 0, [], [], [], [], [], []
    total_queries = len(batch["query"])

    for pred, gt in zip(batch["retrieved_documents"], batch["ground_truth_documents"]):
        expected_set = set(gt)
        retrieved_set = set(pred[:3])

        true_positives = len(expected_set.intersection(retrieved_set))
        precision = true_positives / len(retrieved_set) if len(retrieved_set) > 0 else 0
        recall = true_positives / len(expected_set) if len(expected_set) > 0 else 0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
        ndcgs.append(compute_ndcg_at_k(retrieved_set, expected_set, k=3))
        rrs.append(compute_rr(retrieved_set, expected_set))
    
    accuracy = compute_accuracy(batch["answer"], batch["label"])
        

    return {
        "Total": {"value": total_queries},
        "Precision": {"value": sum(precisions) / total_queries},
        "Recall": {"value": sum(recalls) / total_queries},
        "F1 Score": {"value": sum(f1_scores) / total_queries},
        "NDCG@3": {"value": sum(ndcgs) / total_queries},
        "MRR": {"value": sum(rrs) / total_queries},
        "Accuracy": {"value": accuracy}
    }


def sample_accumulate_metrics_fn(
    aggregated_metrics: Dict[str, listtype],
) -> Dict[str, Dict[str, Any]]:
    """Function to accumulate eval metrics across all batches"""

    num_queries_per_batch = [m["value"] for m in aggregated_metrics["Total"]]
    total_queries = sum(num_queries_per_batch)
    algebraic_metrics = ["Precision", "Recall", "F1 Score", "NDCG@3", "MRR", "Accuracy"]

    return {
        "Total": {"value": total_queries},
        **{
            metric: {
                "value": sum(
                    m["value"] * queries
                    for m, queries in zip(
                        aggregated_metrics[metric], num_queries_per_batch
                    )
                )
                / total_queries,
                "is_algebraic": True,
                "value_range": (0, 1),
            }
            for metric in algebraic_metrics
        },
    }
input.close()
output.close()