import argparse

#Parse command line arguments for the input/output file paths
parser = argparse.ArgumentParser(description='Command line input for the automated RAG pipleline')
parser.add_argument('--input', required=True, help="Input json file path")
parser.add_argument('--output', required=True, help="Desired output json file path")

args = parser.parse_args()
# input = open(args.input, "r")
# output = open(args.output, "w")

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

# =============================================================================
# LOAD INPUT JSON & BUILD HUGGINGFACE DATASET
# =============================================================================


with open(args.input, "r") as f:
    data = json.load(f)

rows = [
    {
        "query_id": int(entry["question_id"]),          
        "query":    str(entry["question"]),
        "reference_answer": entry.get("reference_answer", ""), 
        "ground_truth_spans": entry.get("source_evidence", [])         
    }
    for entry in data
]

dataset = Dataset.from_list(rows)
output_rows = []


# =============================================================================
# CREATE EXPERIMENT
# =============================================================================
experiment = Experiment(experiment_name="exp1-sourcedocs-full-evaluation", mode="eval")

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
        encoding_name="gpt2", chunk_size=512, chunk_overlap=32, add_start_index=True
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

# =============================================================================
# PREPROCESS / POSTPROCESS FUNCTIONS TODO 
# =============================================================================
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
            [{"file": Path(doc.metadata["source"]).name, "lines": [doc.metadata["start_line"], doc.metadata["end_line"]]} for doc in docs]
            for docs in all_context
        ],
        **batch,
    }

def sample_postprocess_fn(batch: Dict[str, listtype]) -> Dict[str, listtype]:
    """No regex extraction needed — LLM judge scores raw prose answers"""
    batch["answer"] = batch["generated_text"]
    
    for qid, answer, context, srcs in zip(
        batch["query_id"], 
        batch["answer"], 
        batch["retrieved_context"], 
        batch["sources"],
        ):
            output_rows.append({
                "question_id": int(qid),
                "answer": answer,
                "retrieved_context": context,
                "sources": srcs,
            })
    return batch

# =============================================================================
# CUSTOM EVALUATION METRIC FUNCTIONS FOR RAG TODO
# =============================================================================
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


# =============================================================================
# GENERATOR CONFIG (RFOpenAIAPIModelConfig) 
# -----------------------------------------------------------------------------
#
openai_config = RFOpenAIAPIModelConfig(
    client_config={"api_key": TRITON_API_KEY, "base_url": "https://tritonai-api.ucsd.edu", "max_retries": 2},
    model_config={
        "model": "api-mistral-small-3.2-2506",
        "max_completion_tokens": 2048,
    },
    rpm_limit=120, 
    tpm_limit=1_000_000, 
    rag=rag_cpu,
    prompt_manager=None,
)

config_set = {
    "openai_config": openai_config,
    "preprocess_fn": openai_sample_preprocess_fn,
    "postprocess_fn": sample_postprocess_fn,
    "compute_metrics_fn": sample_compute_metrics_fn,
    "accumulate_metrics_fn": sample_accumulate_metrics_fn,
    "batch_size": batch_size,
}
config_group = RFGridSearch(config_set)

# =============================================================================
# RUN EVALS
# =============================================================================
results = experiment.run_evals(
    config_group=config_group,
    dataset=dataset,
    num_shards=4,
    num_actors=4,
    seed=42,
)

# =============================================================================
# OUTPUT JSON
# =============================================================================
# output_rows.sort(key=lambda x: x["question_id"])
with open(args.output, "w") as f:
    json.dump(output_rows, f, indent=2)
