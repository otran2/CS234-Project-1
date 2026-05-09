import argparse
import sys
import os
import json

#Parse command line arguments for the input/output file paths
parser = argparse.ArgumentParser(description='Command line input for the automated RAG pipleline')
parser.add_argument('--input', required=True, help="Input json file path")
parser.add_argument('--output', required=True, help="Desired output json file path")
parser.add_argument('--corpus-dir', required=True, help="Root directory of the .rst corpus")
parser.add_argument('--apikey-txt', required=True, help="Path to the gateway API key text file")
parser.add_argument('--generation-model', required=False, default="api-gpt-oss-120b", help="Gateway model id used for generation (default: api-gpt-oss-120b)")

args = parser.parse_args()
# input = open(args.input, "r")
# output = open(args.output, "w")

#Get API Key
from pathlib import Path
TRITON_API_KEY = Path(args.apikey_txt).expanduser().read_text(encoding="utf-8").splitlines()[0].strip()

#RapidFireAI Imports
from rapidfireai.automl import (
    List,
    RFLangChainRagSpec,
    RFOpenAIAPIModelConfig,
    RFPromptManager,
    RFGridSearch,
)

from rapidfireai import Experiment

from typing import List as listtype, Dict, Any, Optional

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
    }
    for entry in data
]

dataset = Dataset.from_list(rows)
output_rows = []
output_rows_jsonl = Path(str(args.output) + ".rows.jsonl")
if output_rows_jsonl.exists():
    output_rows_jsonl.unlink()


# =============================================================================
# CREATE EXPERIMENT
# =============================================================================
experiment = Experiment(experiment_name="otran_aliang_final_configs", mode="evals")

#Knobs for langchain part of RAG pipeline
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_openai import OpenAIEmbeddings

batch_size = 32

rag_cpu = RFLangChainRagSpec(
    document_loader=DirectoryLoader(
        path=args.corpus_dir,
        glob="**/*.rst",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        sample_seed=1337,
    ),
    text_splitter=RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="gpt2", chunk_size=512, chunk_overlap=32, add_start_index=True
    ),
    embedding_cfg={
        "class": OpenAIEmbeddings,
        "model": "api-tgpt-embeddings",
        "api_key": TRITON_API_KEY,
        "base_url": "https://tritonai-api.ucsd.edu",
        "check_embedding_ctx_length": False,
    },
    vector_store_cfg={"type": "faiss", "batch_size": batch_size},
    search_cfg={"type": "similarity", "k": 15},
    reranker_cfg={
        "class": CrossEncoderReranker,
        "model_name": "BAAI/bge-reranker-v2-m3",
        "model_kwargs": {"device": "cpu"},
        "top_n": 1,
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
Question: "What are the two main execution functions provided by the Experiment class for launching workflows?"
Answer: "The two main execution functions are run_fit() for training/evaluation workflows and run_evals() for LLM evaluation workflows."
Source Evidence: source_evidence": [
    { "file": "experiment.rst", "lines": [69, 75] },
    { "file": "experiment.rst", "lines": [154, 160] }
    ]
"""

# Token-aware truncation
MAX_TOKENS_PER_QUERY: int = 2000
SAFETY_TOKENS: int = 50


def count_tokens(text: str) -> int:
    return len(text.split())


def truncate_tokens(text: str, max_tokens: int) -> str:
    toks = text.split()
    if len(toks) <= max_tokens:
        return text
    return " ".join(toks[:max_tokens])

def chunk_to_lines(doc: Document) -> listtype:
    """Convert a chunk's character `start_index` into [start_line, end_line]."""
    src = doc.metadata["source"]
    start_idx = doc.metadata["start_index"]
    # Keep this self-contained so Ray workers can deserialize it reliably.
    text = Path(src).read_text(encoding="utf-8")
    start_line = text[:start_idx].count("\n") + 1
    # Use chunk newline count from the actual chunk text to avoid span inversions.
    end_line = start_line + (doc.page_content or "").count("\n")
    if end_line < start_line:
        end_line = start_line
    return [start_line, end_line]

# =============================================================================
# PREPROCESS / POSTPROCESS FUNCTIONS TODO 
# =============================================================================
def openai_sample_preprocess_fn(
    batch: Dict[str, listtype], rag: RFLangChainRagSpec, prompt_manager: RFPromptManager
) -> Dict[str, listtype]:
    """Function to prepare the final inputs given to the generator model"""

    all_context = rag.get_context(batch_queries=batch["query"], serialize=False)
    serialized_context = rag.serialize_documents(all_context)
    system_tokens = count_tokens(INSTRUCTIONS)
    template_tokens = count_tokens("\nQuestion:\n\nContext:\n\nAnswer:")
    new_serialized = []

    for question, ctx in zip(batch.get("query", []), serialized_context):
        q_tokens = count_tokens(question)
        avail = MAX_TOKENS_PER_QUERY - (system_tokens + q_tokens + template_tokens + SAFETY_TOKENS)
        if avail <= 0:
            new_serialized.append("")
        else:
            new_serialized.append(truncate_tokens(ctx, avail))

    serialized_context = new_serialized
    batch["query_id"] = [int(query_id) for query_id in batch["query_id"]]

    per_doc_lines = [[chunk_to_lines(doc) for doc in docs] for docs in all_context]

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
        "serialized_context": serialized_context,
        "retrieved_context": serialized_context,
        "sources": [
            [
                {"file": Path(doc.metadata["source"]).name, "lines": lines}
                for doc, lines in zip(docs, doc_lines)
            ]
            for docs, doc_lines in zip(all_context, per_doc_lines)
        ],
        "retrieved_spans": [
            [
                (Path(doc.metadata["source"]).name, lines[0], lines[1])
                for doc, lines in zip(docs, doc_lines)
            ]
            for docs, doc_lines in zip(all_context, per_doc_lines)
        ],
        **batch,
    }

def sample_postprocess_fn(batch: Dict[str, listtype]) -> Dict[str, listtype]:
    """No regex extraction needed — LLM judge scores raw prose answers"""
    batch["answer"] = batch["generated_text"]

    for qid, ans, ctx, srcs in zip(
        batch["query_id"],
        batch["answer"],
        batch["retrieved_context"],
        batch["sources"],
    ):
        row = {
            "question_id": int(qid),
            "answer": ans,
            "retrieved_context": ctx,
            "sources": srcs,
        }
        output_rows.append(row)
        # Persist each row so outputs survive multi-process Ray workers.
        with open(output_rows_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    return batch

# =============================================================================
# CUSTOM EVALUATION METRIC FUNCTIONS FOR RAG TODO
# =============================================================================
    # Minimal no-op metric functions to satisfy the evals API when metrics are
    # intentionally disabled. These return empty metrics so the pipeline runs.

def compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    return {}


def accumulate_metrics_fn(aggregated_metrics: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    return {}
# Use metrics implemented in rapidfire_integration_example.py


# =============================================================================
# GENERATOR CONFIG (RFOpenAIAPIModelConfig) 
# -----------------------------------------------------------------------------
#
openai_config = RFOpenAIAPIModelConfig(
    client_config={"api_key": TRITON_API_KEY, "base_url": "https://tritonai-api.ucsd.edu", "max_retries": 2},
    model_config={
        "model": args.generation_model,
        "max_completion_tokens": 2048,
        "temperature": 0.8,
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
    "compute_metrics_fn": compute_metrics_fn,
    "accumulate_metrics_fn": accumulate_metrics_fn,
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
output_rows.sort(key=lambda x: x["question_id"])
if output_rows_jsonl.exists():
    with open(output_rows_jsonl, "r", encoding="utf-8") as f:
        output_rows = [json.loads(line) for line in f if line.strip()]
    output_rows.sort(key=lambda x: x["question_id"])
with open(args.output, "w") as f:
    json.dump(output_rows, f, indent=2)

experiment.end()
