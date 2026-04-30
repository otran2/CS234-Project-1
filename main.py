import argparse
import sys
import os

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
#Issues with API keys
os.environ.setdefault("OPENAI_API_KEY", TRITON_API_KEY)
os.environ.setdefault("JUDGE_BASE_URL", "https://tritonai-api.ucsd.edu/v1")
#os.environ.setdefault("JUDGE_MODEL", "claude-sonnet-4-6")
#os.environ.pop("OPENAI_API_KEY", None)
#os.environ.pop("OPENAI_BASE_URL", None)

sys.path.insert(0, str(Path(__file__).parent / "Metrics"))

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
from project1_eval import call_judge, f1_at_k, precision_at_k, recall_at_k, to_spans

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
output_rows_jsonl = Path(str(args.output) + ".rows.jsonl")
if output_rows_jsonl.exists():
    output_rows_jsonl.unlink()


# =============================================================================
# CREATE EXPERIMENT
# =============================================================================
experiment = Experiment(experiment_name="exp1-sourcedocs-full-evaluation3", mode="evals")

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
        # old version:
        # "sources": [
        #     [{"file": Path(doc.metadata["source"]).name, "lines": _compute_doc_line_span(doc)} for doc in docs]
        #     for docs in all_context
        # ],
        "sources": [
            [
                {"file": Path(doc.metadata["source"]).name, "lines": lines}
                for doc, lines in zip(docs, doc_lines)
            ]
            for docs, doc_lines in zip(all_context, per_doc_lines)
        ],
        # old version:
        # "retrieved_spans": [
        #     [(Path(doc.metadata["source"]).name, span[0], span[1]) for doc, span in zip(docs, [_compute_doc_line_span(doc) for doc in docs])]
        #     for docs in all_context
        # ],
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
def sample_compute_metrics_fn(batch: Dict[str, listtype]) -> Dict[str, Dict[str, Any]]:
    """Compute project retrieval and judge metrics using the shared helpers."""

    total = len(batch["query"])
    mean = lambda xs: sum(xs) / total if total else 0.0

    retrieved_spans = [to_spans(spans) for spans in batch["retrieved_spans"]]
    ground_truth_spans = [to_spans(spans) for spans in batch["ground_truth_spans"]]

    f1s = [f1_at_k(r, g) for r, g in zip(retrieved_spans, ground_truth_spans)]
    ps = [precision_at_k(r, g) for r, g in zip(retrieved_spans, ground_truth_spans)]
    rs = [recall_at_k(r, g) for r, g in zip(retrieved_spans, ground_truth_spans)]

    metrics: Dict[str, Dict[str, Any]] = {
        "Total": {"value": total},
        "F1_at_5": {"value": mean(f1s)},
        "Precision_at_5": {"value": mean(ps)},
        "Recall_at_5": {"value": mean(rs)},
        "Retrieval Score": {"value": (mean(f1s) + mean(ps) + mean(rs)) / 3 if total else 0.0},
    }

    if "generated_text" in batch:
        import os

        model = os.environ.get("JUDGE_MODEL", "claude-sonnet-4-6")
        base_url = os.environ.get("JUDGE_BASE_URL") or None
        corr, faith, comp, failures = [], [], [], 0

        for q, ref, ctx, ans in zip(
            batch["query"],
            batch["reference_answer"],
            batch["serialized_context"],
            batch["generated_text"],
        ):
            result = call_judge(q, ref, ctx, ans, model=model, base_url=base_url)
            if result.get("failed"):
                failures += 1
            corr.append(result["correctness"])
            faith.append(result["faithfulness"])
            comp.append(result["completeness"] / 5.0)

        metrics.update({
            "Correctness_pass_rate": {"value": mean(corr)},
            "Faithfulness_pass_rate": {"value": mean(faith)},
            "Completeness_normalized": {"value": mean(comp)},
            "Generation_Score_3_released": {"value": (mean(corr) + mean(faith) + mean(comp)) / 3 if total else 0.0},
            "Judge Failures": {"value": failures},
        })

    return metrics


def sample_accumulate_metrics_fn(
    aggregated_metrics: Dict[str, listtype],
) -> Dict[str, Dict[str, Any]]:
    """Weighted averages over batches, weighted by query count."""

    num_queries_per_batch = [m["value"] for m in aggregated_metrics.get("Total", [])]
    total_queries = sum(num_queries_per_batch)
    out: Dict[str, Dict[str, Any]] = {"Total": {"value": total_queries}}

    if total_queries == 0:
        return out

    for metric in [
        "F1_at_5",
        "Precision_at_5",
        "Recall_at_5",
        "Retrieval Score",
        "Correctness_pass_rate",
        "Faithfulness_pass_rate",
        "Completeness_normalized",
        "Generation_Score_3_released",
    ]:
        if metric in aggregated_metrics:
            out[metric] = {
                "value": sum(m["value"] * n for m, n in zip(aggregated_metrics[metric], num_queries_per_batch)) / total_queries,
                "is_algebraic": True,
                "value_range": (0, 1),
            }

    if "Judge Failures" in aggregated_metrics:
        out["Judge Failures"] = {"value": sum(m["value"] for m in aggregated_metrics["Judge Failures"])}

    return out


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
output_rows.sort(key=lambda x: x["question_id"])
if output_rows_jsonl.exists():
    with open(output_rows_jsonl, "r", encoding="utf-8") as f:
        output_rows = [json.loads(line) for line in f if line.strip()]
    output_rows.sort(key=lambda x: x["question_id"])
with open(args.output, "w") as f:
    json.dump(output_rows, f, indent=2)

experiment.end()
