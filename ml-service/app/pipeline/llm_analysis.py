"""Summary + Sentiment + QA Ratings via one local LLM (Qwen2.5-3B-Instruct,
GGUF Q4_K_M, llama-cpp-python), constrained to a single JSON object.

One combined call (not three separate calls) to avoid re-processing a long
transcript prompt 3x on CPU — prompt-eval dominates CPU inference cost.
"""

import json
import os
from functools import lru_cache
from typing import TypedDict

from llama_cpp import Llama

from app.core.config import settings


class QaCriterion(TypedDict):
    name: str
    score: int
    rationale: str


class QaRatings(TypedDict):
    overallScore: int
    criteria: list[QaCriterion]


class AnalysisResult(TypedDict):
    summary: str
    sentiment: dict
    qaRatings: QaRatings


QA_RUBRIC = [
    "Greeting",
    "Active Listening",
    "Empathy",
    "Resolution",
    "Compliance",
    "Professionalism",
]

RESULT_SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "sentiment": {
            "type": "object",
            "properties": {"overall": {"type": "string"}},
            "required": ["overall"],
        },
        "qaRatings": {
            "type": "object",
            "properties": {
                "overallScore": {"type": "integer"},
                "criteria": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "score": {"type": "integer"},
                            "rationale": {"type": "string"},
                        },
                        "required": ["name", "score", "rationale"],
                    },
                },
            },
            "required": ["overallScore", "criteria"],
        },
    },
    "required": ["summary", "sentiment", "qaRatings"],
}


@lru_cache(maxsize=1)
def get_llm() -> Llama:
    model_path = settings.models_cache_dir / settings.llm_model_path
    return Llama(
        model_path=str(model_path),
        n_ctx=settings.llm_context_size,
        n_threads=os.cpu_count(),
        verbose=False,
    )


def build_analysis_prompt(diarized_transcript: str) -> str:
    return f"""You are a call-quality analyst. Given the transcript below, respond with a
single JSON object with keys "summary" (string), "sentiment" (object with
"overall" and optionally per-speaker breakdown), and "qaRatings" (object with
"overallScore" 1-10 and "criteria": array of {{name, score, rationale}} for
each of: {", ".join(QA_RUBRIC)}).

Transcript:
{diarized_transcript}
"""


def analyze(diarized_transcript: str) -> AnalysisResult:
    # TODO: for transcripts exceeding the context budget, map-reduce
    # (chunk -> per-chunk summary -> combine) before this structured pass.
    llm = get_llm()
    response = llm.create_chat_completion(
        messages=[{"role": "user", "content": build_analysis_prompt(diarized_transcript)}],
        response_format={
            "type": "json_object",
            "schema": RESULT_SCHEMA,
        },
        temperature=0.2,
    )
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)
