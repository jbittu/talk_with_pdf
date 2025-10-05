from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from config import (
    LLM_MODEL_ID, LLM_LOCAL_PATH, MAX_NEW_TOKENS
)

_llm_pipe = None

def load_llm():
    """Load an open-source chat LLM (CPU-friendly default)."""
    global _llm_pipe
    if _llm_pipe is not None:
        return _llm_pipe

    model_id_or_path = str(LLM_LOCAL_PATH) if LLM_LOCAL_PATH.exists() else LLM_MODEL_ID

    tok = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        torch_dtype="auto",
        device_map="auto" if torch.cuda.is_available() else None
    )

    _llm_pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tok,
        max_new_tokens=min(MAX_NEW_TOKENS, 256),
        repetition_penalty=1.05,
        no_repeat_ngram_size=3,
        eos_token_id=tok.eos_token_id,
    )
    return _llm_pipe

SYSTEM_PROMPT = (
    "You are a concise, helpful assistant. Only use the provided PDF context to answer. "
    "If the answer is not clearly contained in the context, reply: 'I don't know based on the PDF.' "
    "Cite page numbers if present in the snippets (e.g., [Page X])."
)

def _build_prompt(question: str, context: str) -> str:
    return (
        f"<s>[SYSTEM]\n{SYSTEM_PROMPT}\n</s>\n"
        f"[CONTEXT]\n{context}\n\n"
        f"[USER]\n{question}\n\n[ASSISTANT]"
    )

def _truncate_context(context: str, max_chars: int = 4000) -> str:
    return context[:max_chars]

def answer_with_llm(llm_pipe, question: str, context: str) -> str:
    prompt = _build_prompt(question, _truncate_context(context))
    out = llm_pipe(prompt, num_return_sequences=1, do_sample=False)[0]["generated_text"]
    # Try to strip the prompt from the front if model echoes
    assistant_prefix = "[ASSISTANT]"
    if assistant_prefix in out:
        answer = out.split(assistant_prefix)[-1].strip()
        return answer
    else:
        # If we can't find the assistant prefix, return everything after the prompt
        return out[len(prompt):].strip() if out.startswith(prompt) else out.strip()
