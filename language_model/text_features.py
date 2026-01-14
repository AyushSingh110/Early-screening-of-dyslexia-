import torch
import math
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# ------------------------------
# Load frozen LM (once)
# ------------------------------
_tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
_model.eval()
_model.requires_grad_(False)

# ------------------------------
# Perplexity
# ------------------------------
def compute_perplexity(text: str) -> float:
    if len(text.split()) < 20:
        return None

    encodings = _tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = _model(
            **encodings,
            labels=encodings["input_ids"]
        )
        loss = outputs.loss

    return math.exp(loss.item())

# ------------------------------
# Raw linguistic features
# ------------------------------
def compute_raw_language_features(text: str) -> dict:
    words = text.lower().split()
    total_words = len(words)

    if total_words < 5:
        return None

    spelling_error_rate = sum(
        1 for w in words if not w.isalpha()
    ) / total_words

    non_word_ratio = spelling_error_rate
    repetition_score = len(words) / len(set(words)) - 1
    repetition_score = max(0.0, repetition_score)

    avg_word_length = sum(len(w) for w in words) / total_words
    rare_word_ratio = 0.15  # placeholder (OK for screening)

    lm_perplexity = compute_perplexity(text)
    if lm_perplexity is None:
        return None

    return {
        "spelling_error_rate": spelling_error_rate,
        "non_word_ratio": non_word_ratio,
        "phonetic_error_ratio": 0.3,   # placeholder (acceptable for now)
        "repetition_score": repetition_score,
        "avg_word_length": avg_word_length,
        "word_count": total_words,
        "rare_word_ratio": rare_word_ratio,
        "lm_perplexity": lm_perplexity
    }
