def compute_raw_language_features(text: str) -> dict:
    """
    Converts OCR text into raw linguistic features
    """

    words = text.lower().split()
    total_words = len(words)

    if total_words == 0:
        return None

    # VERY SIMPLE placeholders 
    spelling_error_rate = sum(1 for w in words if not w.isalpha()) / total_words
    non_word_ratio = spelling_error_rate
    phonetic_error_ratio = 0.3   
    repetition_score = 0.2      
    avg_word_length = sum(len(w) for w in words) / total_words
    rare_word_ratio = 0.15       

    return {
        "spelling_error_rate": spelling_error_rate,
        "non_word_ratio": non_word_ratio,
        "phonetic_error_ratio": phonetic_error_ratio,
        "repetition_score": repetition_score,
        "avg_word_length": avg_word_length,
        "word_count": total_words,
        "rare_word_ratio": rare_word_ratio
    }
