import os
import re
import numpy as np
import enchant
from metaphone import doublemetaphone
import pickle

# PATHS
BASE_DATA_DIR = "../data/language/raw"
DYSLEXIC_DIR = os.path.join(BASE_DATA_DIR, "dyslexic_like")
NORMAL_DIR = os.path.join(BASE_DATA_DIR, "normal")

OUTPUT_DIR = "../data/language/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#Load the word frequency
WORD_FREQ_PATH = "word_freq.pkl"

with open(WORD_FREQ_PATH, "rb") as f:
    WORD_FREQ = pickle.load(f)

# TOOLS
dictionary = enchant.Dict("en_US")

def tokenize(text):
    return re.findall(r"[a-z]+", text.lower())


# FEATURE FUNCTIONS
def spelling_error_rate(words):
    errors = [w for w in words if not dictionary.check(w)]
    return len(errors) / max(len(words), 1)

def non_word_ratio(words):
    non_words = [w for w in words if not dictionary.check(w)]
    return len(non_words) / max(len(words), 1)

def phonetic_error_ratio(words):
    correct_words = [w for w in words if dictionary.check(w)]
    error_words = [w for w in words if not dictionary.check(w)]

    if not error_words or not correct_words:
        return 0.0

    correct_meta = set(doublemetaphone(w)[0] for w in correct_words)

    phonetic_matches = sum(
        1 for w in error_words if doublemetaphone(w)[0] in correct_meta
    )

    return phonetic_matches / len(error_words)

def repetition_score(words):
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
    return repeats / max(len(words), 1)

def average_word_length(words):
    return np.mean([len(w) for w in words]) if words else 0

def rare_word_ratio(words, word_freq, threshold=5):
    rare_words = [w for w in words if word_freq.get(w, 0) < threshold]
    return len(rare_words) / max(len(words), 1)


# FEATURE EXTRACTION
def extract_features_from_text(text):
    words = tokenize(text)

    return [
        spelling_error_rate(words),
        non_word_ratio(words),
        phonetic_error_ratio(words),
        repetition_score(words),
        average_word_length(words),
        len(words),
        rare_word_ratio(words, WORD_FREQ)
    ]


# DATASET CREATION
X = []
y = []
def process_folder(folder_path, label):
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    print(f"Processing {len(files)} files from {folder_path}")

    for i, file in enumerate(files, 1):
        if i % 5 == 0:
            print(f"  Processed {i}/{len(files)} files")

        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read()

        features = extract_features_from_text(text)
        X.append(features)
        y.append(label)



if __name__ == "__main__":

    # dyslexic_like (label 1)
    process_folder(DYSLEXIC_DIR, 1)

    # normal (label 0)
    process_folder(NORMAL_DIR, 0)

    X = np.array(X)
    y = np.array(y)

    # SAVE
    np.save(os.path.join(OUTPUT_DIR, "X_features.npy"), X)
    np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)

    print("Feature extraction complete")
    print("X shape:", X.shape)
    print("y shape:", y.shape)

