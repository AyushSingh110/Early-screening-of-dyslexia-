import os
import re
import numpy as np
import enchant
from metaphone import doublemetaphone


# PATHS
BASE_DATA_DIR = "../data/language/raw"
DYSLEXIC_DIR = os.path.join(BASE_DATA_DIR, "dyslexic_like")
NORMAL_DIR = os.path.join(BASE_DATA_DIR, "normal")

OUTPUT_DIR = "../data/language/processed"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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
    errors = [w for w in words if not dictionary.check(w)]
    if not errors:
        return 0.0

    phonetic_matches = 0
    for w in errors:
        w_meta = doublemetaphone(w)[0]
        for v in words:
            if dictionary.check(v) and doublemetaphone(v)[0] == w_meta:
                phonetic_matches += 1
                break

    return phonetic_matches / max(len(errors), 1)

def repetition_score(words):
    repeats = sum(1 for i in range(1, len(words)) if words[i] == words[i-1])
    return repeats / max(len(words), 1)

def average_word_length(words):
    return np.mean([len(w) for w in words]) if words else 0


# FEATURE EXTRACTION
def extract_features_from_text(text):
    words = tokenize(text)

    return [
        spelling_error_rate(words),
        non_word_ratio(words),
        phonetic_error_ratio(words),
        repetition_score(words),
        average_word_length(words),
        len(words)
    ]

# DATASET CREATION
X = []
y = []
def process_folder(folder_path, label):
    files = os.listdir(folder_path)
    print(f"Processing {len(files)} files from {folder_path}")

    for i, file in enumerate(files):
        if not file.endswith(".txt"):
            continue

        if i % 5 == 0:
            print(f"  Processed {i}/{len(files)} files")

        with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
            text = f.read()

        features = extract_features_from_text(text)
        X.append(features)
        y.append(label)


# dyslexic_like(label 1)
process_folder(DYSLEXIC_DIR, 1)

# normal(label 0)
process_folder(NORMAL_DIR, 0)

X = np.array(X)
y = np.array(y)


# SAVE
np.save(os.path.join(OUTPUT_DIR, "X_features.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y_labels.npy"), y)

print("Feature extraction complete")
print("X shape:", X.shape)
print("y shape:", y.shape)
