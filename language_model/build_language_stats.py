from datasets import load_dataset
from collections import Counter
import pickle
import re
import os


# CONFIG

OUTPUT_PATH = "language_model/word_freq.pkl"
MAX_DOCS = 20000   


# LOAD DATASET (STREAMING)

print("Loading Wikipedia dataset (streaming)...")

dataset = load_dataset(
    "wikimedia/wikipedia",
    "20231101.en",
    streaming=True
)

print("Dataset stream ready")


# TEXT CLEANING

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.split()

6
# BUILD WORD FREQUENCY

word_freq = Counter()

print("Building word frequency dictionary...")

for i, sample in enumerate(dataset["train"]):
    words = tokenize(sample["text"])
    word_freq.update(words)

    if i % 2000 == 0 and i > 0:
        print(f"Processed {i} documents")

    if i >= MAX_DOCS:
        break


# SAVE

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

with open(OUTPUT_PATH, "wb") as f:
    pickle.dump(word_freq, f)

print("Word frequency dictionary saved")
print("Unique words:", len(word_freq))

