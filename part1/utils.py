import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer

random.seed(0)


import random
import re
import nltk
from nltk.corpus import wordnet as wn


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.

KEYBOARD_NEIGHBORS = {
    "q": "wa",
    "w": "qes",
    "e": "wrd",
    "r": "etf",
    "t": "ryg",
    "y": "tuh",
    "u": "yij",
    "i": "uok",
    "o": "ipl",
    "p": "o",
    "a": "qsz",
    "s": "awdx",
    "d": "sefc",
    "f": "drgv",
    "g": "fthb",
    "h": "gyjn",
    "j": "hukm",
    "k": "jil",
    "l": "ko",
    "z": "ax",
    "x": "zsc",
    "c": "xdv",
    "v": "cfb",
    "b": "vgn",
    "n": "bhm",
    "m": "nj",
}

def get_synonym(word, tag):
    # Convert the sentence's POS tag to a WordNet POS.
    pos_mapping = {
        "J": wn.ADJ,
        "N": wn.NOUN,
        "V": wn.VERB,
        "R": wn.ADV,
    }
    pos = pos_mapping.get(tag[0])
    if pos is None:
        return None
    candidates = set()

    for synset in wn.synsets(word.lower(), pos=pos):
        for lemma in synset.lemmas():
            candidate = lemma.name()
            # Use single alphabetic words; exclude the original word.
            if candidate.isalpha() and candidate.lower() != word.lower():
                candidates.add(candidate.lower())

    if not candidates:
        return None
    # Sorting makes seeded random selection reproducible.
    synonym = random.choice(sorted(candidates))
    if word.isupper():
        synonym = synonym.upper()
    elif word[0].isupper():
        synonym = synonym.capitalize()

    return synonym

def introduce_multiple_typos(word):
    positions = [
        i for i, char in enumerate(word)
        if char.lower() in KEYBOARD_NEIGHBORS
    ]
    if not positions:
        return word

    # 4–5 characters: one typo; 6 or more: one or two.
    num_typos = 1 if len(word) <= 5 else random.randint(1, 2)
    num_typos = min(num_typos, len(positions))
    # Choose distinct positions so a character is changed only once.
    selected_positions = random.sample(positions, num_typos)
    characters = list(word)

    for i in selected_positions:
        original = characters[i]
        replacement = random.choice(
            KEYBOARD_NEIGHBORS[original.lower()]
        )
        if original.isupper():
            replacement = replacement.upper()
        characters[i] = replacement

    return "".join(characters)

def drop_random_char(word):
    if len(word) <= 3:
        return word

    # Keep the first and last characters.
    position = random.randint(1, len(word) - 2)
    return word[:position] + word[position + 1:]


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation

    text = example["text"]
    # Keep whitespace and punctuation as separate tokens.
    tokens = re.findall(r"\s+|\w+|[^\w\s]", text)
    # Determine word classes using the original sentence.
    nonspace_positions = [
        i for i, token in enumerate(tokens)
        if not token.isspace()
    ]
    tagged_tokens = nltk.pos_tag(
        [tokens[i] for i in nonspace_positions]
    )
    tags = {
        i: tag
        for i, (_, tag) in zip(nonspace_positions, tagged_tokens)
    }

    synonym_prob = 0.30
    typo_prob = 0.40
    drop_prob = 0.10

    for i, token in enumerate(tokens):
        # Only modify alphabetic words longer than 3 characters.
        if not token.isalpha() or len(token) <= 3:
            continue
        r = random.random()
        if r < synonym_prob:
            synonym = get_synonym(token, tags[i])
            if synonym is not None:
                tokens[i] = synonym
        elif r < synonym_prob + typo_prob:
            tokens[i] = introduce_multiple_typos(token)
        elif r < synonym_prob + typo_prob + drop_prob:
            tokens[i] = drop_random_char(token)
        # Otherwise, keep the original token unchanged.
    example["text"] = "".join(tokens)

    ##### YOUR CODE ENDS HERE ######
    return example
