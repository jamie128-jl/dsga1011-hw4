# ADDED FILE: Shared tokenizer setup; this file is not part of the professor's starter code.
from transformers import AddedToken, T5TokenizerFast

MODEL_NAME = "google-t5/t5-small"

SQL_TOKENS = [
    "<",
    ">",
    "<=",
    ">=",
    "!=",
    "<>",
    # Keep the leading space so decoded phrases do not stick to the preceding column name.
    " IS NULL",
    " IS NOT NULL",
]

def build_tokenizer():
    tokenizer = T5TokenizerFast.from_pretrained(MODEL_NAME)
    added_tokens = [
        AddedToken(text, normalized=False, special=False)
        for text in SQL_TOKENS
    ]
    tokenizer.add_tokens(added_tokens, special_tokens=False)
    return tokenizer
