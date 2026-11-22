import re
from pathlib import Path
from statistics import mean
from transformers import T5TokenizerFast

# ADDED: Reuse the extended tokenizer for the post-preprocessing statistics.
from tokenizer_utils import build_tokenizer


DATA_DIR = Path(__file__).resolve().parent / "data"

MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512


def read_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\r\n") for line in f]

def preprocess_question(text):
    # Remove leading/trailing whitespace; do not add a prefix.
    return text.strip()

def preprocess_sql(text):
    # Remove leading/trailing whitespace and collapse repeated whitespace.
    return re.sub(r"\s+", " ", text.strip())

def token_statistics(texts, tokenizer, max_length=None):
    lengths = []
    vocabulary = set()

    for text in texts:
        options = {
            "add_special_tokens": True,
            "truncation": max_length is not None,
        }

        if max_length is not None:
            options["max_length"] = max_length

        token_ids = tokenizer.encode(text, **options)

        lengths.append(len(token_ids))
        vocabulary.update(token_ids)

    return mean(lengths) if lengths else 0.0, len(vocabulary)

def compute_statistics(
    questions,
    queries,
    tokenizer,
    max_source_length=None,
    max_target_length=None,
):
    if len(questions) != len(queries):
        raise ValueError("Question and SQL counts do not match.")

    question_length, question_vocab = token_statistics(
        questions, tokenizer, max_source_length
    )
    sql_length, sql_vocab = token_statistics(
        queries, tokenizer, max_target_length
    )

    return {
        "Number of examples": len(questions),
        "Mean sentence length": question_length,
        "Mean SQL query length": sql_length,
        "Vocabulary size (natural language)": question_vocab,
        "Vocabulary size (SQL)": sql_vocab,
    }

def print_table(title, train_stats, dev_stats):
    print(f"\n{title}")
    print("| Statistics Name | Train | Dev |")
    print("|---|---:|---:|")

    for name in train_stats:
        train_value = train_stats[name]
        dev_value = dev_stats[name]

        if name.startswith("Mean"):
            print(f"| {name} | {train_value:.2f} | {dev_value:.2f} |")
        else:
            print(f"| {name} | {train_value} | {dev_value} |")

def main():
    tokenizer = T5TokenizerFast.from_pretrained("google-t5/t5-small")
    after_tokenizer = build_tokenizer()  # ADDED: Table 1 keeps the default tokenizer.

    before = {}
    after = {}

    for split in ["train", "dev"]:
        questions = read_lines(DATA_DIR / f"{split}.nl")
        queries = read_lines(DATA_DIR / f"{split}.sql")

        # Table 1: original text, without truncation.
        before[split] = compute_statistics(
            questions,
            queries,
            tokenizer,
        )

        processed_questions = [
            preprocess_question(text) for text in questions
        ]
        processed_queries = [
            preprocess_sql(text) for text in queries
        ]

        # MODIFIED: Table 2 uses the extended tokenizer and the 512-token training limit.
        after[split] = compute_statistics(
            processed_questions,
            processed_queries,
            after_tokenizer,  # MODIFIED: Match the tokenizer actually used for training.
            max_source_length=MAX_SOURCE_LENGTH,
            max_target_length=MAX_TARGET_LENGTH,
        )

    print_table(
        "Table 1: Before preprocessing",
        before["train"],
        before["dev"],
    )

    print_table(
        "Table 2: After preprocessing",
        after["train"],
        after["dev"],
    )

if __name__ == "__main__":
    main()
