# DS-GA 1011 Homework 4

BERT sentiment classification and T5 text-to-SQL fine-tuning for Fall 2026.
This repository contains the implemented Part 1 and Part 2 source code. Model checkpoints are shared separately through Google Drive; prediction files are submitted separately to Gradescope.

The assignment starter code and data are provided by [gregdurrett/nyu-dsga1011-a4](https://github.com/gregdurrett/nyu-dsga1011-a4). The source files retain the starter comments and the implementation annotations.

## Repository structure

```text
part1/                  BERT training, transformations, and augmentation
part2/                  T5 data loading, tokenization, training, and evaluation
  tokenizer_utils.py    Shared T5 tokenizer with added SQL tokens
  data_stats.py         Before/after preprocessing statistics
```

Use separate Python 3.11 environments because the two parts require different Transformers versions. Run training on a CUDA GPU. The reported T5 experiments used an NVIDIA L4 in Google Colab.

## Part 1

```bash
conda create -n dsga1011-hw4-part1 python=3.11
conda activate dsga1011-hw4-part1
cd part1
python -m pip install -r requirements.txt
python -m nltk.downloader wordnet punkt averaged_perceptron_tagger
```

The IMDB dataset is downloaded through Hugging Face Datasets.

Debug training:

```bash
python -u main.py --train --eval --debug_train
```

Full training and original-test evaluation:

```bash
python -u main.py --train --eval --learning_rate 2e-5 --batch_size 16 --num_epochs 3
```

Evaluate the Q1 model on transformed test reviews:

```bash
python -u main.py --eval_transformed --model_dir out --batch_size 16
```

Train a fresh BERT model on the original 25,000 reviews plus 5,000 transformed training reviews, then evaluate it:

```bash
python -u main.py --train_augmented --eval_transformed --learning_rate 2e-5 --batch_size 16 --num_epochs 3
python -u main.py --eval --model_dir out_augmented --batch_size 16
```

For each eligible alphabetic word longer than three characters, the transformation selects synonym replacement (30%), keyboard-neighbor typos (40%), character deletion (10%), or no modification (20%). Labels are retained.

| Model | Original test accuracy | Transformed test accuracy |
|---|---:|---:|
| Original training data | 93.700% | 81.748% |
| Augmented training data | 93.516% | 85.784% |

## Part 2

### Environment and data

From the repository root:

```bash
conda create -n dsga1011-hw4-part2 python=3.11
conda activate dsga1011-hw4-part2
cd part2
python -m pip install -r requirements.txt
python -m nltk.downloader punkt
```

The dataset and SQLite database are distributed with the course starter repository. To restore them from the repository root:

```bash
git clone https://github.com/gregdurrett/nyu-dsga1011-a4.git starter-data
cp -R starter-data/part2/data part2/data
mkdir -p part2/records
cp starter-data/part2/records/ground_truth_dev.pkl part2/records/
```

The expected files are `data/train.nl`, `data/train.sql`, `data/dev.nl`, `data/dev.sql`, `data/test.nl`, `data/flight_database.db`, and `records/ground_truth_dev.pkl`, relative to `part2/`.

### Processing and model

Questions are stripped of leading/trailing whitespace. SQL queries are stripped and repeated whitespace is collapsed. No task prefix or schema is added. The shared `google-t5/t5-small` tokenizer handles both question and SQL sequences, capped at 512 tokens each. It includes added ordinary SQL tokens for `<`, `>`, `<=`, `>=`, `!=`, `<>`, ` IS NULL`, and ` IS NOT NULL`. The model embeddings are resized to the tokenizer vocabulary.

Dynamic right padding uses PAD ID 0. Decoder inputs are shifted SQL targets with PAD prepended. Cross-entropy excludes padding targets. All model parameters are fine-tuned. Dev and test generation use greedy decoding with `num_beams=1`, `do_sample=False`, and at most 512 new tokens.

Compute Q4 statistics from `part2/`:

```bash
python data_stats.py
```

### Training experiments

Run each command from `part2/`. Each invocation independently loads the pretrained T5 checkpoint. The first experiment used learning rate `1e-4` and name `lr1e4_seed42`; the selected experiment uses `3e-4` and name `lr3e4_seed42`. All other settings are held fixed.

```bash
python -u -c "from train_t5 import main; from utils import set_random_seeds; set_random_seeds(42); main()" \
    --finetune \
    --experiment_name lr3e4_seed42 \
    --learning_rate 3e-4 \
    --optimizer_type AdamW \
    --weight_decay 0.01 \
    --scheduler_type cosine \
    --num_warmup_epochs 1 \
    --batch_size 8 \
    --test_batch_size 8 \
    --max_n_epochs 15 \
    --patience_epochs 5
```

The training script evaluates the dev set after each epoch. Early stopping uses dev Record F1 with patience 5. The best checkpoint is restored for final dev evaluation and test generation. Checkpoints and their tokenizer are saved under `checkpoints/ft_experiments/<experiment_name>/best/`; the latest epoch is saved under `last/`.

Generated queries are saved to `results/t5_ft_<experiment_name>_{dev,test}.sql`. Associated records are saved to `records/t5_ft_<experiment_name>_{dev,test}.pkl`. For submission, copy the selected test files to the required names `t5_ft_experiment_test.sql` and `t5_ft_experiment_test.pkl`.

### Results

All values below are percentages. The test scores are the official Gradescope results for the selected model.

| Split / experiment | Learning rate | Record F1 | Record EM | SQL Query EM |
|---|---:|---:|---:|---:|
| Dev / Experiment 1 | 1e-4 | 61.74 | 52.58 | 1.72 |
| Dev / selected model | 3e-4 | 77.26 | 74.68 | 2.36 |
| Test / selected model | 3e-4 | 74.22 | 71.99 | 0.00 |

To evaluate saved dev predictions, run from `part2/`:

```bash
python evaluate.py \
    --predicted_sql results/t5_ft_lr3e4_seed42_dev.sql \
    --predicted_records records/t5_ft_lr3e4_seed42_dev.pkl \
    --development_sql data/dev.sql \
    --development_records records/ground_truth_dev.pkl
```

Test ground-truth answers are held out. Test performance is obtained through the assignment evaluator.
