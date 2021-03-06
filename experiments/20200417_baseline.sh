#!/bin/bash
#

TRAIN_BASE=../naacl_flp
PREDS_BASE=experiments/2020/0421_baseline
EVALS_BASE="$PREDS_BASE"
mkdir -p "$PREDS_BASE"

function run {
   TF_CPP_MIN_LOG_LEVEL="3" python3 ./run.py \
     --train_corpus "$TRAIN_BASE/$TRAIN_CORPUS" \
     --train_labels "$TRAIN_BASE/$TRAIN_LABELS" \
     --test_corpus  "$TRAIN_BASE/$TEST_CORPUS" \
     --test_labels  "$TRAIN_BASE/$TEST_LABELS" \
     --pos \
     --embeddings $EMBEDDINGS $PREDICTARGS $EVALARGS \
     --n_splits "$N_SPLITS" \
     $EXTRAARGS -v 2>&1 | tee "$(dirname "$PREDFILE")/$(basename "$PREDFILE" .pred)".stdout
}

N_SPLITS=10

TRAIN_CORPUS=vuamc_corpus_train.csv
TRAIN_LABELS=all_pos_tokens.csv
TEST_CORPUS=vuamc_corpus_test.csv
TEST_LABELS=all_pos_tokens_test.csv

EMBEDDINGS=../data/wiki.en.bin.gz
PREDFILE="$PREDS_BASE"/vuamc-all_pos-wiki.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/wiki.en.bin.gz ../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-wiki-low-med-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-low.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-med.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-low-med-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-lmh.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-low-med-high-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-lmh-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run


TRAIN_LABELS=verb_tokens.csv
TEST_LABELS=verb_tokens_test.csv

EMBEDDINGS=../data/wiki.en.bin.gz
PREDFILE="$PREDS_BASE"/vuamc-verb-wiki.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/wiki.en.bin.gz ../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-verb-wiki-low-med-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-verb-low-med-high-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-verb-lmh-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run



TRAIN_CORPUS=toefl_sharedtask_dataset_train.csv
TRAIN_LABELS=toefl_skll_train_all_pos_tokens.csv
TEST_CORPUS=toefl_sharedtask_dataset_test.csv
TEST_LABELS=toefl_sharedtask_evaluation_kit/toefl_all_pos_test_tokens.csv

EMBEDDINGS=../data/wiki.en.bin.gz
PREDFILE="$PREDS_BASE"/toefl-all_pos-wiki.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/wiki.en.bin.gz ../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-wiki-low-med-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-low.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-med.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-low-med-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-lmh.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-low-med-high-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-lmh-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run


TRAIN_LABELS=toefl_skll_train_verb_tokens.csv
TEST_LABELS=toefl_sharedtask_evaluation_kit/toefl_verb_test_tokens.csv

EMBEDDINGS=../data/wiki.en.bin.gz
PREDFILE="$PREDS_BASE"/toefl-verb-wiki.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/wiki.en.bin.gz ../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-verb-wiki-low-med-high.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_low_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_med_min1_dim50.bin.gz ../data/NLI_2013/NLI_2013_high_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-verb-low-med-high-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-verb-lmh-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run



TRAIN_CORPUS=vuamc_corpus_train.csv
TRAIN_LABELS=all_pos_tokens.csv
TEST_CORPUS=vuamc_corpus_test.csv
TEST_LABELS=all_pos_tokens_test.csv

EMBEDDINGS="../data/bnc2_tt2.bin.gz ../data/ententen13_tt2_1.bin.gz ../data/ukwac/ukwac.bin.gz ../data/wiki.en.bin.gz"
PREDFILE="$PREDS_BASE"/vuamc-all_pos-bnc-entt-ukwac-wiki.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

TRAIN_CORPUS=toefl_sharedtask_dataset_train.csv
TRAIN_LABELS=toefl_skll_train_all_pos_tokens.csv
TEST_CORPUS=toefl_sharedtask_dataset_test.csv
TEST_LABELS=toefl_sharedtask_evaluation_kit/toefl_all_pos_test_tokens.csv

EMBEDDINGS="../data/bnc2_tt2.bin.gz ../data/ententen13_tt2_1.bin.gz ../data/ukwac/ukwac.bin.gz ../data/wiki.en.bin.gz"
PREDFILE="$PREDS_BASE"/toefl-all_pos-bnc-entt-ukwac-wiki.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
PREDICTARGS="--predict --predictfile $PREDFILE"
EVALARGS="--evaluate --evaluatefile $EVALFILE"
#( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run
