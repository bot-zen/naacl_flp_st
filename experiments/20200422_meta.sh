#!/bin/bash
#

TRAIN_BASE=../naacl_flp
PREDS_BASE=experiments/2020/$(basename "$0" .sh)
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
PREDICTARGS=""

TRAIN_CORPUS=toefl_sharedtask_dataset_train.csv
TRAIN_LABELS=toefl_skll_train_all_pos_tokens.csv
TEST_CORPUS=toefl_sharedtask_dataset_test.csv
TEST_LABELS=toefl_sharedtask_evaluation_kit/toefl_all_pos_test_tokens.csv

#EMBEDDINGS=../data/bnc2_tt2.bin.gz
EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"

EXTRAARGS=""
PREDFILE="$PREDS_BASE"/toefl-all_pos-lmh-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$EVALFILE" ] && echo "already exists: $EVALFILE" ) || run

EXTRAARGS="--metadata ./experiments/2020/toefl_metadata.csv"
PREDFILE="$PREDS_BASE"/toefl-all_pos-lmh-bnc-meta_cat_prof.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$EVALFILE" ] && echo "already exists: $EVALFILE" ) || run

TRAIN_LABELS=toefl_skll_train_verb_tokens.csv
TEST_LABELS=toefl_sharedtask_evaluation_kit/toefl_verb_test_tokens.csv

EXTRAARGS=""
PREDFILE="$PREDS_BASE"/toefl-verb-lmh-bnc.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$EVALFILE" ] && echo "already exists: $EVALFILE" ) || run

EXTRAARGS="--metadata ./experiments/2020/toefl_metadata.csv"
PREDFILE="$PREDS_BASE"/toefl-verb-lmh-bnc-meta_cat_prof.pred
EVALFILE="$EVALS_BASE"/$(basename "$PREDFILE" .pred).eval
EVALARGS="--evaluate --evaluatefile $EVALFILE"
( [ -s "$EVALFILE" ] && echo "already exists: $EVALFILE" ) || run
