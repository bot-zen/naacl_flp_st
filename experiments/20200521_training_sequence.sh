#!/bin/bash
#

TRAIN_BASE=../naacl_flp/vuamc_toefl-combined
PREDS_BASE=experiments/2020/0521_training_sequence
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

EMBEDDINGS="../data/NLI_2013/NLI_2013_all_min1_dim50.bin.gz ../data/bnc2_tt2.bin.gz"
EVALARGS=""
N_SPLITS=0
EXTRAARGS="--no-shuffle"


for N in $(seq -f"%01g" 0 9)
do
    TRAIN_IDS="$(for S in $(seq $(( (N+1) )) $(( (N+9 ) ))); do echo 0$(( S % 10 )); done)"
    TEST_ID=$(( N % 10 ))

    for POS in all_pos verb
    do
		PREDFILE="$PREDS_BASE"/toefl-vuamc_on_toefl-${POS}-${TEST_ID}.pred
		PREDICTARGS="--predict --predictfile $PREDFILE"
		TRAIN_CORPUS="toefl_vuamc-cvtrain0${TEST_ID}.csv"
		TRAIN_LABELS="toefl_vuamc-train_tokens-${POS}-cvtrain0${TEST_ID}.csv"
		TEST_CORPUS="toefl-train_dataset-split0${TEST_ID}.csv"
		TEST_LABELS="toefl-train_tokens-${POS}-split0${TEST_ID}.csv"
		( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

		PREDFILE="$PREDS_BASE"/toefl-vuamc_on_vuamc-${POS}-${TEST_ID}.pred
		PREDICTARGS="--predict --predictfile $PREDFILE"
		TRAIN_CORPUS="toefl_vuamc-cvtrain0${TEST_ID}.csv"
		TRAIN_LABELS="toefl_vuamc-train_tokens-${POS}-cvtrain0${TEST_ID}.csv"
		TEST_CORPUS="vuamc-train_dataset-split0${TEST_ID}.csv"
		TEST_LABELS="vuamc-train_tokens-${POS}-split0${TEST_ID}.csv"
		( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

		PREDFILE="$PREDS_BASE"/vuamc-toefl_on_toefl-${POS}-${TEST_ID}.pred
		PREDICTARGS="--predict --predictfile $PREDFILE"
		TRAIN_CORPUS="vuamc_toefl-cvtrain0${TEST_ID}.csv"
		TRAIN_LABELS="vuamc_toefl-train_tokens-${POS}-cvtrain0${TEST_ID}.csv"
		TEST_CORPUS="toefl-train_dataset-split0${TEST_ID}.csv"
		TEST_LABELS="toefl-train_tokens-${POS}-split0${TEST_ID}.csv"
		( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run

		PREDFILE="$PREDS_BASE"/vuamc-toefl_on_vuamc-${POS}-${TEST_ID}.pred
		PREDICTARGS="--predict --predictfile $PREDFILE"
		TRAIN_CORPUS="vuamc_toefl-cvtrain0${TEST_ID}.csv"
		TRAIN_LABELS="vuamc_toefl-train_tokens-${POS}-cvtrain0${TEST_ID}.csv"
		TEST_CORPUS="vuamc-train_dataset-split0${TEST_ID}.csv"
		TEST_LABELS="vuamc-train_tokens-${POS}-split0${TEST_ID}.csv"
		( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run


		EXTRAARGS="--shuffle"
		PREDFILE="$PREDS_BASE"/toefl-vuamc-shuff_on_toefl-${POS}-${TEST_ID}.pred
		PREDICTARGS="--predict --predictfile $PREDFILE"
		TRAIN_CORPUS="toefl_vuamc-cvtrain0${TEST_ID}.csv"
		TRAIN_LABELS="toefl_vuamc-train_tokens-${POS}-cvtrain0${TEST_ID}.csv"
		TEST_CORPUS="toefl-train_dataset-split0${TEST_ID}.csv"
		TEST_LABELS="toefl-train_tokens-${POS}-split0${TEST_ID}.csv"
		( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run


		EXTRAARGS="--shuffle"
		PREDFILE="$PREDS_BASE"/toefl-vuamc-shuff_on_vuamc-${POS}-${TEST_ID}.pred
		PREDICTARGS="--predict --predictfile $PREDFILE"
		TRAIN_CORPUS="toefl_vuamc-cvtrain0${TEST_ID}.csv"
		TRAIN_LABELS="toefl_vuamc-train_tokens-${POS}-cvtrain0${TEST_ID}.csv"
		TEST_CORPUS="vuamc-train_dataset-split0${TEST_ID}.csv"
		TEST_LABELS="vuamc-train_tokens-${POS}-split0${TEST_ID}.csv"
		( [ -s "$PREDFILE" ] && echo "already exists: $PREDFILE" ) || run
    done
done

TRAIN_BASE=../naacl_flp
PREDICTARGS=""
N_SPLITS=10
EXTRAARGS="--shuffle"
for TASK in verb all_pos
do
    TRAIN_CORPUS=vuamc_corpus_train.csv
    TRAIN_LABELS=all_pos_tokens.csv
    EVALFILE="$EVALS_BASE"/vuamc-shuff_on_vuamc-${TASK}.eval
    EVALARGS="--evaluate --evaluatefile $EVALFILE"
    ( [ -s "$EVALFILE" ] && echo "already exists: $EVALFILE" ) || run

    TRAIN_CORPUS=toefl_sharedtask_dataset_train.csv
    TRAIN_LABELS=toefl_skll_train_all_pos_tokens.csv
    EVALFILE="$EVALS_BASE"/toefl-shuff_on_toefl-${TASK}.eval
    EVALARGS="--evaluate --evaluatefile $EVALFILE"
    ( [ -s "$EVALFILE" ] && echo "already exists: $EVALFILE" ) || run
done





