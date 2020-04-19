#!/usr/bin/env python3
"""
Eurac Research / Masaryk University and Alpen-Adria University contribution
to the NAACL-FLP-shared-task 2018 and 2020.

Author: egon w. stemle <egon.stemle@eurac.edu>
Contributor: alexander onysko <alexander.onysko@aau.at>
"""

from collections import Counter, OrderedDict
from sys import exit, stdout

import argparse
import csv
import logging

from nltk import pos_tag

import numpy as np
import pandas

from tensorflow import keras
from tensorflow.keras import backend as k
from tensorflow.keras.layers import Masking, LSTM, Dense, TimeDistributed, Bidirectional, concatenate
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import tensorflow_addons as tfa

DESCRIPTION = __doc__
EPILOG = None

class Corpus():
    """
    Represent flpst files like:

    --- 8< --- vuamc_corpus_train.csv --- 8< ---
    "txt_id","sentence_id","sentence_txt"
"a1e-fragment01","1","Latest corporate unbundler M_reveals laid-back M_approach : Roland Franklin , who is M_leading a 697m pound break-up bid for DRG , talks M_to Frank Kane"
"a1e-fragment01","2","By FRANK KANE"
    ...
    --- >8 ---

    --- 8< --- all_pos_tokens.csv --- 8< ---
    a1h-fragment06_114_1,0
    a1h-fragment06_114_3,0
    a1h-fragment06_114_4,0
    a1h-fragment06_115_2,0
    a1h-fragment06_115_3,0
    a1h-fragment06_116_1,0
    a1h-fragment06_116_2,0
    a1h-fragment06_116_5,1
    --- >8 ---

    and their 'test' counterparts without class labels (or 'M_'-information).

    Parameters:
        vuamc_fn: file name string
        tokens_fn: file name string
        mode: "train" will read labels from tokens_fn

    """
    def __init__(self, vuamc_fn, tokens_fn, metadata_fn=False, mode="train",
                 sanity_check=False):
        self.log = logging.getLogger(type(self).__name__)
        self.vuamc_delimiter = self.tokens_delimiter = ","
        self.vuamc_quotechar = self.tokens_quotechar = '"'

        # set initial values for params
        self.vuamc_fn = vuamc_fn
        self.tokens_fn = tokens_fn
        self.mode = mode

        # load files
        self.vuamc = self._load_vuamc(self.vuamc_fn)
        self.tokens = self._load_tokens(self.tokens_fn)
        self.metadata = None
        if metadata_fn:
            self.metadata = self._load_metadata(metadata_fn)

        if sanity_check:
            self._sanity_check()

        self._sentences = None

    def _load_vuamc(self, fn):
        """
        self.vuamc['a1h-fragment06']['134']['tokens'][23]
        self.vuamc['a1h-fragment06']['134']['labels'][23]
        """
        data = OrderedDict()
        with open(fn) as csvfile:
            csvreader = csv.DictReader(csvfile, delimiter=self.vuamc_delimiter,
                                       quotechar=self.vuamc_quotechar)
            self.log.debug("Opening file: %s", fn)
            for row in csvreader:
                txt_id = row['txt_id']
                sentence_id = row['sentence_id']
                sentence_txt = row['sentence_txt']

                if txt_id not in data:
                    data[txt_id] = OrderedDict()

                if txt_id in data and sentence_id in data[txt_id]:
                    exit("Identical keys in line {}".format(csvreader.line_num))
                else:
                    data[txt_id][sentence_id] = OrderedDict()

                tokens = OrderedDict()
                labels = OrderedDict()
                for token_id, token in enumerate(
                        sentence_txt.strip().split(" ")):
                    if token.startswith("M_"):
                        token = token[2:]
                        labels[token_id+1] = 1
                    else:
                        labels[token_id+1] = 0
                    tokens[token_id+1] = token

                data[txt_id][sentence_id]['tokens'] = tokens
                data[txt_id][sentence_id]['labels'] = labels

            return data


    def _load_tokens(self, fn):
        """
        self.tokens['a1h-fragment06']['134'][23]
        """
        data = OrderedDict()
        with open(fn) as csvfile:
            csvreader = csv.reader(csvfile, delimiter=self.tokens_delimiter,
                                   quotechar=self.tokens_quotechar)
            self.log.debug("Opening file: %s", fn)
            for row in csvreader:
                txt_id, sentence_id, token_id = row[0].split('_')[0:3]

                if self.mode == "train":
                    label = int(row[1])
                    # ?FIXME:
                    # here, we are using the additional information from the
                    # filtered stop words. still, we are only using the test
                    # partition so, all in all we should be on the safe side.
                    label = self.vuamc[txt_id][sentence_id]['labels'][int(token_id)]
                else:
                    label = -1

                if txt_id not in data:
                    data[txt_id] = OrderedDict()

                if sentence_id not in data[txt_id]:
                    data[txt_id][sentence_id] = OrderedDict()

                if (txt_id in data and sentence_id in data[txt_id] and
                        int(token_id) in data[txt_id][sentence_id]):
                    exit("Identical keys in line {}".format(csvreader.line_num))

                data[txt_id][sentence_id][int(token_id)] = label

            return data

    def _load_metadata(self, fn):
        """Load corpus metadata per section/sentence/file.

        For the TOEFL corpus this might look like this:
            txt_id,txt_length,prompt,l1,proficiency
            1005892,234,P3,ARA,medium
            1012679,340,P3,ARA,high
        """
        logging.info("Loading metadata from {}...".format(fn))
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        datacsv = pandas.read_csv(fn)
        dataset = datacsv.values
        txt_ids = (dataset[:, 0]).astype(str)
        # txt_lengths = (dataset[:, 1]).astype(int)
        txt_cats = (dataset[:, 2:]).astype(str)

        le = LabelEncoder()
        le_enc = le.fit_transform(txt_ids)

        ohe = OneHotEncoder()
        ohe_enc = ohe.fit_transform(txt_cats).toarray()

        # data_array = np.hstack((np.reshape(le_enc, (-1, 1)), ohe_enc))
        data_array = ohe_enc
        data = {}
        for row_id, txt_id in enumerate(txt_ids):
            data[txt_id] = data_array[row_id]
        logging.info("...done.")
        return data

    def _sanity_check(self):
        """
        Check that the 'txt_id, sentence_id, token_id, class_label'-s from the
        files:
            - vuamc_corpus.csv
            - ..._tokens.csv
        match.
        """
        for txt_id in self.tokens:
            for sentence_id in self.tokens[txt_id]:
                for token_id in self.tokens[txt_id][sentence_id]:
                    self.log.info(
                        "%s %d %d", " ".join([str(x) for x in [txt_id,
                                                               sentence_id,
                                                               token_id]]),
                        self.tokens[txt_id][sentence_id][token_id],
                        self.vuamc[txt_id][sentence_id]['labels'][token_id])
                    assert (self.tokens[txt_id][sentence_id][token_id] ==
                            self.vuamc[txt_id][sentence_id]['labels'][token_id])

    def sentence(self, txt_id, sentence_id):
        sentence = []
        for token_id in self.vuamc[txt_id][sentence_id]['tokens'].keys():
            if token_id in self.tokens[txt_id][sentence_id]:
                label = self.tokens[txt_id][sentence_id][token_id]
            else:
                label = 0
            sentence.append((self.vuamc[txt_id][sentence_id]['tokens'][token_id],
                             label))
        return sentence

    @property
    def sentences(self):
        """ Yield list (sentences) of tuples (word, label). """
        def populate_sentences():
            """ Helper to populate sentences. """
            for txt_id in self.tokens:
                for sentence_id in self.tokens[txt_id]:
                    yield self.sentence(txt_id, sentence_id)
        if self._sentences is None:
            self._sentences = list(populate_sentences())

        return self._sentences

    @staticmethod
    def X_y_sentence(model, sentence, maxlen=None):
        retval_X = []
        retval_y = []
        # ..._pads: list of padded entried - multiple if len(toks) > max_len
        sentence_toks_pads = Utils.pad_toks([tok[0] for tok in sentence],
                                            maxlen=maxlen)
        sentence_ys_pads = Utils.pad_toks([tok[1] for tok in sentence],
                                          value=0, maxlen=maxlen)
        for tmp_id, sentence_toks_pad in enumerate(sentence_toks_pads):
            sentence_retval = []
            sentence_retval.extend(Utils.toks2feat(sentence_toks_pad, model))
            retval_X.append(np.array(sentence_retval))
            retval_y.append(np.array(sentence_ys_pads[tmp_id]))
        return retval_X, retval_y

    def X_y(self, model, sentence=None, maxlen=None):
        retval_X = []
        retval_y = []
        if sentence is None:
            sentences = self.sentences
        else:
            sentences = [sentence]

        for _id, sent in enumerate(sentences):
            if _id % 500 == 0:
                self.log.info("Calculating X_y: sentence %d/%d",
                              _id, len(sentences))
            X, y = Corpus.X_y_sentence(model, sent, maxlen=maxlen)
            retval_X.extend(X)
            retval_y.extend(y)
        return retval_X, retval_y

    def X(self, model):
        X, _ = self.X_y(model)
        return X

    def y(self, model):
        _, y = self.X_y(model)
        return y

    def Xposs(self, categorical=False, maxlen=None):
        retval = []
        for sent in self.sentences:
            poss = [tag[1] for tag in pos_tag([tok[0] for tok in sent])]
            sentence_poss_pads = Utils.pad_toks(poss,
                                                value=0,
                                                maxlen=maxlen)
            for _id, sentence_poss_pad in enumerate(sentence_poss_pads):
                retval.append(Utils.poss2feat(sentence_poss_pad))
        retval = np.array(retval)
        if categorical:
            retval = to_categorical(retval, len(Utils.poss)+2)
        return retval

    def Xmetas(self, maxlen=None):
        retval = []
        for txt_id in self.tokens.keys():
            for sentence_id in self.tokens[txt_id]:
                metas = [self.metadata[txt_id]] * len(self.sentence(txt_id,
                                                                    sentence_id))
                metas_pads = Utils.pad_toks(
                    metas, value=np.array([0] * len(self.metadata[txt_id])),
                    maxlen=maxlen)
                retval.extend(metas_pads)
        return np.array(retval).astype(np.float32)

class Utils():
    """ Utility and helper functions. """
    padding_str = "__PADDING__"
    poss = ['$', "''", '(', ')', ',', '.', ':', 'CC', 'CD', 'DT', 'EX', 'FW',
            'IN', 'JJ', 'JJR', 'JJS', 'MD', 'NN', 'NNP', 'NNPS', 'NNS', 'PDT',
            'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'TO', 'UH', 'VB',
            'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

    @staticmethod
    def pos2feat(tok):
        try:
            # 1-based: reserve 0 for padding (and the last for unknown)
            return Utils.poss.index(tok) + 1
        except ValueError:
            return len(Utils.poss) + 1

    @staticmethod
    def poss2feat(toks):
        return [Utils.pos2feat(tok) for tok in toks]

    @staticmethod
    def toks2feat(toks, model, context_length=10):
        """
        Return the feature vectors of a list of tokens for a model.

        Make something of the subword information encoded in FastText word
        representations:
        https://radimrehurek.com/gensim/auto_examples/tutorials/run_fasttext.html
        """
        log = logging.getLogger(__name__)
        retval = []
        for tok_id, token in enumerate(toks):
            feat_token = None  # token to use to calculate the feature vector
            if token == Utils.padding_str:
                retval.append(model.vector_size * [0])
            else:
                if token in model.wv.vocab:
                    # the easy case: the token is in the model -> use it
                    feat_token = token
                elif token.lower() in model.wv.vocab:
                    # also quite easy: the lower-case version of the token is
                    # in the model -> use it
                    feat_token = token.lower()
                elif token in ['‘', '’', '—', '…']:
                    # cleaning up some common chars (missing from some models)
                    if token in ['‘', '’']:
                        subst = "'"
                    elif token == '—':
                        subst = "-"
                    elif token == '…':
                        subst = "..."

                    if subst in model.wv.vocab:
                        feat_token = subst
                        log.debug("Token   : '%s' substituted with '%s'", token, feat_token)

                if feat_token:
                    retval.append(model[feat_token])
                else:
                    # the token is not in the model -> 'guess' the most likely
                    # vector representation (summing up char ngrams) and use
                    # this vector
                    retval.append(model[token])
        return retval

    @staticmethod
    def pad_toks(toks, maxlen=50, value=None, split=True):
        """ Pad a list of toks with value to maxlen length. """
        if value is None:
            value = Utils.padding_str

        if len(toks) <= maxlen:
            return [toks + [value] * (maxlen-len(toks))]
        elif split:
            return [toks[0:maxlen]] + Utils.pad_toks(toks[maxlen:],
                                                     maxlen=maxlen,
                                                     value=value, split=True)
        else:
            return toks

    @staticmethod
    def weighted_categorical_crossentropy(weights):
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')


        from:
            https://github.com/keras-team/keras/issues/6261
        """
        """
        A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
        @url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
        @author: wassname
        """

        weights = k.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= k.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = k.clip(y_pred, k.epsilon(), 1 - k.epsilon())
            # calc
            loss = y_true * k.log(y_pred) * weights
            loss = -k.sum(loss, -1)
            return loss

        return loss

    @staticmethod
    def weighted_binary_crossentropy(zero_weight, one_weight):
        def loss(y_true, y_pred, from_logits=False, label_smoothing=0):  # pylint: disable=missing-docstring
            from tensorflow.python.framework import ops
            from tensorflow.python.framework import smart_cond
            from tensorflow.python.ops import math_ops

            y_pred = ops.convert_to_tensor(y_pred)
            y_true = math_ops.cast(y_true, y_pred.dtype)
            label_smoothing = ops.convert_to_tensor(label_smoothing,
                                                    dtype=k.floatx())

            def _smooth_labels():
                return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

            y_true = smart_cond.smart_cond(label_smoothing, _smooth_labels, lambda: y_true)
            bce = k.binary_crossentropy(y_true, y_pred, from_logits=from_logits)
            weight_vector = y_true * one_weight + (1. - y_true) * zero_weight
            weighted_bce = k.mean(weight_vector * bce, axis=-1)
            return weighted_bce
        return loss

def main():
    """Main loop."""

    ### PARSE PARAMS and INIT LOGGING
    args = _parse_args()
    logging_format = "%(asctime)s - %(funcName)8s(),ln%(lineno)3s: %(message)s"
    logging.basicConfig(format=logging_format, level=args.log_level,
                        datefmt='%H:%M:%S')
    logging.debug("Command line arguments:%s", args)

    from gensim.models.fasttext import load_facebook_model

    from sklearn.model_selection import KFold

    ### SET PARAMS
    seed = 42
    seq_max_length = args.seq_max_length
    n_splits = args.n_splits
    batch_size = args.batch_size
    epochs = args.epochs
    recurrent_dropout = args.recurrent_dropout
    optimizer = args.optimizer

    metrics = [
        keras.metrics.TruePositives(name='tp'),
        keras.metrics.FalsePositives(name='fp'),
        keras.metrics.TrueNegatives(name='tn'),
        keras.metrics.FalseNegatives(name='fn'),
        keras.metrics.BinaryAccuracy(name='accuracy'),
        keras.metrics.Precision(name='precision'),
        keras.metrics.Recall(name='recall'),
        keras.metrics.AUC(name='auc'),
        tfa.metrics.FBetaScore(name='f1-score', num_classes=2, average="micro",
                               threshold=0.5)
    ]

    ### SET CORPUS
    corpus = Corpus(args.corpus_fn, args.tokens_tags, args.metadata_fn)

    ### SET EMBEDDINGS
    embedding_models = []
    for embedding_fn in args.embeddings:
        embedding_models.append(load_facebook_model(embedding_fn))

    ### EXTRACT FEATURES
    xs = []
    feature_vec_lengths = []
    for embed_mod in embedding_models:
        # for 'y' actually only the results from one model would suffice, for
        # example, from embedding_models[0], but y is calculated in any case,
        # so using it here is cheaper - rather than later down:
        # _, y = corpus.X_y(embedding_models[0], maxlen=seq_max_length)
        x, y = corpus.X_y(embed_mod, maxlen=seq_max_length)
        xs.append(x)
        feature_vec_lengths.append(embed_mod.vector_size)
    logging.info(feature_vec_lengths)

    total = len(np.array(y).flatten())
    pos = sum(np.array(y).flatten())
    neg = total - pos
    logging.info("total: %d (pos:%d, neg:%d)", total, pos, neg)

    # Set the output layer's bias to reflect the dataset imbalance
    # https://www.tensorflow.org/tutorials/structured_data/imbalanced_data
    # initial_bias = np.log([pos/neg])
    # logging.info("Output weight for loss: %2f", initial_bias)
    initial_bias = None

    ### Use one-hot encoded categories for binary output (bool,bool)
    # weight_for_0 = (1 / neg)*(total)/2.0
    # weight_for_1 = (1 / pos)*(total)/2.0
    # class_weight = {0: weight_for_0, 1: weight_for_1}
    # logging.info("Weight for class 0: %2f", weight_for_0)
    # logging.info("Weight for class 1: %2f", weight_for_1)

    loss_weight = np.log((total/pos)-1)
    # loss = Utils.weighted_categorical_crossentropy([1, loss_weight])  # !
    # logging.info("Set loss_weight 1 : %f", loss_weight)
    # y_cat = to_categorical(y, 2)
    # loss = "categorical_crossentropy"

    ### Use single binary output (bool)
    _y = []
    for epoch in y:
        _epoch = []
        for val in epoch:
            _epoch.append([val])
        _y.append(_epoch)
    y_cat = np.array(_y)
    #loss = 'binary_crossentropy'
    loss = Utils.weighted_binary_crossentropy(1, loss_weight)

    if args.pos:
        logging.info("POS-tagging training corpus and calculating features...")
        xposs_cat = corpus.Xposs(categorical=True, maxlen=seq_max_length)
        logging.info("...done.")
    if corpus.metadata:
        xmetas_cat = corpus.Xmetas(maxlen=seq_max_length)

    def get_model(metrics, output_bias=None):
        """ Create and return the model. """

        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)

        k.clear_session()
        # INPUTS
        inputs = []
        models = []
        for embed_mod_id, _ in enumerate(embedding_models):
            inputs.append(
                Input(shape=(seq_max_length, feature_vec_lengths[embed_mod_id],)))

            models.append(
                Masking(mask_value=0.)(inputs[-1])
            )

        if args.pos:
            inputs.append(
                Input(shape=(seq_max_length, len(xposs_cat[0][0]),))
            )
            models.append(
                Masking(mask_value=0.)(inputs[-1])
            )
        if corpus.metadata:
            inputs.append(
                Input(shape=(seq_max_length, len(xmetas_cat[0][0]),))
            )
            models.append(
                Masking(mask_value=0.)(inputs[-1])
            )


        # Combinde INPUTS (including masks)
        if len(models) > 1:
            model = concatenate(models)
        else:
            model = models[0]

        # CORE MODEL
        model = Bidirectional(LSTM(50, return_sequences=True,
                                   dropout=0,  # !
                                   # dropout=0.1,
                                   # dropout=0.25,
                                   recurrent_dropout=recurrent_dropout,
                                   implementation=1))(model)

        # (unfold LSTM and)
        # one-hot encode binary label
        # outputs = TimeDistributed(Dense(2, activation="softmax"))(model)
        outputs = TimeDistributed(Dense(1, activation="sigmoid",
                                        bias_initializer=output_bias))(model)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    ### TRAIN & EVALUATE
    ###
    # define cross validation tests
    if n_splits > 0 and args.evaluate:
        logging.info("Training and evaluating corpus with %d folds...", n_splits)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        cvscores = {}
        for train, test in kfold.split(xs[0], y_cat):
            model = get_model(metrics, initial_bias)
            X_train = [np.array(x)[train] for x in xs]
            X_test = [np.array(x)[test] for x in xs]
            if args.pos:
                X_train += [xposs_cat[train]]
                X_test += [xposs_cat[test]]
            if corpus.metadata:
                X_train += [xmetas_cat[train]]
                X_test += [xmetas_cat[test]]

            ###
            # TRAIN
            model.fit(X_train, y_cat[train],
                      batch_size=batch_size, epochs=epochs, verbose=0)

            ###
            # EVALUATE the model
            scores = model.evaluate(X_test, y_cat[test], verbose=1)
            scores_summary = []
            for _id in range(len(model.metrics_names)):
                scores_summary.append("%s:%.2f" %(model.metrics_names[_id],
                                                  scores[_id]))
                scores_list = cvscores.get(model.metrics_names[_id], [])
                scores_list.append(scores[_id])
                cvscores[model.metrics_names[_id]] = scores_list
            #logging.info("\t".join(scores_summary))
            print("\t".join(scores_summary), file=args.eval_fn)
            args.eval_fn.flush()

        cvscores_summary = []
        for key in cvscores:
            cvscores_summary.append("%s:%.2f (+/- %.2f)" %(key,
                                                           np.mean(cvscores[key]),
                                                           np.std(cvscores[key])))
        logging.info("\t".join(cvscores_summary))
        print("\n"+("\n".join(cvscores_summary))+"\n###", file=args.eval_fn)

        model.summary(print_fn=lambda x: print(x, file=args.eval_fn))
        logging.info("...done.")

    if not args.predict:
        exit(0)
    ### PREDICT
    ###
    # Re-init the model
    logging.info("building model...")
    model = get_model(metrics, initial_bias)
    logging.info("...done.")

    logging.info("training model...")
    X_input = [np.array(x) for x in xs]
    if args.pos:
        X_input += [xposs_cat]
    if corpus.metadata:
        X_input += [xmetas_cat]

    model.fit(X_input, y_cat, batch_size=batch_size, epochs=epochs, verbose=1)
    logging.info("...done.")

    ### SET CORPUS
    corpus_test = Corpus(args.corpus_test_fn, args.tokens_test_tags,
                         args.metadata_fn, mode="test")


    ### EXTRACT FEATURES
    logging.info("calculating features from embeddings...")
    x_preds = []
    for embed_mod in embedding_models:
        x, _ = corpus_test.X_y(embed_mod, maxlen=seq_max_length)
        x_preds.append(x)
    logging.info("...done.")

    if args.pos:
        logging.info("Tagging testing corpus and calculating features...")
        xposs_cat = corpus_test.Xposs(categorical=True, maxlen=seq_max_length)
        logging.info("...done.")
    if corpus.metadata:
        xmetas_cat = corpus_test.Xmetas(maxlen=seq_max_length)

    logging.info("using model to calculate predictions...")
    X_input = [np.array(x_pred) for x_pred in x_preds]
    if args.pos:
        X_input += [xposs_cat]
    if corpus_test.metadata:
        X_input += [xmetas_cat]
    preds = tf.cast(tf.math.round(model.predict(X_input,
                                                batch_size=batch_size)),
                    tf.int32)
    logging.info("...done.")

    # OUTPUT predictions
    logging.info("writing predictions...")
    pred_id = 0
    for txt_id in corpus_test.tokens:
        for sentence_id in corpus_test.tokens[txt_id]:
            sentence = corpus_test.sentence(txt_id, sentence_id)
            tokens = corpus_test.tokens[txt_id][sentence_id]
            for tok_id, _ in enumerate(sentence):
                y_pred = preds[pred_id]
                if tok_id+1 in tokens:
                    # print(pred_id, tok_id, tok_id%seq_max_length)
                    print("{}_{}_{},{},{}".format(txt_id, sentence_id,
                                                  tok_id+1, y_pred[tok_id %
                                                                   seq_max_length][0],
                                                  sentence[tok_id][0]),
                          file=args.predicts_fn)
                if (tok_id+1) % seq_max_length == 0 and tok_id+1 < len(sentence):
                    pred_id += 1
            pred_id += 1
    logging.info("...done.")

def _parse_args():
    """
    Set up argparse parser and return the populated namespace.

    Returns:
        A populated argparse namespace.
    """
    def add_bool_arg(parser, name, default=False, arg_help=None):
        """Helper function to add --ARG and --no-ARG exclusive options."""
        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument('--' + name, dest=name, action='store_true',
                           help=arg_help)
        group.add_argument('--no-' + name, dest=name, action='store_false')
        parser.set_defaults(**{name:default})

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=DESCRIPTION, epilog=EPILOG,
        fromfile_prefix_chars='@')

    parser.add_argument(
        "-v", "--verbose",
        action="count", default=0,
        help="Verbose mode.  Multiple -v options increase the verbosity.  "
        "The maximum is 2.")

    parser.add_argument(
        "--train_corpus",
        dest="corpus_fn",
        default="../naacl_flp/vuamc_corpus_train.csv",
        help="Name of csv file with training corpus texts.  (default: %(default)s)")

    parser.add_argument(
        "--train_labels",
        dest="tokens_tags",
        default="../naacl_flp/all_pos_tokens.csv",
        # tokens_tags = "../naacl_flp/verb_tokens.csv"
        help="Name of csv file with training corpus ids and labels.  (default: %(default)s)")

    parser.add_argument(
        "--test_corpus",
        dest="corpus_test_fn",
        default="../naacl_flp/vuamc_corpus_test.csv",
        help="Name of csv file with testing corpus texts.  (default: %(default)s)")

    parser.add_argument(
        "--test_labels",
        dest="tokens_test_tags",
        default="../naacl_flp/all_pos_tokens_test.csv",
        # tokens_tags = "../naacl_flp/verb_tokens_test.csv"
        help="Name of csv file with testing corpus ids without labels.  (default: %(default)s)")

    # for the TOEFL data this looks like this:
    # text_id, tokens, prompt, proficiency
    # 1005892,9,234,P3,ARA,medium
    # 1012679,25,340,P3,ARA,high
    parser.add_argument(
        "--metadata",
        dest="metadata_fn",
        default=False,
        help="Name of csv file with testing and training corpus metadata-per-text/-per-category.  (default: %(default)s)")

    parser.add_argument(
        "--predict",
        default=False,
        action="store_true",
        help="Use full training corpus and predict on the test corpus.  (default: %(default)s)")

    add_bool_arg(
        parser,
        'pos',
        default=True,
        arg_help="Use POS-tagger features.  (default: %(default)s)")

    parser.add_argument(
        "--predictfile",
        dest="predicts_fn",
        nargs='?', type=argparse.FileType('w'), default=stdout,
        help="Write predicted labels to this file.  (stdout)")

    parser.add_argument(
        "--embeddings",
        nargs='*', required=True,
        help="File names of pre-trained FastText embedding .bin files.")
    #embedding_models.append(
    #    FastText.load_fasttext_format('../data/bnc2_tt2'))
    ## embedding_models.append(
    ##     FastText.load_fasttext_format('../data/wiki.en.bin'))
    ## embedding_models.append(
    ##     FastText.load_fasttext_format('../data/ententen13_tt2_1.bin'))
    #embedding_models.append(
    #    FastText.load_fasttext_format('../data/NLI_2013/NLI_2013_low_min1_dim50.bin'))
    #embedding_models.append(
    #    FastText.load_fasttext_format('../data/NLI_2013/NLI_2013_med_min1_dim50.bin'))
    #embedding_models.append(
    #    FastText.load_fasttext_format('../data/NLI_2013/NLI_2013_high_min1_dim50.bin'))


    parser.add_argument(
        "--seq_max_length",
        default=15, type=int,
        help="Maximum sequence length.  (default: %(default)s)")

    parser.add_argument(
        "--evaluate",
        default=False,
        action="store_true",
        help="Evaluate the model with n_splits X-validation.  (default: %(default)s)")

    parser.add_argument(
        "--evaluatefile",
        dest="eval_fn",
        nargs='?', type=argparse.FileType('w'), default=stdout,
        help="Write x-validation scores this file.  (stdout)")

    parser.add_argument(
        "--n_splits",
        default=3, type=int,
        help="Number of splits for X-validation.  (default: %(default)s)")

    parser.add_argument(
        "--batch_size",
        default=32, type=int,
        help="Batch size for training the network.  (default: %(default)s)")

    parser.add_argument(
        "--epochs",
        default=20, type=int,
        help="Epochs for training the network.  (default: %(default)s)")

    parser.add_argument(
        "--recurrent_dropout",
        default=0.25, type=float,
        help="Dropout value for the RNN layer.  (default: %(default)s)")

    parser.add_argument(
        "--optimizer",
        default="rmsprop",
        help="Optimizer for the NN model.  (default: %(default)s)")

    args = parser.parse_args()
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    args.log_level = log_levels[min(len(log_levels)-1, args.verbose)]

    return args

if __name__ == "__main__":
    main()
