#!/usr/bin/env python3
"""
Author: egon w. stemle <egon.stemle@eurac.edu>
Contributor: alexander onysko <alexander.onysko@aau.at>

Eurac Research and Alpen-Adria University contribution to NAACL-FLP-shared-task
"""

from collections import Counter, OrderedDict
from sys import argv, exit

import csv
import logging

import pandas as pd
import numpy as np

from gensim.models.wrappers import FastText

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
    """
    def __init__(self, vuamc_fn, tokens_fn, mode='train', sanity_check=False):
        self.log = logging.getLogger(type(self).__name__)
        self.vuamc_fn = vuamc_fn
        self.tokens_fn = tokens_fn


        self.vuamc_delimiter = self.tokens_delimiter = ","
        self.vuamc_quotechar = self.tokens_quotechar = '"'

        self.vuamc = self._load_vuamc(self.vuamc_fn)
        self.tokens = self._load_tokens(self.tokens_fn)

        self.mode = mode

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
            for row in csvreader:
                label = int(row[1])
                txt_id, sentence_id, token_id = row[0].split('_')

                if txt_id not in data:
                    data[txt_id] = OrderedDict()

                if sentence_id not in data[txt_id]:
                    data[txt_id][sentence_id] = OrderedDict()

                if (txt_id in data and sentence_id in data[txt_id] and
                        int(token_id) in data[txt_id][sentence_id]):
                    exit("Identical keys in line {}".format(csvreader.line_num))

                data[txt_id][sentence_id][int(token_id)] = label

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

    @property
    def sentences(self):
        """ Yield list (sentences) of tuples (word, label). """
        def populate_sentences():
            """ Helper to populate sentences. """
            for txt_id in self.tokens:
                for sentence_id in self.tokens[txt_id]:
                    sentence = []
                    for token_id in self.vuamc[txt_id][sentence_id]['tokens'].keys():
                        if token_id in self.tokens[txt_id][sentence_id]:
                            label = self.tokens[txt_id][sentence_id][token_id]
                        else:
                            label = 0
                        sentence.append((self.vuamc[txt_id][sentence_id]['tokens'][token_id],
                                         label))
                    yield sentence
        if self._sentences is None:
            self._sentences = list(populate_sentences())

        return self._sentences

class Utils():
    """ Utility and helper functions. """
    padding_str = "__PADDING__"

    def __init__(self):
        self.log = logging.getLogger(type(self).__name__)
        self._models = None
        self._missing_w2vs = dict()

    @property
    def models(self):
        return self._models

    @models.setter
    def models(self, model):
        if not self._models:
            self._models = list()
        self._models.append(model)

    @staticmethod
    def toks2feat(toks, model, context_length=10):
        """
        Return the feature vectors of a list of tokens for a model.
        """
        log = logging.getLogger(__name__)
        retval = []
        for tok_id, token in enumerate(toks):
            feat_token = None  # token to use to calculate the feature vector
            if token in model:
                # the easy case: the token is in the model -> use it
                feat_token = token
            elif token in ['‘', '’', '—', '…']:
                # cleaning up some common chars (missing from some models)
                if token in ['‘', '’']:
                    subst = "'"
                elif token == '—':
                    subst = "-"
                elif token == '…':
                    subst = "..."

                if subst in model:
                    feat_token = subst
                    log.info("Token   : '%s' substituted with '%s'", token, feat_token)

            if not feat_token and not token == Utils.padding_str:
                # the token is not in the model -> 'guess' the most likely word
                # given a context_length context, and use this word

                context_toks = []  # +/-context_length tokens that exist in the
                                   # model
                pre_toks = [_tok for _tok in toks[:tok_id] if _tok in
                            model]
                post_toks = [_tok for _tok in toks[tok_id+1:] if _tok in
                             model]
                while len(context_toks) < context_length and (
                        pre_toks or post_toks):
                    if pre_toks:
                        context_toks.insert(0, pre_toks.pop())
                    if post_toks:
                        context_toks.append(post_toks.pop(0))

                log.info("'%s'    : not in model:%s", token, str(model))
                log.debug("Tokens  : %s", str(toks))
                log.debug("pre/post: %s | %s", str(pre_toks), str(post_toks))
                log.debug("Context : %s", str(context_toks))

                if context_toks:
                    feat_token = model.most_similar(context_toks, topn=1)[0][0]
                    log.info("Token   : '%s' substituted with '%s'", token, feat_token)

            if token == Utils.padding_str:
                retval.append(model.vector_size * [0])
            elif feat_token:
                retval.append(model[feat_token])
            else:
                # there wasn't enough context to 'guess' a word -> use a new
                # (stable) feature representation.
                # this is very unlikely to happen!
                retval.append(model.seeded_vector(feat_token))
                log.info("Token   : '%s' substituted with a seeded vector.", token)
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
    def X_y(corpus, models):
        retval_X = []
        retval_y = []
        for sentence in corpus.sentences:
            sentence_toks_pads = Utils.pad_toks([tok[0] for tok in sentence])
            sentence_ys_pads = Utils.pad_toks([tok[1] for tok in sentence], value=0)

            for tmp_id, sentence_toks_pad in enumerate(sentence_toks_pads):
                sentence_retval = []
                for model in models:
                    sentence_retval.extend(Utils.toks2feat(sentence_toks_pad, model))
                # FIXME: implement 
                #if len(models) > 1:
                #    sentence_retval = list(map((lambda *args: np.hstack([*args])), *sentence_retval))

                retval_X.append(np.array(sentence_retval))
                retval_y.append(np.array(sentence_ys_pads[tmp_id]))
        return retval_X, retval_y

    @staticmethod
    def f1_score_least_frequent(y_true, y_pred):
        """
        Calculate the F1 score of the least frequent label/class in ``y_true`` for
        ``y_pred``.

        Parameters
        ----------
        y_true : array-like of float
            The true/actual/gold labels for the data.
        y_pred : array-like of float
            The predicted/observed labels for the data.

        Returns
        -------
        ret_score : float
            F1 score of the least frequent label.

        from:
            https://raw.githubusercontent.com/EducationalTestingService/skll/master/skll/metrics.py
        """
        from sklearn.metrics import f1_score
        least_frequent = np.bincount(y_true).argmin()
        return f1_score(y_true, y_pred, average=None)[least_frequent]

    @staticmethod
    def get_class_weights(y, smooth_factor=0):
        """
        Returns the weights for each class based on the frequencies of the samples
        :param smooth_factor: factor that smooths extremely uneven weights
        :param y: list of true labels (the labels must be hashable)
        :return: dictionary with the weight for each class

        from:
        https://github.com/keras-team/keras/issues/5116

        Utils.get_class_weights([y for x in y_tr for y in x])
        """
        counter = Counter(y)

        if smooth_factor > 0:
            p = max(counter.values()) * smooth_factor
            for k in counter.keys():
                counter[k] += p

        majority = max(counter.values())

        return {cls: float(majority / count) for cls, count in counter.items()}

    @staticmethod
    def f1(y_true, y_pred):
        from keras import backend as K
        def recall(y_true, y_pred):
            """Recall metric.

            Only computes a batch-wise average of recall.

            Computes the recall, a metric for multi-label classification of
            how many relevant items are selected.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision(y_true, y_pred):
            """Precision metric.

            Only computes a batch-wise average of precision.

            Computes the precision, a metric for multi-label classification of
            how many selected items are relevant.
            """
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision
        precision = precision(y_true, y_pred)
        recall = recall(y_true, y_pred)
        return 2*((precision*recall)/(precision+recall))

    """
    A weighted version of categorical_crossentropy for keras (2.0.6). This lets you apply a weight to unbalanced classes.
    @url: https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d
    @author: wassname
    """
    @staticmethod
    def weighted_categorical_crossentropy(weights):
        from keras import backend as K
        """
        A weighted version of keras.objectives.categorical_crossentropy

        Variables:
            weights: numpy array of shape (C,) where C is the number of classes

        Usage:
            weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
            loss = weighted_categorical_crossentropy(weights)
            model.compile(loss=loss,optimizer='adam')
        """

        weights = K.variable(weights)

        def loss(y_true, y_pred):
            # scale predictions so that the class probas of each sample sum to 1
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            # clip to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
            # calc
            loss = y_true * K.log(y_pred) * weights
            loss = -K.sum(loss, -1)
            return loss

        return loss

    @staticmethod
    def weighted_loss(onehot_labels, logits):
        """ Scale loss based on class weights. """
        # # compute weights based on their frequencies
        # class_weights = .... # set your class weights here
        # # computer weights based on onehot labels
        # weights = tf.reduce_sum(class_weights * onehot_labels, axis=-1)
        # # compute (unweighted) softmax cross entropy loss
        # unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=[onehot_labels], logits=[logits])
        # # apply the weights, relying on broadcasting of the multiplication
        # weighted_losses = unweighted_losses * weights
        # # reduce the result to get your final loss
        # loss = tf.reduce_mean(weighted_losses)
        # return loss
        pass

def main():
    import logging

    from keras import backend as K
    from keras.layers import Masking, Flatten, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
    from keras.models import Model, Input
    from keras.utils import to_categorical

    import sklearn
    from sklearn.model_selection import KFold

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.debug("test")

    if len(argv) == 1:
        corpus_fn = "../naacl_flp/vuamc_corpus_train.csv"
        tokens_tags = "../naacl_flp/all_pos_tokens.csv"
    else:
        corpus_fn = argv[1]

    corpus = Corpus(corpus_fn, tokens_tags)
    # vuamc_corpus = pd.read_csv(corpus_fn, encoding="utf-8")
    # toks_tags = pd.read_csv(tokens_tags, encoding="utf-8")

    models = []
    #models.append(FastText.load_fasttext_format('../data/bnc2_tt2'))
    models.append(FastText.load_fasttext_format('../data/NLI_2013/NLI_2013_low-med.bin'))
    #models.append(FastText.load_fasttext_format('../data/NLI_2013/NLI_2013_high.bin'))
    feature_vec_length = sum([model.vector_size for model in models])

    X, y = Utils.X_y(corpus, models)

    seed = 42
    #
    n_splits = 3
    batch_size = 16 # 32
    epochs = 3 # 5
    #
    seq_max_length = 50
    #
    # loss = Utils.weighted_categorical_crossentropy([1,1])

    loss = "categorical_crossentropy"
    y_cat = to_categorical(y, 2)
    #
    #loss = "binary_crossentropy"
    #y_res = np.array(y).reshape(len(y),feature_vec_length,1)
    #
    #f1 = Utils.f1_score_least_frequent
    #metrics = [f1]
    metrics = ["accuracy"]

    # define 10-fold cross validation test harness
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    cvscores = []
    prfsscores = []
    labels_preds = []
    labels_y = []
    for train, test in kfold.split(X, y_cat):
        #sample_weight = y_cat[train] * 100
        #sample_weight = sample_weight.reshape(len(y_cat[train]), 100)

        # Create Model
        inputs = Input(shape=(seq_max_length, feature_vec_length,))
        model = Masking(mask_value=[0]*feature_vec_length)(inputs)
        model = Bidirectional(LSTM(100, return_sequences=True,
                                   recurrent_dropout=0.1))(model)
        #model = Flatten()(model)
        #outputs = Dense(seq_max_length, activation="sigmoid")(model)
        outputs = TimeDistributed(Dense(2, activation="softmax"))(model)
        #outputs = TimeDistributed(Dense(1, activation="sigmoid"))(model)
        model = Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer="rmsprop", loss=loss, metrics=metrics)
        model.fit(np.array(X)[train], y_cat[train],
                  batch_size=batch_size, epochs=epochs, verbose=1)

        ###
        # evaluate the model
        x_test = np.array(X)[test]
        #
        scores = model.evaluate(x_test, y_cat[test], verbose=0)
        #
        p = model.predict(x_test, batch_size=batch_size)
        q = K.argmax(p)
        r = K.eval(q)
        for xt_id, xt in enumerate(x_test):
            dims = len(xt[np.all(xt, axis = 1)])
            ##prfsscores.append(sklearn.metrics.precision_recall_fscore_support(np.array(y)[test][xt_id][:dims], r[xt_id][:dims]))
            labels_preds.extend(r[xt_id][:dims])
            labels_y.extend(np.array(y)[test][xt_id][:dims])
            #prfsscores.append(
            #    sklearn.metrics.precision_recall_fscore_support(
            #        np.array(y)[test][xt_id][:dims], p))

        print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
        cvscores.append(scores[1] * 100)
    print("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    #print(sklearn.metrics.confusion_matrix(labels_y, labels_preds))
    #print(np.mean([f[2] for f in prfsscores]))
    print(Utils.f1_score_least_frequent(labels_y, labels_preds))

if __name__ == "__main__":
    main()
