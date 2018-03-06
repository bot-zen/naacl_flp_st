#!/usr/bin/env python3
"""
Author: egon w. stemle <egon.stemle@eurac.edu>
Contributor: alexander onysko <alexander.onysko@aau.at>

Eurac Research and Alpen-Adria University contribution to NAACL-FLP-shared-task
"""

from collections import Counter, OrderedDict
from sys import exit

import csv
import logging

import numpy as np

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
    def __init__(self, vuamc_fn, tokens_fn, mode="train", sanity_check=False):
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
                if self.mode == "train":
                    label = int(row[1])
                else:
                    label = -1
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

        for sent in sentences:
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

class Utils():
    """ Utility and helper functions. """
    padding_str = "__PADDING__"

    def __init__(self):
        self.log = logging.getLogger(type(self).__name__)

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
                    log.debug("Token   : '%s' substituted with '%s'", token, feat_token)

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


        from:
            https://github.com/keras-team/keras/issues/6261
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

def main():
    import logging

    from gensim.models.wrappers import FastText

    from keras import backend as K
    from keras.layers import Masking, Flatten, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional, concatenate
    from keras.models import Model, Input
    from keras.utils import to_categorical

    import sklearn
    from sklearn.model_selection import KFold

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logging.debug("test")

    corpus_fn = "../naacl_flp/vuamc_corpus_train.csv"
    tokens_tags = "../naacl_flp/all_pos_tokens.csv"
    corpus = Corpus(corpus_fn, tokens_tags)

    models = []
    models.append(FastText.load_fasttext_format('../data/bnc2_tt2'))
    # models.append(FastText.load_fasttext_format('../data/NLI_2013/NLI_2013_low-med.bin'))
    models.append(FastText.load_fasttext_format('../data/ententen13_tt2_1.bin'))
    models.append(FastText.load_fasttext_format('../data/NLI_2013/NLI_2013_med-high.bin'))

    #
    # seq_max_length = 50
    seq_max_length = 10  # !
    # seq_max_length = 15

    X0, y = corpus.X_y(models[0], maxlen=seq_max_length)
    X1, _ = corpus.X_y(models[1], maxlen=seq_max_length)
    X2, _ = corpus.X_y(models[2], maxlen=seq_max_length)
    feature_vec_length0 = sum([model.vector_size for model in [models[0]]])
    feature_vec_length1 = sum([model.vector_size for model in [models[1]]])
    feature_vec_length2 = sum([model.vector_size for model in [models[2]]])

    seed = 42
    #
    n_splits = 3
    # batch_size = 16
    batch_size = 32  # !
    # batch_size = 64

    # epochs = 3
    # epochs = 5
    epochs = 20  # !
    # epochs = 32

    #
    #recurrent_dropout=0.1
    recurrent_dropout=0.25  # !
    #recurrent_dropout=0.5

    #
    # cf. http://ruder.io/optimizing-gradient-descent/
    optimizer = "rmsprop"  # !
    #optimizer = "adadelta"
    #optimizer = "adam"
    #
    loss = Utils.weighted_categorical_crossentropy([1, 5])  # !
    #loss = "categorical_crossentropy"
    y_cat = to_categorical(y, 2)
    #
    #loss = "binary_crossentropy"
    #y_res = np.array(y).reshape(len(y),feature_vec_length,1)
    #
    # f1 = Utils.f1
    # metrics = ["categorical_accuracy", f1]
    metrics = ["categorical_accuracy"]


    def get_model():
        """ Create and return the model. """

        # INPUTS
        input0 = Input(shape=(seq_max_length, feature_vec_length0,))
        model0 = Masking(mask_value=[0]*feature_vec_length0)(input0)

        input1 = Input(shape=(seq_max_length, feature_vec_length1,))
        model1 = Masking(mask_value=[0]*feature_vec_length1)(input1)

        input2 = Input(shape=(seq_max_length, feature_vec_length2,))
        model2 = Masking(mask_value=[0]*feature_vec_length2)(input2)

        # Combinde INPUTS (including masks)
        model = concatenate([model0,
                             model1,
                             model2
                            ])

        # CORE MODEL
        model = Bidirectional(LSTM(32, return_sequences=True,
                                   recurrent_dropout=recurrent_dropout))(model)

        # (unfold LSTM and)
        # one-hot encode binary label
        outputs = TimeDistributed(Dense(2, activation="softmax"))(model)

        model = Model(inputs=[input0,
                              input1,
                              input2
                             ], outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    ### TRAIN & EVALUATE
    ###
    # define cross validation tests
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # cvscores = []
    # labels_preds = []
    # labels_y = []
    # for train, test in kfold.split(X0, y_cat):
    #     model = get_model()

    #     ###
    #     # TRAIN
    #     model.fit([np.array(X0)[train],
    #                np.array(X1)[train],
    #                np.array(X2)[train]
    #               ], y_cat[train],
    #               batch_size=batch_size, epochs=epochs, verbose=0)

    #     ###
    #     # EVALUATE the model
    #     x0_test = np.array(X0)[test]
    #     x1_test = np.array(X1)[test]
    #     x2_test = np.array(X2)[test]
    #     #
    #     scores = model.evaluate([x0_test,
    #                              x1_test,
    #                              x2_test], y_cat[test], verbose=0)
    #     logging.info("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    #     cvscores.append(scores[1] * 100)
    #     #
    #     p = model.predict([x0_test,
    #                        x1_test,
    #                        x2_test], batch_size=batch_size)
    #     q = K.argmax(p)
    #     r = K.eval(q)
    #     for xt_id, xt in enumerate(x0_test):
    #         dims = len(xt[np.all(xt, axis=1)])
    #         labels_preds.extend(r[xt_id][:dims])
    #         labels_y.extend(np.array(y)[test][xt_id][:dims])
    # logging.info("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
    # logging.info(Utils.f1_score_least_frequent(labels_y, labels_preds))

    ### PREDICT
    ###
    # Re-init the model
    model = get_model()
    model.fit([np.array(X0),
               np.array(X1),
               np.array(X2)
              ], y_cat,
              batch_size=batch_size, epochs=epochs, verbose=0)

    corpus_test_fn = "../naacl_flp/vuamc_corpus_test.csv"
    tokens_test_tags = "../naacl_flp/all_pos_tokens_test.csv"
    corpus_test = Corpus(corpus_test_fn, tokens_test_tags, mode="test")

    x0_pred = []
    x1_pred = []
    x2_pred = []
    for txt_id in corpus_test.tokens:
        for sentence_id in corpus_test.tokens[txt_id]:
            sentence = corpus_test.sentence(txt_id, sentence_id)

            x0_pred.extend(Corpus.X_y_sentence(sentence=sentence,
                                               model=models[0],
                                               maxlen=seq_max_length)[0])
            x1_pred.extend(Corpus.X_y_sentence(sentence=sentence,
                                               model=models[1],
                                               maxlen=seq_max_length)[0])
            x2_pred.extend(Corpus.X_y_sentence(sentence=sentence,
                                               model=models[2],
                                               maxlen=seq_max_length)[0])

    y_preds = []
    p = model.predict([np.array(x0_pred),
                       np.array(x1_pred),
                       np.array(x2_pred)], batch_size=batch_size)
    q = K.argmax(p)
    y_preds = K.eval(q)

    pred_id = 0
    for txt_id in corpus_test.tokens:
        for sentence_id in corpus_test.tokens[txt_id]:
            sentence = corpus_test.sentence(txt_id, sentence_id)
            tokens = corpus_test.tokens[txt_id][sentence_id]
            for tok_id in range(len(sentence)):
                y_pred = y_preds[pred_id]
                if tok_id+1 in tokens:
                    # print(pred_id, tok_id, tok_id%seq_max_length)
                    print("{}_{}_{},{},{}".format(txt_id, sentence_id,
                                                  tok_id+1, y_pred[tok_id %
                                                                   seq_max_length],
                                                  sentence[tok_id][0]))
                if (tok_id+1) % seq_max_length == 0 and tok_id+1 < len(sentence):
                    pred_id += 1
            pred_id += 1

            #X0_sent, _ = Corpus.X_y_sentence(
            #    sentence=sentence, model=models[0], maxlen=seq_max_length)
            #X1_sent, _ = Corpus.X_y_sentence(
            #    sentence=sentence, model=models[1], maxlen=seq_max_length)

            #y_preds = []
            #for sent_id, sent in enumerate(X0_sent):
            #    p = model.predict([np.array([X0_sent[sent_id]]),
            #                       np.array([X1_sent[sent_id]])], batch_size=batch_size)
            #    q = K.argmax(p)
            #    r = K.eval(q)
            #    y_preds.extend(r)
            #y_preds = np.array(y_preds).flatten()

            #tokens = corpus_test.tokens[txt_id][sentence_id]
            #for tok_id in range(len(sentence)):
            #    if tok_id+1 in tokens:
            #        print("{}_{}_{},{},{}".format(txt_id, sentence_id,
            #                                      tok_id+1, y_preds[tok_id],
            #                                      sentence[tok_id][0]))

if __name__ == "__main__":
    main()
