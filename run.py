#!/usr/bin/env python3
"""
Eurac Research and Alpen-Adria University contribution
to the NAACL-FLP-shared-task 2018.

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
                txt_id, sentence_id, token_id = row[0].split('_')

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

    def Xposs(self, maxlen=None):
        retval = []
        for sent in self.sentences:
            poss = [tag[1] for tag in pos_tag([tok[0] for tok in sent])]
            sentence_poss_pads = Utils.pad_toks(poss,
                                                value=0,
                                                maxlen=maxlen)
            for tmp_id, sentence_poss_pad in enumerate(sentence_poss_pads):
                retval.append(Utils.poss2feat(sentence_poss_pad))
        return np.array(retval)

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

                log.debug("'%s'    : not in model:%s", token, str(model))
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
    from keras.layers import Masking, LSTM, Dense, TimeDistributed, Bidirectional, concatenate
    from keras.models import Model, Input
    from keras.utils import to_categorical

    from sklearn.model_selection import KFold

    ### PARSE PARAMS and INIT LOGGING
    args = _parse_args()
    logger = logging.getLogger()
    logging_format = "%(funcName)8s(),ln%(lineno)3s: %(message)s"
    logging.basicConfig(format=logging_format, level=args.log_level)
    logging.debug("Command line arguments:%s", args)

    ### SET PARAMS
    seed = 42
    seq_max_length = args.seq_max_length
    n_splits = args.n_splits
    batch_size = args.batch_size
    epochs = args.epochs
    recurrent_dropout = args.recurrent_dropout
    optimizer = args.optimizer
    metrics = args.metrics.split(',')

    ### SET CORPUS
    corpus = Corpus(args.corpus_fn, args.tokens_tags)

    ### SET EMBEDDINGS
    embedding_models = []
    for embedding_fn in args.embeddings:
        embedding_models.append(FastText.load_fasttext_format(embedding_fn))

    ### EXTRACT FEATURES
    xs = []
    feature_vec_lengths = []
    for embed_mod in embedding_models:
        x, _ = corpus.X_y(embed_mod, maxlen=seq_max_length)
        xs.append(x)
        feature_vec_lengths.append(embed_mod.vector_size)
    _, y = corpus.X_y(embedding_models[0], maxlen=seq_max_length)
    logging.info(feature_vec_lengths)


    loss_weight = np.log((len(np.array(y).flatten())/sum(np.array(y).flatten()))-1)
    loss = Utils.weighted_categorical_crossentropy([1, loss_weight])  # !
    logging.info("Set loss_weight 1 : {}".format(loss_weight))
    #loss = "categorical_crossentropy"
    y_cat = to_categorical(y, 2)
    #
    #loss = "binary_crossentropy"
    #y_res = np.array(y).reshape(len(y),feature_vec_length,1)
    #

    if args.pos:
        logging.info("POS-tagging training corpus and calculating features...")
        xposs = corpus.Xposs(maxlen=seq_max_length)
        xposs_cat = to_categorical(xposs, len(Utils.poss)+2)
        logging.info("...done.")

    def get_model():
        """ Create and return the model. """

        # INPUTS
        inputs = []
        models = []
        for embed_mod_id, _ in enumerate(embedding_models):
            inputs.append(
                Input(shape=(seq_max_length, feature_vec_lengths[embed_mod_id],)))

            models.append(
                Masking(mask_value=[0]*feature_vec_lengths[embed_mod_id])(inputs[-1]))

        if args.pos:
            inputs.append(Input(shape=(seq_max_length, len(Utils.poss)+2,)))
            models.append(Masking(mask_value=[0]*(len(Utils.poss)+2))(inputs[-1]))

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
                                   recurrent_dropout=recurrent_dropout))(model)

        # (unfold LSTM and)
        # one-hot encode binary label
        outputs = TimeDistributed(Dense(2, activation="softmax"))(model)

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return model

    ### TRAIN & EVALUATE
    ###
    # define cross validation tests
    if n_splits > 0 and args.evaluate:
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

        cvscores = []
        labels_preds = []
        labels_y = []
        for train, test in kfold.split(xs[0], y_cat):
            model = get_model()
            X_train = [np.array(x)[train] for x in xs]
            X_test = [np.array(x)[test] for x in xs]
            if args.pos:
                X_train += [xposs_cat[train]]
                X_test += [xposs_cat[test]]

            ###
            # TRAIN
            model.fit(X_train, y_cat[train],
                      batch_size=batch_size, epochs=epochs, verbose=0)

            ###
            # EVALUATE the model
            scores = model.evaluate(X_test, y_cat[test], verbose=0)

            logging.info("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
            print(scores[1], file=args.eval_fn)
            args.eval_fn.flush()
            cvscores.append(scores[1] * 100)


        logging.info("%.2f\t+/- %.2f" % (np.mean(cvscores), np.std(cvscores)))
        print("%.2f\t+/- %.2f" % (np.mean(cvscores), np.std(cvscores)),
              file=args.eval_fn)
        print("\n###", file=args.eval_fn)
        model.summary(print_fn=lambda x: print(x, file=args.eval_fn))
        # logging.info(Utils.f1_score_least_frequent(labels_y, labels_preds))

    if not args.predict:
        exit(0)
    ### PREDICT
    ###
    # Re-init the model
    logging.info("building model...")
    model = get_model()
    logging.info("...done.")

    logging.info("training model...")
    X_input = [np.array(x) for x in xs]
    if args.pos:
        X_input += [xposs_cat]
    model.fit(X_input, y_cat, batch_size=batch_size, epochs=epochs, verbose=0)
    logging.info("...done.")

    ### SET CORPUS
    corpus_test = Corpus(args.corpus_test_fn, args.tokens_test_tags, mode="test")

    logging.info("calculating features from embeddings...")
    x_preds = []
    for embed_mod_id, embed_mod in enumerate(embedding_models):
        x_preds.append([])
        logging.info("   embedding %d/%d...", embed_mod_id+1, len(embedding_models))
        for txt_id in corpus_test.tokens:
            for sentence_id in corpus_test.tokens[txt_id]:
                sentence = corpus_test.sentence(txt_id, sentence_id)
                x, _ = Corpus.X_y_sentence(sentence=sentence,
                                           model=embed_mod,
                                           maxlen=seq_max_length)
                x_preds[embed_mod_id].extend(x)
        logging.info("   ...done.")
    logging.info("...done.")

    if args.pos:
        logging.info("Tagging testing corpus and calculating features...")
        xposs = corpus_test.Xposs(maxlen=seq_max_length)
        xposs_cat = to_categorical(xposs, len(Utils.poss)+2)
        logging.info("...done.")

    logging.info("using model to calculate predictions...")
    y_preds = []
    X_input = [np.array(x_pred) for x_pred in x_preds]
    if args.pos:
        X_input += [xposs_cat]
    p = model.predict(X_input, batch_size=batch_size)
    q = K.argmax(p)
    y_preds = K.eval(q)
    logging.info("...done.")

    # OUTPUT predictions
    pred_id = 0
    for txt_id in corpus_test.tokens:
        for sentence_id in corpus_test.tokens[txt_id]:
            sentence = corpus_test.sentence(txt_id, sentence_id)
            tokens = corpus_test.tokens[txt_id][sentence_id]
            for tok_id, _ in enumerate(sentence):
                y_pred = y_preds[pred_id]
                if tok_id+1 in tokens:
                    # print(pred_id, tok_id, tok_id%seq_max_length)
                    print("{}_{}_{},{},{}".format(txt_id, sentence_id,
                                                  tok_id+1, y_pred[tok_id %
                                                                   seq_max_length],
                                                  sentence[tok_id][0]),
                          file=args.predicts_fn)
                if (tok_id+1) % seq_max_length == 0 and tok_id+1 < len(sentence):
                    pred_id += 1
            pred_id += 1

def _parse_args():
    """
    Set up argparse parser and return the populated namespace.

    Returns:
        A populated argparse namespace.
    """
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

    parser.add_argument(
        "--predict",
        default=False,
        action="store_true",
        help="Use full training corpus and predict on the test corpus.  (default: %(default)s)")

    parser.add_argument(
        "--pos",
        default=True,
        action="store_true",
        help="Use POS-tagger features.  (default: %(default)s)")

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

    parser.add_argument(
        "--metrics",
        default="categorical_accuracy",
        help="Function(s) used to judge the performance of the model.  (default: %(default)s)")

    args = parser.parse_args()
    log_levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    args.log_level = log_levels[min(len(log_levels)-1, args.verbose)]

    return args

if __name__ == "__main__":
    main()
