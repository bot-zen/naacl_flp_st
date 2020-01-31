# README

**Author:** egon w. stemle <egon.stemle@eurac.edu>
**Contributor:** alexander onysko <alexander.onysko@aau.at>

This is the [Eurac Research](http://www.eurac.edu/linguistics) and [Alpen-Adria
University](https://www.aau.at/en/english/) system that participated in the
[NAACL-FLP-shared-task 2018](https://www.aclweb.org/anthology/W18-0907/) (ST)
on metaphor detection.
The ST was part of the [workshop on processing figurative
language](https://sites.google.com/site/figlangworkshop/) at the [16th annual
conference of the *North American Chapter of the Association for Computational
Linguistics* (NAACL2018)](http://naacl2018.org/).

The system combines a small assertion of trending techniques, which implement
matured methods from NLP and ML; in particular, the system uses word embeddings
from standard corpora and from corpora representing different proficiency
levels of language learners in a LSTM BiRNN architecture.

The full system description is [part of the
proceedings](https://www.aclweb.org/anthology/W18-0918/).
```
@inproceedings{stemle-onysko:2018:naacl-flpst,
 author = {Stemle, Egon and Onysko, Alexander},
 booktitle = {Proceedings of the Workshop on Figurative Language Processing},
 date = {2018-06},
 doi = {10.18653/v1/W18-0918},
 eventtitle = {Workshop on Figurative Language Processing},
 location = {New Orleans, LA},
 pages = {133--138},
 publisher = {Association for Computational Linguistics},
 title = {Using Language Learner Data for Metaphor Detection},
 url = {http://aclweb.org/anthology/W18-0918}
}
```

## Word embeddings


## Reproducing our results

For example, using the BNC word embeddings file [English (British National Corpus, 100 million
tokens), Word form \[character
ngrams](https://embeddings.sketchengine.co.uk/static/models/word/bnc2_tt2.bin)
and running the application:
```
$ python3 run.py --embeddings wiki.en.bin --pos --evaluate (-v)
```
should produce something like:
```
0.9451438748196399
0.9503945852684172
0.951720339145739
94.91    +/- 0.28
```
which is only 3-fold CV (as per the app's default) but is on the right track when compared to the 10-fold CV results we reported.

```
0.941254297858
0.950919731608
0.946648737272
0.946342582666
0.950634256269
0.944976784954
0.95125307985
0.946258148831
0.942959071433
0.948779042716
94.70   +/- 0.32
```


Using the wiki word embeddings file [Wiki word vectors, Englich](https://fasttext.cc/docs/en/crawl-vectors.html) and running:
https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.en.zip
```
python3 run.py --embeddings ../data/wiki.en.bin --pos --predict --predictfile wiki_predictions.csv (-v)
```

# ERRATUM

The paper falls short of pointing out the fact that we part-of-speech tag the
training & testing data. Indeed, the paper only states that "Also note that *all
results* in this paper refer only to the all content part-of-speech task." -
which does absolutely nothing to enlighten the reader about the part-of-speech
tagging of the data (as a pre-processing step).

## What (additionally) happens

The training and testing data gets pos-tagged (with plain
[`nltk.pos_tag()`](https://www.nltk.org/_modules/nltk/tag.html#pos_tag)), the
pos tags are converted into categorical data, and these features are added to
the other word embedding features. 
The word embeddings are not pos tagged (would be difficult anyways...) and only
plain non-pos-tagged word embeddings are used.

The idea was that it should be much easier to fine-tune a pos tagger for the
relatively small actual data compared to the 'unlabeled data' encoded
in the word embeddings.

Also note that we use nltk 
```
 Bird, Steven, Edward Loper and Ewan Klein (2009).
 Natural Language Processing with Python.  O'Reilly Media Inc.
```
