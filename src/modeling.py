#!/usr/bin/env python

# infile should be a *.txt file with one report per line

# call:
# ./topics_clean.py --infile ~/NLP/TestDaten/Radiology/2017-all-edits.txt --outdir models/ --rerun

## first version from Patrick Wright, now it is continued in notebook/topic-modeling.ipynb

import argparse
import time
import sys
import os
import codecs
import spacy
import pandas as pd
import itertools as it
from gensim.models import Phrases
from gensim.models.word2vec import LineSentence
from gensim.corpora import Dictionary, MmCorpus
from gensim.models.ldamulticore import LdaMulticore
#import pyLDAvis
#import pyLDAvis.gensim
import warnings
import _pickle as pickle
from nltk.corpus import stopwords

parser = argparse.ArgumentParser()
parser.add_argument('--infile')
parser.add_argument('--outdir', default=".")
parser.add_argument('--rerun', action='store_true')

args = parser.parse_args()

infile = args.infile
outdir = args.outdir
rerun = args.rerun

# downlod models with:
# python -m spacy download de
nlp = spacy.load('de')

## functions
def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """

    return token.is_punct or token.is_space

def line_review(filename):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """

    with codecs.open(filename, encoding='utf_8') as f:
        for review in f:
            yield review.replace('\\n', '\n')

def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """

    for parsed_review in nlp.pipe(line_review(filename),
                                  batch_size=10000, n_threads=2):

        for sent in parsed_review.sents:
            yield u' '.join([token.lemma_ for token in sent
                             if not punct_space(token)])

if rerun:
    print("rerunning.", end='')
    sys.stdout.flush()
    time.sleep(.5)
    print(".", end='')
    sys.stdout.flush()
    time.sleep(.5)
    print(".", end='')
    sys.stdout.flush()
    time.sleep(.5)
    print(".")

# phrases
# unigram
unigram_sentences_filepath = outdir + "/unigram.txt"

if rerun:
    with codecs.open(unigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for sentence in lemmatized_sentence_corpus(infile):
            f.write(sentence + '\n')

unigram_sentences = LineSentence(unigram_sentences_filepath)


# bigram
bigram_model_filepath = outdir + "/bigram"
if rerun:
    bigram_model = Phrases(unigram_sentences)
    bigram_model.save(bigram_model_filepath)

# load the finished model from disk
bigram_model = Phrases.load(bigram_model_filepath)

bigram_sentences_filepath = bigram_model_filepath + ".txt"
if rerun:
    with codecs.open(bigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for unigram_sentence in unigram_sentences:
            bigram_sentence = u' '.join(bigram_model[unigram_sentence])
            f.write(bigram_sentence + '\n')

bigram_sentences = LineSentence(bigram_sentences_filepath)


# trigram
trigram_model_filepath = outdir + "/trigram"
if rerun:
    trigram_model = Phrases(bigram_sentences)
    trigram_model.save(trigram_model_filepath)

# load the finished model from disk
trigram_model = Phrases.load(trigram_model_filepath)

trigram_sentences_filepath = trigram_model_filepath + ".txt"
if rerun:
    with codecs.open(trigram_sentences_filepath, 'w', encoding='utf_8') as f:
        for bigram_sentence in bigram_sentences:
            trigram_sentence = u' '.join(trigram_model[bigram_sentence])
            f.write(trigram_sentence + '\n')

trigram_sentences = LineSentence(trigram_sentences_filepath)

trigram_records_filepath = trigram_model_filepath + "_final.txt"

de_stops = stopwords.words('german')
# extend by some custom words
de_stops.extend(["jedoch","sowie","datum"])
# include capitals
tmp_lst = []
for w in de_stops:
    tmp_lst.append(w.title())
de_stops.extend(tmp_lst)

if rerun:
    with codecs.open(trigram_records_filepath, 'w', encoding='utf_8') as f:

        for parsed_record in nlp.pipe(line_review(infile),
                                      batch_size=10000, n_threads=2):

            # lemmatize the text, removing punctuation and whitespace
            unigram_review = [token.lemma_ for token in parsed_record
                              if not punct_space(token)]

            # apply the first-order and second-order phrase models
            bigram_review = bigram_model[unigram_review]
            trigram_review = trigram_model[bigram_review]

            # remove any remaining stopwords
            # list is from nltk
            trigram_review = [term for term in trigram_review
                              if term not in de_stops]

            # write the transformed review as a line in the new file
            trigram_review = u' '.join(trigram_review)
            f.write(trigram_review + '\n')

# LDA
trigram_dictionary_filepath = os.path.join('.','trigram_dict_all_diags.dict')

if rerun:
    trigram_reviews = LineSentence(trigram_records_filepath)

    # learn the dictionary by iterating over all of the reviews
    trigram_dictionary = Dictionary(trigram_reviews)

    # filter tokens that are very rare or too common from
    # the dictionary (filter_extremes) and reassign integer ids (compactify)
    trigram_dictionary.filter_extremes(no_below=10, no_above=0.4)
    trigram_dictionary.compactify()

    trigram_dictionary.save(trigram_dictionary_filepath)

# load the finished dictionary from disk
trigram_dictionary = Dictionary.load(trigram_dictionary_filepath)

trigram_bow_filepath = os.path.join('.', 'trigram_bow_corpus_all_diags.mm')

def trigram_bow_generator(filepath):
    """
    generator function to read reviews from a file
    and yield a bag-of-words representation
    """

    for review in LineSentence(filepath):
        yield trigram_dictionary.doc2bow(review)

if rerun:
    # generate bag-of-words representations for
    # all reviews and save them as a matrix
    MmCorpus.serialize(trigram_bow_filepath,
                       trigram_bow_generator(trigram_records_filepath))

# load the finished bag-of-words corpus from disk
trigram_bow_corpus = MmCorpus(trigram_bow_filepath)

lda_model_filepath = os.path.join('.', 'lda_model_all_diags')

if rerun:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        # workers => sets the parallelism, and should be
        # set to your number of physical cores minus one
        lda = LdaMulticore(trigram_bow_corpus,
                           num_topics=50,
                           id2word=trigram_dictionary,
                           workers=3)

    lda.save(lda_model_filepath)

# load the finished LDA model from disk
lda = LdaMulticore.load(lda_model_filepath)

def explore_topic(topic_number, topn=10):
    """
    accept a user-supplied topic number and
    print out a formatted list of the top terms
    """

    print(u'{:20} {}'.format(u'term', u'frequency') + u'\n')

    for term, frequency in lda.show_topic(topic_number, topn=20):
        print(u'{:20} {:.3f}'.format(term, round(frequency, 3)))


for i in range(1,50):
    print("Topic: " + str(i))
    explore_topic(topic_number=int(i))
    print("\n")


