import codecs
import spacy

nlp = spacy.load('de')


def punct_space(token):
    """
    helper function to eliminate tokens
    that are pure punctuation or whitespace
    """

    return token.is_punct or token.is_space


def line_review(lines):
    """
    generator function to read in reviews from the file
    and un-escape the original line breaks in the text
    """

    for review in lines:
            yield review.replace('\\n', '\n')


def lemmatized_sentence_corpus(filename):
    """
    generator function to use spaCy to parse reviews,
    lemmatize the text, and yield sentences
    """

    for parsed_review in nlp.pipe(
            filename, batch_size=10000, n_threads=8):

        for sent in parsed_review.sents:
            yield u' '.join(
                [token.lemma_ for token in sent if not punct_space(token)])
