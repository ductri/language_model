import re


p1 = re.compile('[0-9]+')
p2 = re.compile('\n+')


# def __replace_digit(doc):
#     return p1.sub('__d__', doc)


def __replace_breakline(doc):
    return p2.sub(' ', doc)


def preprocess_text(doc):
    doc = __replace_breakline(doc)
    return doc


def infer_preprocess(doc):
    doc = preprocess_text(doc)
    return doc


def train_preprocess(doc):
    doc = preprocess_text(doc)
    return doc
