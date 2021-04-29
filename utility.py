from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re


def clean_str(sent):
    sent = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sent)
    sent = re.sub(r"\'s", " is ", sent)
    sent = re.sub(r"\'ve", " have ", sent)
    sent = re.sub(r"n\'t", " not ", sent)
    sent = re.sub(r"\'re", " are ", sent)
    sent = re.sub(r"\'d", " would ", sent)
    sent = re.sub(r"\'ll", " will ", sent)
    sent = re.sub(r"'m", " am ", sent)
    sent = re.sub(r",", " ", sent)
    sent = re.sub(r"!", " ", sent)
    sent = re.sub(r"\(", " ", sent)
    sent = re.sub(r"\)", " ", sent)
    sent = re.sub(r"\?", " ", sent)
    sent = re.sub(r"\s{2,}", " ", sent)
    return sent
    # sent = re.sub(r"[0-9]", "", sent)
    # sent = sent.strip().lower()
    # sent = remove_stop_word(sent)
    # sent = stem_word(sent)
    # return " ".join(sent)


def remove_stop_word(sent):
    sent = [word for word in sent.split() if word not in stopwords.words('english')]
    return sent


def stem_word(sent):
    stemmer = SnowballStemmer("english")
    sent = [stemmer.stem(word) for word in sent]
    return sent
