import pandas as pd
import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
nltk.download('stopwords')
import re, string
nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import pos_tag
from collections import defaultdict
from collections.abc import Iterable
from nltk import ngrams
nltk.download('wordnet')

def prepare_data_set(hotelData):
    hotelData["Negative_Review"] = hotelData["Negative_Review"].apply(lambda row: "" if "No Negative" == row else row)
    hotelData["Positive_Review"] = hotelData["Positive_Review"].apply(lambda row: "" if "No Positive" == row else row)

    hotelData["Review"] = hotelData["Positive_Review"] + " " + hotelData["Negative_Review"]

    df = hotelData[["Review", "Reviewer_Score"]]

    df["Reviewer_Score"] = df["Reviewer_Score"].apply(lambda score: binning(score))

    return df


# Binning

def binning(score):
    result = "SOMETHING VERY WEIRD HAPPENED HERE"

    if (score <= 10) & (score >= 8.5):
        return 0
    if (score < 8.5) & (score >= 7.0):
        return 1
    if (score < 7.0) & (score >= 0):
        return 2

    return result

# Different Preprocessing strategies

def tokenize(text, sentenceSeperate=False, includePunctation=False, excludeSpecPuct=[]):
    data = []

    # intern functions
    def withPunctation(text):
        temp = []
        # delete unwanted punctuation
        for delPunct in excludeSpecPuct:
            text = text.replace(delPunct, " ")
        # help tokenization with replacing some untokenized punctations
        for puct in ["-", "/", "—"]:
            text = text.replace(puct, " " + puct + " ")
        # tokenize the sentence into words
        for j in word_tokenize(text):
            temp.append(j)
        return temp

    def withoutPunctation(text):
        token_pattern = re.compile(r"(?u)\b\w\w+\b")  # split on whitespace (and remove punctation)
        return token_pattern.findall(text)

    text = text.lower()

    if sentenceSeperate:
        # iterate through each sentence in the file
        for sentence in sent_tokenize(text):
            if includePunctation:
                data.append(withPunctation(sentence))
            else:
                data.append(withoutPunctation(sentence))
    else:
        if includePunctation:
            data = withPunctation(text)
        else:
            data = withoutPunctation(text)
    return data


def removeStopwords(wordArray):
    my_stopwords = set(stopwords.words('english'))
    withoutStopwords = []

    # test if its a list of words or a list of sentences with words
    if len(wordArray) > 0 and isinstance(wordArray[0], Iterable) and not isinstance(wordArray[0], str):
        for sentence in wordArray:
            withoutStopwords.append(removeStopwords(sentence))

    else:
        for item in wordArray:
            if item not in my_stopwords:
                withoutStopwords.append(item)
    return withoutStopwords


def applyStemming(wordArray):
    stemmer = PorterStemmer()
    stems = []

    # test if its a list of words or a list of sentences with words
    if len(wordArray) > 0 and isinstance(wordArray[0], Iterable) and not isinstance(wordArray[0], str):
        for sentence in wordArray:
            stems.append(applyStemming(sentence))
    else:
        for item in wordArray:
            stems.append(stemmer.stem(item))
    return stems


def applyLemmatizing(
        wordArray):  # Quelle (stark verändert): https://www.guru99.com/stemming-lemmatization-python-nltk.html
    tag_map = defaultdict(lambda: wn.NOUN)
    tag_map['J'] = wn.ADJ
    tag_map['V'] = wn.VERB
    tag_map['R'] = wn.ADV

    # intern function
    def lemmazizeText(text):
        temp = []
        for token, tag in pos_tag(text):
            temp.append(lemma_function.lemmatize(token, tag_map[tag[0]]))
        return temp

    lemma_function = WordNetLemmatizer()
    baseWords = []
    if len(wordArray) > 0 and isinstance(wordArray[0], Iterable) and not isinstance(wordArray[0], str):
        for sentence in wordArray:
            baseWords.append(lemmazizeText(sentence))
    else:
        baseWords = lemmazizeText(wordArray)

    return baseWords


def addNGram(wordArray, NGramLength=2):
    holetext = wordArray
    temp = []
    if len(wordArray) > 0 and isinstance(wordArray[0], Iterable) and not isinstance(wordArray[0], str):
        print("drin")
        for sentence in wordArray:
            temp.append(' '.join(sentence))
        holetext = (' '.join(temp)).split()
    nGrams = list(ngrams(holetext, NGramLength))

    # make nGram from two words to one
    nGramsFull = pd.Series(nGrams).apply(lambda row: ' '.join(row))
    wordArrayCopy = wordArray.copy()
    wordArrayCopy.extend(nGramsFull)
    return (wordArrayCopy)


std_dict = {
    "token": True,  # mandatory True
    "token_sentenceSeperate": False,
    "token_includePunctation": False,
    "token_excludeSpecPuct": [],
    "rem_stpwrds": True,
    "stemm": True,
    "lemmatize": True,
    "nGram": True,
    "nGram_length": 2
}


def preprocess(review, dict):
    if dict["token"]:
        review = tokenize(review, sentenceSeperate=dict["token_sentenceSeperate"],
                          includePunctation=dict["token_includePunctation"],
                          excludeSpecPuct=dict["token_excludeSpecPuct"])
    if dict["remStpwrds"]:
        review = removeStopwords(review)
    if dict["stemm"]:
        review = applyStemming(review)
    if dict["lemmatize"]:
        review = applyLemmatizing(review)
    if dict["nGram"]:
        review = addNGram(review, NGramLength=dict["nGram_length"])

    return review