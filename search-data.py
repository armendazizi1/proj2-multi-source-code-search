import sys
import glob
import os
import ast
import pandas as pd
from pprint import pprint
from collections import defaultdict

from gensim import corpora, similarities
from gensim.corpora import Dictionary
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from gensim.models import TfidfModel, LsiModel, LdaModel, doc2vec
import re

def freq(corpus):
    dictionary = corpora.Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
    freqIndex = similarities.SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))
    return dictionary, freqIndex


def top_five(similarity, df):
    sorted_similarities = sorted(range(len(similarity)), key=lambda k: similarity[k], reverse=True)
    top5 = [df.iloc[entity].values.tolist() for entity in sorted_similarities[:5]]
    return top5



def main():
    df = pd.read_csv("data.csv")
    print(sys.argv[1])
    # query = "Optimizer that implements the Adadelta algorithm"
    # print(df['name'])
    query = sys.argv[1]
    stopWords = ['test', 'tests', 'main', 'is']
    corpus = []
    for index, row in df.iterrows():
        name = row['name']
        splitByUnderscore = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
        splitted_words=[]
        for word in splitByUnderscore:
            splitted = word.lower().split('_')
            if splitted is not None:
                removeStopWords = [w for w in splitted if w not in stopWords]
                comments =[]
                if(pd.notna(row['comment'])):
                    # print(row['comment'])
                    comments = row['comment'].lower().split()
                # print(comments)
                splitted_words.extend(removeStopWords)
                splitted_words.extend(comments)
        corpus.append(splitted_words)


    # print(corpus)

    print("\nFreq start...")
    freqDictionary, freqIndex = freq(corpus)
    bow = freqDictionary.doc2bow(query.lower().split())
    similarity = freqIndex[bow]

    top5 = top_five(similarity,df)
    for i in top5:
        print(i)

    print("Freq end...\n")


if __name__ == "__main__":
    main()