import sys
import string

import gensim
import pandas as pd

from gensim import corpora, similarities
from gensim.similarities import SparseMatrixSimilarity, MatrixSimilarity
from gensim.models import TfidfModel, LsiModel, doc2vec
import re

def print_top5(top5):
    for i in range(len(top5)):
        print('#', i + 1, '  Python entity: ', top5[i][1])
        print('File: ', top5[i][2])
        print('Line: ', top5[i][3])
        print('Comment: ', top5[i][5], '\n')

def top_five(similarity, df):
    sorted_similarities = sorted(range(len(similarity)), key=lambda k: similarity[k], reverse=True)
    top5 = [df.iloc[entity].values.tolist() for entity in sorted_similarities[:5]]
    return top5


def top_five_doc2vec(similarity, df):
    topfive = []
    for label, index in [('FIRST', 0), ('SECOND', 1), ('THIRD', 2), ('FOURTH', 3), ('FIFTH', 4)]:
        topfive.append(df.iloc[similarity[index][0]].values.tolist())

    return topfive


def freq(corpus):
    dictionary = corpora.Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
    freqIndex = SparseMatrixSimilarity(corpus_bow, num_features=len(dictionary))

    return dictionary, freqIndex


def freq_similarity(dictionary, freq_index, query):
    bow = dictionary.doc2bow(query.lower().split())
    similarity = freq_index[bow]

    return similarity


def tf_idf(corpus):
    dictionary = corpora.Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
    tfidf = TfidfModel(corpus_bow)
    tf_idf_index = SparseMatrixSimilarity(tfidf[corpus_bow], num_features=len(dictionary))

    return dictionary, tf_idf_index


def tf_idf_similarty(dictionary, tf_idf_index, query):
    bow = dictionary.doc2bow(query.lower().split())
    similarity = tf_idf_index[bow]

    return similarity


def lsi(corpus):
    dictionary = corpora.Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]

    tfidf = TfidfModel(corpus_bow)
    corpus_tfidf = tfidf[corpus_bow]

    model = LsiModel(corpus_tfidf, id2word=dictionary, num_topics=300)
    corpus_lsi = model[corpus_tfidf]

    return dictionary, corpus_lsi, tfidf, model


def lsi_similarity(dictionary, corpus_lsi, tfidf, model, query):
    vec_bow = dictionary.doc2bow(query.lower().split())

    vec_lsi = model[tfidf[vec_bow]]

    index = MatrixSimilarity(corpus_lsi)
    # index.save('./deerwester.index')
    # index = MatrixSimilarity.load('./deerwester.index')
    similarity = index[vec_lsi]

    return similarity, index


def read_corpus(corpus):
    for i, line in enumerate(corpus):
        line = " ".join(line)
        tokens = gensim.utils.simple_preprocess(line)
        yield gensim.models.doc2vec.TaggedDocument(tokens, [i])


def doc2vec(corpus):
    train_corpus = list(read_corpus(corpus))
    model = gensim.models.doc2vec.Doc2Vec(vector_size=300, min_count=2, epochs=40)
    model.build_vocab(train_corpus)
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

    return model


def doc2vec_similarity(model, query):
    vector = model.infer_vector(query.lower().split())
    similarity = model.docvecs.most_similar([vector], topn=5)

    return similarity


def get_corpus():
    df = pd.read_csv("data.csv")
    stopWords = ['test', 'tests', 'main', 'is']
    corpus = []
    for index, row in df.iterrows():
        name = row['name']
        splitByUnderscore = re.sub('([A-Z][a-z]+)', r' \1', re.sub('([A-Z]+)', r' \1', name)).split()
        splitted_words = []
        for word in splitByUnderscore:
            splitted = word.lower().split('_')
            if splitted is not None:
                removeStopWords = [w for w in splitted if w not in stopWords]
                comments = []
                if (pd.notna(row['comment'])):
                    # print(row['comment'])
                    comments = row['comment'].lower()

                    comments = comments.replace('_', ' ')
                    comments = comments.replace('-', ' ')
                    comments = comments.translate(str.maketrans('', '', string.punctuation))
                    comments = comments.split()

                # print(comments)
                splitted_words.extend(removeStopWords)
                splitted_words.extend(comments)
        corpus.append(splitted_words)
    return corpus, df


def main():
    print(sys.argv[1])
    # query = "AST Visitor that looks for specific API usage without editing anything"
    # print(df['name'])
    query = sys.argv[1]
    corpus, df = get_corpus()

    # print(corpus)

    print("\nFreq start...\n")
    freq_dictionary, freq_index = freq(corpus)
    freq_sims = freq_similarity(freq_dictionary, freq_index, query)
    top5 = top_five(freq_sims, df)
    print_top5(top5)
    print("Freq end...\n")

    print("\nTf_idf start...\n")
    tf_idf_dictionary, tf_idf_index = tf_idf(corpus)
    tf_idf_sims = tf_idf_similarty(tf_idf_dictionary, tf_idf_index, query)

    top5 = top_five(tf_idf_sims, df)
    print_top5(top5)
    print("Tf_idf end...\n")

    print("\nLSI start...\n")
    lsi_dictionary, corpus_lsi, tfidf, lsi_model = lsi(corpus)
    lsi_sims, lsi_index = lsi_similarity(lsi_dictionary, corpus_lsi, tfidf, lsi_model, query)

    top5 = top_five(lsi_sims, df)
    print_top5(top5)
    print("LSI end...\n")

    print("\nDoc2Vec start...\n")
    d2v_model = doc2vec(corpus)
    d2v_similarity = doc2vec_similarity(d2v_model, query)

    topfive = top_five_doc2vec(d2v_similarity, df)

    print_top5(topfive)
    print("Doc2Vec end...\n")






if __name__ == "__main__":
    main()
