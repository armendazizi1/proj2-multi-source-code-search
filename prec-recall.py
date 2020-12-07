from search_data import get_corpus, freq, tf_idf, lsi, doc2vec, read_corpus, top_five, top_five_doc2vec, \
    freq_similarity, tf_idf_similarty, lsi_similarity, doc2vec_similarity, print_top5

import seaborn as sns
from matplotlib import pyplot
from sklearn.manifold import TSNE
import pandas as pd


def extract_ground_truth():
    queries = []
    entity_names = []
    file_names = []

    file1 = open('ground-truth-unique.txt', 'r')
    count = 0
    while True:
        # Get next line from file
        line = file1.readline()

        # if line is empty
        # end of file is reached
        if not line:
            break
        if line.strip() == '':
            # print('empty')
            continue

        if count == 0:
            queries.append(line.strip())
        elif count == 1:
            entity_names.append(line.strip())
        elif count == 2:
            file_names.append(line.strip())
            count = -1
        count += 1
    file1.close()

    return queries, entity_names, file_names


def main():
    corpus, df = get_corpus()

    queries, entity_names, file_names = extract_ground_truth()

    ####### TRAINING THE SEARCH ENGINES
    freq_dictionary, freq_index = freq(corpus)

    tf_idf_dictionary, tf_idf_index = tf_idf(corpus)

    lsi_dictionary, corpus_lsi, tfidf, lsi_model = lsi(corpus)

    d2v_model = doc2vec(corpus)

    search_engines = ["FREQ","TF-IDF", "LSI", "Doc2Vec"]
    # search_engines = ["Doc2Vec"]

    lsi_vectors = []
    doc2vec_vectors = []
    hues = []


    ## get hues
    for query in queries:
        hues.extend([query] * 6)

    ####### EVALUATING THE SEARCH ENGINES
    for search_engine in search_engines:

        print("\n", search_engine, " start...")
        precision = 0
        correct_answers = 0
        recall = 0
        ground_truth_counter = 0

        for query in queries:
            # print("\nquery: ", query, '\n')
            if (search_engine == "FREQ"):
                freq_sims = freq_similarity(freq_dictionary, freq_index, query)
                topFive = top_five(freq_sims, df)

            elif (search_engine == "TF-IDF"):
                tf_idf_sims = tf_idf_similarty(tf_idf_dictionary, tf_idf_index, query)
                topFive = top_five(tf_idf_sims, df)



            elif (search_engine == "LSI"):
                lsi_sims, lsi_index = lsi_similarity(lsi_dictionary, corpus_lsi, tfidf, lsi_model, query)
                topFive = top_five(lsi_sims, df)

                ## PRINT TOP 5
                # print_top5(topFive)

                corpus_bow = lsi_dictionary.doc2bow(query.lower().split())
                vector = lsi_model[corpus_bow]
                similarity = abs(lsi_index[vector])

                embedding_lsi = [lsi_model[corpus_bow]]

                sorted_similarities = sorted(range(len(similarity)), key=lambda k: similarity[k], reverse=True)
                embedding_lsi = embedding_lsi + [corpus_lsi[idx] for idx in sorted_similarities[:5]]

                for vector in embedding_lsi:
                    vectors = []
                    for idx, value in vector:
                        # print(idx, value)
                        vectors.append(value)
                    lsi_vectors.append(vectors)



            elif (search_engine == "Doc2Vec"):
                d2v_similarity = doc2vec_similarity(d2v_model, query)
                topFive = top_five_doc2vec(d2v_similarity, df)

                ## PRINT TOP5
                # print_top5(topFive)

                vector = d2v_model.infer_vector(query.lower().split())
                sorted_similarities = [doc for doc in d2v_model.docvecs.most_similar([vector], topn=5)]

                embedding_doc2vec = [d2v_model.infer_vector(query.lower().split())]

                for idx, score in sorted_similarities:
                    corpus_doc = corpus[idx]
                    embedding_doc2vec.append(d2v_model.infer_vector(corpus_doc))

                for vec in embedding_doc2vec:
                    doc2vec_vectors.append(vec)


            # for i in topFive:
            #     print(i)

            for i in range(len(topFive)):
                entity = topFive[i][1]
                file = topFive[i][2]
                if (entity == entity_names[ground_truth_counter] and file == file_names[ground_truth_counter]):
                    correct_answers += 1
                    precision += 1 / (i + 1)
                    break
            # print("prec: ", precision)
            # print()
            ground_truth_counter += 1

        # print("correct answers: ",correct_answers)

        average_precision = precision / len(queries)
        recall = correct_answers / (len(queries))

        print("precision: ", average_precision)
        print("recall", recall)

        print("\n", search_engine, " end...")

    produce_plot('LSI', lsi_vectors, hues)
    produce_plot('DOC2VEC', doc2vec_vectors, hues)
    # print(queries)


def produce_plot(search_engine, vectors, hues):
    tsne = TSNE(n_components=2, verbose=0, perplexity=2, n_iter=3000)
    tsne_results = tsne.fit_transform(vectors)
    df = pd.DataFrame()
    df['x'] = tsne_results[:, 0]
    df['y'] = tsne_results[:, 1]
    pyplot.figure(figsize=(9, 9))
    print("len", len(hues))
    sns.scatterplot(
        x="x", y="y",
        hue=hues,
        palette=sns.color_palette("husl", n_colors=10),
        data=df,
        legend="full",
        alpha=1.0
    )

    pyplot.savefig(search_engine + ".png")


if __name__ == "__main__":
    main()
