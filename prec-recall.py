from search_data import get_corpus, freq, tf_idf, lsi, doc2vec, read_corpus, top_five, top_five_doc2vec, \
    freq_similarity, tf_idf_similarty, lsi_similarity, doc2vec_similarity


def extract_ground_truth():
    queries = []
    entity_names = []
    file_names = []

    file1 = open('ground-truth.txt', 'r')
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

    ####### EVALUATING THE SEARCH ENGINES
    for search_engine in search_engines:

        print("\n", search_engine, " start...")
        precision = 0
        correct_answers =0
        recall = 0
        ground_truth_counter = 0

        for query in queries:
            if (search_engine == "FREQ"):
                freq_sims = freq_similarity(freq_dictionary, freq_index, query)
                topFive = top_five(freq_sims, df)

            elif (search_engine == "TF-IDF"):
                tf_idf_sims = tf_idf_similarty(tf_idf_dictionary, tf_idf_index, query)
                topFive = top_five(tf_idf_sims, df)
            elif (search_engine == "LSI"):
                lsi_sims = lsi_similarity(lsi_dictionary, corpus_lsi, tfidf, lsi_model, query)
                topFive = top_five(lsi_sims, df)

            elif (search_engine == "Doc2Vec"):
                d2v_similarity = doc2vec_similarity(d2v_model, query)
                topFive = top_five_doc2vec(d2v_similarity, df)
            # for i in topFive:
            #     print(i)

            for i in range(len(topFive)):
                entity= topFive[i][1]
                file = topFive[i][2]
                if(entity == entity_names[ground_truth_counter] and file == file_names[ground_truth_counter]):
                    correct_answers += 1
                    precision += 1/(i+1)
                    break
            # print("prec: ", precision)
            # print()
            ground_truth_counter += 1


        # print("correct answers: ",correct_answers)


        average_precision = precision/len(queries)
        recall = correct_answers/(len(queries))

        print("precision: ", average_precision)
        print("recall", recall)

        print("\n", search_engine, " end...")


    # print(queries)



if __name__ == "__main__":
    main()