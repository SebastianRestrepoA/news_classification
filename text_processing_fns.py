import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import scipy as sp
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import rcParams
import unicodedata
import sys
import pandas as pd
import numpy as np
rcParams.update({'figure.autolayout': True})


# TEXT PROCESSING FUNCTIONS

def count_words(news):

    """ This function computes the number of words in each news belonging to the dataset.

    :param news: pandas series with the texts of the dataset.

    :return: pandas series with the number of words used in each news.


    """
    return news.apply(lambda x: len(str(x).split()))


def remove_stopwords(news):

    """ This function eliminates the stop words from each news belonging to the dataset.

    :param news: pandas series with the texts of the dataset.

    :return: pandas series with the dataset without stop words.


    """
    stop = stopwords.words('english')
    return news.apply(lambda x: " ".join(x for x in x.split() if x not in stop))


def remove_tilde(text):

    """ This function removes the accent mark (or "tilde") from a utterance.

    :param text: string variable with a news.

    :return: string variable with the text without "tildes".


    """

    return ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))


def remove_characters(news):

    """ This function removes the irrelevant puntuation characters (!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~)
    from each text belonging to the dataset.

    :param news: pandas series with the texts of the dataset.

    :return: pandas series with the dataset without irrelevant characters.


    """
    return news.str.replace('[^\w\s]', '')


def lowercase_transform(news):

    """ This function transforms the utterances belonging to the knowledge base to lowercase.

    :param news: pandas series with the texts of the knowledge base.

    :return: pandas series with the knowledge base transformed to lowercase.


    """

    return news.apply(lambda x: " ".join(x.strip().lower() for x in x.split()))


def text_lemmatization(text):

    """ This function apply lemmatization over all the words of a utterance.

    :param text: string variable.

    :return: string variable lemmatized.

    """

    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    words_lemma = [lemmatizer.lemmatize(word) for word in words]

    return ' '.join(words_lemma)


def lemmatization_transform(news):

    """ This function apply lemmatization in every news belonging to the dataset.

    :param news: pandas series with the utterances of the knowledge base.

    :return: pandas series with lemmatized knowledge base.


    """

    return news.apply(lambda x: text_lemmatization(x))


def stemming_transform(news):

    """ This function apply stemming function in every utterance belonging to the knowledge base.

    :param news: pandas series with the news of the knowledge base.

    :return: pandas series with stemmed knowledge base.

    """

    return news.apply(lambda x: text_stemming(x))


def text_stemming(utterance):

    """ This function apply stemming over all the words of a utterance.

    :param utterance: string variable.

    :return: stemmed string variable.

    """

    stemmer = SnowballStemmer("spanish")
    words = word_tokenize(utterance)
    words_stem = [stemmer.stem(word) for word in words]

    return ' '.join(words_stem)


def words_frequency(text, category_name, plot=False):

    """ This function computes word frequency of a text.

    :param text: string variable with all concatenated news of a specific category of the dataset.
    :param category_name: string variable with the category name in analysis.
    :param  plot: this option allows to graph the frequency distribution of the words. Default, False

    :return: pandas dataframe variable with the frequency of every word in the knowledge base.

    """

    # break the string into list of words
    text_list = text.split()

    # gives set of unique words
    unique_words = pd.DataFrame(sorted(set(text_list)), columns=['word_names'])
    unique_words[category_name] = unique_words['word_names'].apply(lambda x: text_list.count(x))
    unique_words = unique_words.sort_values(by=[category_name], ascending=False).set_index('word_names')
    # unique_words.rename(columns={'Counts': category_name})
    print(unique_words.head())

    if plot:
        unique_words.plot(kind='bar')
        plt.title(category_name)
        plt.show()

    return unique_words


def lexical_diversity(text):

    """ This function  calculates lexical richness of a text.

    :param text: string variable.

    :return: float variable with the percentage of lexical diversity of the text in analysis.

    """
    text = text.split()
    return (len(set(text)) / len(text))*100


def extract_bigrams(text):

    """ This function produces bigrams list given a text.

    :param text: string variable.

    :return: list variable with the bigrams obtained from the input text.

    """
    utterance = text.split()
    return list(nltk.bigrams(utterance))


def extract_trigrams(text):

    """ This function produces trigrams list given a utterance.

    :param text: string variable.

    :return: list variable with the trigrams obtained from the input utterance.
    """

    text = text.split()
    return list(nltk.trigrams(text))


def intent_bigrams_frequency(news, category_name):

    """ This function computes the frequency of bigrams in the knowledge base of specific intent.


    :param news: pandas series with the texts of dataset.
           category_name: string variable with the name of the category in analysis.

    :return: pandas dataframe with the frequency of the bigrams in the knowledge base.


    """
    b = news.apply(lambda x: extract_bigrams(x))
    b = b.reset_index(drop=True)
    a = b[0]
    for i in range(1, len(b)):
        a = a + b[i]

    fdist = nltk.FreqDist(a)
    df = pd.DataFrame(fdist.items(), columns=['Bigrams', category_name])

    return df.sort_values(by=[category_name], ascending=False).set_index('Bigrams')


def intent_trigrams_frequency(news, category_name):

    """ This function computes the frequency of trigrams in the knowledge base of specific category.


    :param news: pandas series with the texts of knowledge base.
           category_name: string variable with the name of the intent in analysis.

    :return: pandas dataframe with the frequency of the trigrams in the knowledge base.


    """

    tri = news.apply(lambda x: extract_trigrams(x))
    tri = tri.reset_index(drop=True)
    a = tri[0]
    for i in range(1, len(tri)):
        a = a + tri[i]

    fdist = nltk.FreqDist(a)
    df = pd.DataFrame(fdist.items(), columns=['Trigrams', category_name])
    return df.sort_values(by=[category_name], ascending=False).set_index('Trigrams')


def fn_calculate_word_frequency_per_category(dataset, generate_excel=False):

    """ This function computes the frequency of words, bigrams and trigrams in every category.


    :param dataset: Da variable with local path where is saved the knowledge base.
           generate_excel: this option generates a excel file with the words, bigrams and
           trigrams frequency of the dataset. Default False

    :return: pandas dataframe with the words frequency of the knowledge base.


    """
    unique_category = list(set(dataset["category"]))

    lex_diversity = []

    df_old = []
    df_bigrams = []
    df_trigrams = []

    for idx, i in enumerate(unique_category):

        if idx == 0:
            df_bigrams = intent_bigrams_frequency(dataset["lemmatization"][dataset["category"] == i], i)
            df_trigrams = intent_trigrams_frequency(dataset["lemmatization"][dataset["category"] == i], i)
            df_old = words_frequency(dataset["lemmatization"][dataset["category"] == i].str.cat(sep=' '), i, plot=False)
        else:
            df_bigrams_new = intent_bigrams_frequency(dataset["lemmatization"][dataset["category"] == i], i)
            df_trigrams_new = intent_trigrams_frequency(dataset["lemmatization"][dataset["category"] == i], i)
            df_new = words_frequency(dataset["lemmatization"][dataset["category"] == i].str.cat(sep=' '), i, plot=False)

            df_old = pd.concat((df_old, df_new), axis=1)
            df_bigrams = pd.concat((df_bigrams, df_bigrams_new), axis=1)
            df_trigrams = pd.concat((df_trigrams, df_trigrams_new), axis=1)

    df_final = df_old.fillna(0).sort_values(by=unique_category, ascending=False)
    bigrams_final = df_bigrams.fillna(0).sort_values(by=unique_category, ascending=False)
    trigrams_final = df_trigrams.fillna(0).sort_values(by=unique_category, ascending=False)

    if generate_excel:
        writer = pd.ExcelWriter('Word frequency analysis.xlsx', engine='xlsxwriter')
        df_final.to_excel(writer, sheet_name='word frequency')
        bigrams_final.to_excel(writer, sheet_name='bigrams frequency')
        trigrams_final.to_excel(writer, sheet_name='trigrams frequency')
        writer.save()
        print("excel saved correctly")

    return df_final, bigrams_final, trigrams_final


def fn_get_word_frequency(df_wordfreq, utterance, real_intent, pred_intent, bigrams=False, trigrams=False):

    words = utterance.split(' ')

    if bigrams:
        words = list(nltk.bigrams(words))

    if trigrams:
        words = list(nltk.trigrams(words))

    return df_wordfreq.loc[words, [real_intent, pred_intent]]


def fn_get_jaccard_sim(vStr1, vStr2):

    """ This function computes the jaccard similarity coefficient between two utterances (or strings).

    :param vStr1: string variable.
           vStr2: string variable.
    :return float variable with the jaccard similarity coefficient.

    """
    a = set(vStr1.split())
    b = set(vStr2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def fn_get_vectors(strs):
    text = [t for t in strs]
    vectorizer = CountVectorizer(text)
    vectorizer.fit(text)
    return vectorizer.transform(text).toarray()


def fn_get_cosine_sim(vUtterancesKnowledgeBase):

    """ This function computes the cosine similarity between the utterances of the knowledge base.

    :param vUtterancesKnowledgeBase: list variable with the utterances of the knowledge base (.xlsx file)

    :return ndarray matrix with the cosine similarity estimated from the knowledge base.

    """

    vectors = [t for t in fn_get_vectors(vUtterancesKnowledgeBase)]
    return cosine_similarity(vectors)


def fn_utterances_similarity_between_intents(vPathKnowledgeBase, vThreshold, output_file_name):

    """ This function computes the jaccard and cosine similarity index between the utterances of the knowledge base and
     it generates a excel file with the utterances that achieved an jaccard index above setted threshold.


    :param vPathKnowledgeBase: string variable with local path where is saved the knowledge base (.xlsx file)
           vThreshold: float variable used to filter the utterances with similarity below of this number.

    """

    writer = pd.ExcelWriter(vPathKnowledgeBase[0:vPathKnowledgeBase.rfind('/') + 1] +
                            output_file_name + '.xlsx', engine='xlsxwriter')

    df = pd.read_excel(vPathKnowledgeBase)

    df['Utterance'] = lowercase_transform(df['Utterance'])
    df = df.reset_index()
    # df = df[df.columns[:-2]]
    jaccard_matrix = np.zeros((len(df['Utterance']), len(df['Utterance'])))

    for idx, utterance in enumerate(list(df['Utterance'])):
        jaccard_matrix[:, idx] = df['Utterance'].apply(lambda x: fn_get_jaccard_sim(utterance, x))

    cosine_matrix = fn_get_cosine_sim(list(df['Utterance']))

    df_final = []

    for idx in range(0, df.shape[0]):

        idx_utterances = np.where((jaccard_matrix[idx+1:, idx] > vThreshold) == 1)
        idx_utterances = idx_utterances[0]+idx+1
        utterances_selected = df.iloc[idx_utterances].reset_index(drop=True)
        utterance_to_compare = pd.DataFrame([df['Utterance'][idx]] * utterances_selected.shape[0],
                                            columns=['To compare'])
        intent_to_compare = pd.DataFrame([df['Intent'][idx]] * utterances_selected.shape[0],
                                         columns=['Intent to compare'])
        jaccard_data = pd.DataFrame(jaccard_matrix[idx_utterances, idx], columns=['Jaccard'])
        cosine_data = pd.DataFrame(cosine_matrix[idx_utterances, idx], columns=['Cosine'])

        if utterances_selected is not None:
            df_contenated = pd.concat((intent_to_compare, utterance_to_compare, utterances_selected, jaccard_data,
                                       cosine_data), axis=1).sort_values(by='Jaccard', ascending=False)

            df_final.append(df_contenated[['Intent to compare', 'To compare', 'Utterance', 'Intent', 'Jaccard', 'Cosine',
                               'index', 'Ongoing']])

    df_final = pd.concat(df_final)
    df_final.to_excel(writer)

    writer.save()
    print('Similarity between intents have finished')


def feature_engineering(vUtterances, vectorizer=False, tf_idf=False, ngram=False):

    """ This function coverts utterances into feature matrices such as word2vect, TF-IDF and n-gram for training
    of ML models.

    :param vUtterances: pandas series with the utterances of knowledge base.

    :return: dictionary with 3 features matrices corresponding to the count vector, TF-IDF and n-grams transforms.

    """

    features = {}
    vUtterances.index = range(0, len(vUtterances))

    if vectorizer is True:

        # transform the training data using count vectorizer object
        # create a count vectorizer object
        count_vect = CountVectorizer(analyzer='word')
        count_vect.fit(vUtterances) # Create a vocabulary from all utterances
        x_count = count_vect.transform(vUtterances)  # Count how many times is each word from each utterance in the
        # vocabulary.
        features['count_vectorizer'] = {'object': count_vect, 'matrix': x_count}
        # pd.DataFrame(x_count.toarray(), columns=count_vect.get_feature_names())

    if tf_idf is True:

        # word level tf-idf

        " TF-IDF score represents the relative importance of a term in the document and the entire corpus. "
        tfidf_vect = TfidfVectorizer(analyzer='word', max_features=5000)  # token_pattern=r'\w{1,}'
        tfidf_vect.fit(vUtterances)
        x_tfidf = tfidf_vect.transform(vUtterances)
        features['TF-IDF'] = {'object': tfidf_vect, 'matrix': x_tfidf}

    if ngram is True:

        # ngram level tf-idf
        tfidf_vect_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2, 3), max_features=5000) # token_pattern=r'\w{1,}'
        tfidf_vect_ngram.fit(vUtterances)
        x_tfidf_ngram = tfidf_vect_ngram.transform(vUtterances)
        features['ngram'] = {'object': tfidf_vect_ngram, 'matrix': x_tfidf_ngram}

    return features


def fn_utterances_similarity_two_docs(path_doc1, path_doc2, output_file_name):

    writer = pd.ExcelWriter(output_file_name + '.xlsx')
    utterances_knowledge_base = pd.read_excel(path_doc1)
    utterances_knowledge_base['Utterance'] = lowercase_transform(utterances_knowledge_base['Utterance'])
    utterances_knowledge_base = utterances_knowledge_base.reset_index(drop=True)
    utterances_knowledge_base = utterances_knowledge_base[utterances_knowledge_base.columns[:-2]]

    utterances_doc_to_depure = pd.read_excel(path_doc2)
    utterances_doc_to_depure['Utterance'] = lowercase_transform(utterances_doc_to_depure['Utterance'])
    utterances_doc_to_depure = utterances_doc_to_depure.reset_index(drop=True)
    # utterances_doc_to_depure = utterances_doc_to_depure[utterances_doc_to_depure.columns[:-2]]

    id = utterances_doc_to_depure['Utterance'].isin(utterances_knowledge_base['Utterance'])
    new_doc = utterances_doc_to_depure[id == False]
    # new_doc = utterances_doc_to_depure[id == True]

    writer = pd.ExcelWriter('utterances_No_agregadas.xlsx', engine='xlsxwriter')
    new_doc.to_excel(writer, index=False)
    writer.save()


def dist_raw(v1, v2):
    delta = v1-v2
    return sp.linalg.norm(delta.toarray())


def dist_norm(v1, v2):
    v1_normalized = v1 / sp.linalg.norm(v1.toarray())
    v2_normalized = v2 / sp.linalg.norm(v2.toarray())

    delta = v1_normalized - v2_normalized

    return sp.linalg.norm(delta.toarray())


def similarity_measurement(num_samples, posts, new_post, new_post_vec, X_train, norm=False):

    best_i = None
    best_dist = sys.maxsize

    if norm is True:
        dist = dist_norm
    else:
        dist = dist_raw

    for i in range(0, num_samples):

        post = posts[i]
        if post == new_post:
            continue
        post_vec = X_train.getrow(i)
        d = dist(post_vec, new_post_vec)

        print("=== Post %i with dist=%.2f: %s" % (i, d, post))

        if d < best_dist:
            best_dist = d
            best_i = i

    print("Best post is %i with dist=%.2f" % (best_i, best_dist))
    return best_i, best_dist


def tfidf(t, d, D):
    tf = float(d.count(t)) / sum(d.count(w) for w in set(d))
    idf = sp.log(float(len(D)) / (len([doc for doc in D if t in doc])))
    return tf * idf


