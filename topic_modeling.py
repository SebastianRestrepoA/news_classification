import pandas as pd
import string
from text_processing_fns import remove_stopwords, lemmatization_transform, lowercase_transform, remove_characters
from gensim import corpora, models, similarities
import gensim
from gensim.test.utils import datapath


" 1. ============== load data base ========================= "
news_data = pd.read_json('./News_Category_Dataset_v2.json', lines=True)

" 2. TEXT PROCESSING"

# Drop those rows which has authors and short_description column as empty.
news_data.drop(news_data[(news_data['authors'] == '') & (news_data['short_description'] == '')].index,
               inplace=True)
# Concatenate headline and short description
news_data['information'] = news_data[['headline', 'short_description']].apply(lambda x: ' '.join(x), axis=1)

# transform our text information in lowercase
news_data['information_processed'] = lowercase_transform(news_data.information)

# # Removing punctuation characters such as: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
news_data['information_processed'] = remove_characters(news_data['information_processed'])

# Removing stop words from text
news_data["information_processed"] = remove_stopwords(news_data.information_processed)

# lemmatization_transform
news_data["information_processed"] = lemmatization_transform(news_data.information_processed)

print("lemmatization ok")

news_vocabulary = [news.split() for news in news_data["information_processed"]]
dictionary = corpora.Dictionary(news_vocabulary)  # Creating the term dictionary of our courpus
news_term_matrix = [dictionary.doc2bow(news) for news in news_vocabulary]

print("news_term_matrix ok")

# Instance LDA model
Lda = gensim.models.ldamodel.LdaModel

# Running and Trainign LDA model on the document term matrix.
ldamodel = Lda(news_term_matrix, num_topics=20,
               id2word=dictionary,
               alpha='auto',
               passes=10)
print("LDA model has been trained")
# Save LDA model.
ldamodel.save("ldaModel20topicsalphaAuto.model")

print(ldamodel.print_topics(num_topics=20, num_words=5))
# loaded_model = models.LdaModel.load("ldaModel50topicsalphaAuto.model")

# Evaluate LDA model
topics = [ldamodel[c] for c in news_term_matrix]
