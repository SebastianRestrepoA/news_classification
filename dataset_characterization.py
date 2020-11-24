import pandas as pd
import string
from text_processing_fns import remove_stopwords, lemmatization_transform, fn_calculate_word_frequency_per_category, \
    lowercase_transform, remove_characters

" 1. ============== load data base ========================= "
news_data = pd.read_json('./News_Category_Dataset_v2.json', lines=True)

" 2. TEXT PROCESSING"

# news_data = news_data[:10000]

# transform our text information in lowercase
news_data['short_description_processed'] = lowercase_transform(news_data.short_description)

# # Removing punctuation characters such as: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
news_data['short_description_processed'] = remove_characters(news_data['short_description_processed'])

# Removing stop words from text
news_data["not_stop_words"] = remove_stopwords(news_data.short_description_processed)

# lemmatization_transform

news_data["lemmatization"] = lemmatization_transform(news_data.not_stop_words)

# Drop those rows which has authors and short_description column as empty.
news_data.drop(news_data[(news_data['authors'] == '') & (news_data['short_description'] == '')].index,
               inplace=True)
print("lemmatization ok")

category_frecuency = news_data.groupby(by='category').size()
category_frecuency_df = pd.DataFrame({'category': list(category_frecuency.index),
                                      'frequency': list(category_frecuency.values)}).sort_values(by='frequency',
                                                                                                 ascending=True)
ax = category_frecuency_df.plot.barh(x='category', y='frequency', rot=0)
ax.set_xlabel('Frecuencia')
ax.set_ylabel('Categoria')
ax.set_title("Nro. de noticias por categoria")

fn_calculate_word_frequency_per_category(news_data, True)
