import pandas as pd
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from text_processing_fns import feature_engineering, remove_stopwords, lemmatization_transform, lowercase_transform,\
    remove_characters
from classifiers import fn_search_best_svm_classifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# import seaborn as sns

" 1. ============== load data base ========================= "
news_data = pd.read_json('./News_Category_Dataset_v2.json', lines=True)

" 2. TEXT PROCESSING"

# news_data = news_data[:2500]
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


# Drop those rows which has authors and short_description column as empty.
news_data.drop(news_data[(news_data['authors'] == '') & (news_data['short_description'] == '')].index,
               inplace=True)

# news[news['authors'] == ''].groupby(by='category').size()
# news[(news['authors'] == '') & (news['short_description'] == '' )].index

# label encode the target variable to transform non-numerical labels
encoder = LabelEncoder()
y = encoder.fit_transform(news_data.category)  # numerical labels
news_categories = encoder.classes_
nfolds = 10

# Splitting into train sets and test sets
x_train, x_test, y_train, y_test = train_test_split(news_data['information_processed'], y,
                                                    train_size=0.8, test_size=0.2,
                                                    random_state=555,
                                                    stratify=y)

features_matrix = feature_engineering(x_train, tf_idf=True)
train_data = features_matrix['TF-IDF']['matrix']
test_data = features_matrix['TF-IDF']['object'].transform(x_test)

print(" TF-IDF has been executed")

" MODEL DEVELOPMENT "
# best_svm_parameters = fn_search_best_svm_classifier(train_data, y_train, nfolds, 'TF-IDF', display_results=True)
# best_kernel = best_svm_parameters['kernel'][0]
# best_c = best_svm_parameters['C'][0]
# # best_gamma = best_svm_parameters['gamma'][0]
#
# print(str(best_kernel))
# print(str(best_c))
# # print(str(best_gamma))

" MODEL TESTING"

# classifier = SVC(kernel=best_kernel, C=best_c).fit(train_data, y_train)
# y_predict = classifier.predict(test_data)
# print('Accuracy SVC RBF: {}'.format(accuracy_score(y_predict, y_test)))

# Naive-Bayes
classifier = ComplementNB()
classifier.fit(train_data, y_train)
y_predict_nb = classifier.predict(test_data)
print('Accuracy NB: {}'.format(accuracy_score(y_predict_nb, y_test)))

# Linear SVC
model = LinearSVC()
model.fit(train_data, y_train)
y_predict_svc = model.predict(test_data)
print('Accuracy SVC: {}'.format(accuracy_score(y_predict_svc, y_test)))

# Logistic Regression
logistic_Regression = LogisticRegression()
logistic_Regression.fit(train_data, y_train)
y_predict_lr = logistic_Regression.predict(test_data)
print('Accuracy LR: {}'.format(accuracy_score(y_predict_lr, y_test)))

"PERFORMANCE MEASURES"
metrics = classification_report(y_predict_svc, y_test, target_names=news_categories, output_dict=True)
metrics.pop('micro avg', None)
metrics.pop('weighted avg', None)
metrics.pop('accuracy', None)

metrics = pd.DataFrame.from_dict(metrics, orient='index')
writer_metrics = pd.ExcelWriter('metrics_svm_model.xlsx', engine='xlsxwriter')
metrics.to_excel(writer_metrics)
writer_metrics.save()

"ERROR ANALYSIS"
errors_idx = y_predict_svc != y_test
fail_news = pd.concat((x_test[errors_idx].reset_index(drop=True),
                       pd.DataFrame(encoder.inverse_transform(y_test[errors_idx]),
                                    columns=['real_category']),
                       pd.DataFrame(encoder.inverse_transform(y_predict_svc[errors_idx]),
                                    columns=['predicted_category'])),
                      axis=1)
writer_fails = pd.ExcelWriter('fail_utterances_svm_model.xlsx', engine='xlsxwriter')
fail_news.to_excel(writer_fails)
writer_fails.save()
writer_metrics.save()

#########################################################################
# K-FOLD APPROACH

# kf = StratifiedKFold(n_splits=10)
# y_real_total = []
# y_pred_total = []
# fold = 1
# fail_utterances = []
#
# for train, val in kf.split(x, y):
#
#     X_train, X_val, y_train, y_val = x[train], x[val], y[train], y[val]
#     utterances_val, intent_names_val = news_data["information"][val], news_data["information"][val]
#
#     classifier = SVC(kernel=best_kernel, gamma=best_gamma, C=best_c).fit(X_train, y_train)
#
#     # Model training
#     classifier.fit(X_train, y_train)
#
#     # Model prediction
#     y_pred = classifier.predict(X_val)
#     y_pred_total.append(y_pred)
#     y_real_total.append(y_val)
#
#     # to save the utterances that were not recognize correctly
#
#     errors_idx = y_pred != y_val
#     fail_utterances.append(pd.concat((utterances_val[errors_idx].reset_index(drop=True),
#                                       intent_names_val[errors_idx].reset_index(drop=True),
#                                       pd.DataFrame(encoder.inverse_transform(y_pred[errors_idx]),
#                                                    columns=['Predicted intent'])),
#                                      axis=1))

#  The predictions obtained from cross validation procedure are concatenated and the performance measures are estimated.

# y_real_total = np.concatenate(y_real_total)
# y_pred_total = np.concatenate(y_pred_total)
# metrics = classification_report(y_pred_total, y_real_total, target_names=news_categories, output_dict=True)
# metrics.pop('micro avg', None)
# metrics.pop('weighted avg', None)
# fail_utterances = pd.concat(fail_utterances, ignore_index=True)
#
# # save results in excel file
# writer_metrics = pd.ExcelWriter('metrics_svm_model.xlsx', engine='xlsxwriter')
# writer_fails = pd.ExcelWriter('fail_utterances_svm_model.xlsx', engine='xlsxwriter')
#
# metrics = pd.DataFrame.from_dict(metrics, orient='index')
# metrics.to_excel(writer_metrics)
# fail_utterances.to_excel(writer_fails)
# writer_fails.save()
# writer_metrics.save()
#
#
