import xlrd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier


workbook = xlrd.open_workbook('email_data.xls')
worksheet = workbook.sheet_by_index(0)

train_data = []
train_target = []

for i in range(1, 100):
    train_data.append(worksheet.cell(i , 1).value + worksheet.cell(i , 5).value + worksheet.cell(i , 6).value)

for i in range(1, 100):
    train_target.append(int(worksheet.cell(i , 3).value))

workbook = xlrd.open_workbook('email_test.xls')
worksheet = workbook.sheet_by_index(0)
test_data = []

for i in range(1, 100):
   test_data.append(worksheet.cell(i , 1).value + worksheet.cell(i , 4).value + worksheet.cell(i , 5).value)

text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                     alpha=1e-3, n_iter=5, random_state=42)),
                     ])

a = np.array( train_target )
text_clf = text_clf.fit(train_data, a)

predicted = text_clf.predict(test_data)

for i in range(0, 99):
    print predicted[i]
