
import numpy as np
import pandas as pd

from sklearn import neighbors
from sklearn import preprocessing
from sklearn import model_selection

df = pd.read_csv('breast-cancer-wisconsin.data')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.3)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
# print(accuracy)

ex_measures = np.array([[4,2,1,1,1,2,3,2,1], [4,2,1,3,1,3,3,2,1]])
ex_measures = ex_measures.reshape(len(ex_measures, -1)

prediction = clf.predict(ex_measures)
print(prediction)
