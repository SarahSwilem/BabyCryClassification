import pandas as pd
from   sklearn.model_selection import train_test_split
filename = 'Features.csv'
dataset = pd.read_csv(filename)
X = dataset.iloc[:, 0:26].values
y = dataset.iloc[:, 26].values

#Make a Test-Train-Split of the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=7)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

import _pickle as cPickle
def pickle_model(model, modelname):
    with open('models/' + str(modelname) + '.pkl', 'wb') as f:
        return cPickle.dump(model, f)
model = classifier
pickle_model(model, "myKNN")
