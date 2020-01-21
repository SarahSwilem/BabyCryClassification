from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score
import pandas as pd


filename = 'Data.csv'
# Remember to remove the header in this case!
# otherwise, you are going to miss a sample from the dataset!
dataset = pd.read_csv(filename, header=None)

X_data = dataset.iloc[:, 0:12].values
y_label = dataset.iloc[:, 12].values

knn = KNeighborsClassifier(n_neighbors=3)
y_pred = cross_val_predict(knn, X_data, y_label, cv=5) # This function does it all for you!
print(confusion_matrix(y_label, y_pred))
print(classification_report(y_label, y_pred))
print(accuracy_score(y_label, y_pred))

