import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

def load_datasets(feature_paths, label_paths):
    feature = np.ndarray(shape=(0,41))
    label = np.ndarray(shape=(0,1))

    for file in feature_paths:
        df = pd.read_table(file, delimiter=',', na_values='?', header=None)
        imp = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=0, copy=True)
        imp.fit(df)
        df = imp.transform(df)
        feature = np.concatenate((feature, df))

    for file in label_paths:
        df = pd.read_table(file, header=None)
        label = np.concatenate((label, df))

    label = np.ravel(label)
    return feature, label

if __name__ == '__main__':
    ''' 数据路径 '''
    featurePaths = ['第三题数据集/A/A.feature','第三题数据集/B/B.feature','第三题数据集/C/C.feature','第三题数据集/D/D.feature','第三题数据集/E/E.feature']
    labelPaths = ['第三题数据集/A/A.label','第三题数据集/B/B.label','第三题数据集/C/C.label','第三题数据集/D/D.label','第三题数据集/E/E.label']
    ''' 读入数据  '''
    x_train,y_train = load_datasets(featurePaths[:4],labelPaths[:4])
    x_test,y_test = load_datasets(featurePaths[4:],labelPaths[4:])
    x_train, x_, y_train, y_ = train_test_split(x_train, y_train, test_size = 0.1)

print('Start training knn')
knn = KNeighborsClassifier().fit(x_train, y_train)
print('Training done')
answer_knn = knn.predict(x_test)
print('Prediction done')

print('Start training DT')
dt = DecisionTreeClassifier().fit(x_train, y_train)
print('Training done')
answer_dt = dt.predict(x_test)
print('Prediction done')

print('Start training Bayes')
gnb = GaussianNB().fit(x_train, y_train)
print('Training done')
answer_gnb = gnb.predict(x_test)
print('Prediction done')

print('\n\nThe classification report for knn:')
print(classification_report(y_test, answer_knn))
print('\n\nThe classification report for DT:')
print(classification_report(y_test, answer_dt))
print('\n\nThe classification report for Bayes:')
print(classification_report(y_test, answer_gnb))
