import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn import neighbors
from sklearn import tree
from sklearn import naive_bayes
from sklearn import svm
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier 
import time



path = "./Data/UCI/ConfLongDemo_JSI.txt"
df = pd.read_table(path, sep=',')

#df.head()
#df.info()
def preprocessing(df):
    cat_features = ['SequenceName', 'TagIdentificator']
    for col in cat_features:
        df[col] = df[col].astype('object')
    X_cat = df[cat_features]
    X_cat = pd.get_dummies(X_cat)
    print(X_cat.head())

    scale_X = StandardScaler()
    num_features = ['Timestamp', 'x', 'y', 'z']
    X_num = scale_X.fit_transform(df[num_features])
    col_mean = np.nanmean(X_num, axis=0)
    inds = np.where(np.isnan(X_num))
    X_num[inds] = np.take(col_mean,inds[1])
    #X_num.fillna(X_num.mean())
    
    #imp = SimpleImputer(missing_values='NaN', strategy='mean', fill_value = 0)
    #imp.fit(X_num)
    #X_num = imp.transform(X_num)
    #print(X_num)
     
    X_num = normalize(X_num, norm='l2')
    X_num = pd.DataFrame(data=X_num, columns=num_features, index=df.index)
    print(X_num.head())

    X = pd.concat([X_cat, X_num], axis=1, ignore_index=False)
    y = df['activity']
    print(X.head())
    print(X.shape)
    print(y.shape)

    return X, y

X, y = preprocessing(df)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.10, shuffle=True)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
k = 100
#print(X_test)
#print(y_test.ravel())
#print(type(y_test.ravel()))
#print(type(X_test))
#print(np.isnan(y_test).any())
clfs = {
#        'K_neighbor': neighbors.KNeighborsClassifier(),
        'decision_tree': tree.DecisionTreeClassifier(min_samples_leaf=k),
#        'naive_gaussian': naive_bayes.GaussianNB(),
#        'svm': svm.SVC(),
#        'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5),
#        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),
#        'random_forest': RandomForestClassifier(n_estimators=50),
#        'adaboost': AdaBoostClassifier(n_estimators=50),
#        'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
        }

def find_point(Xsets):
    DistanceList = []
 #   for i in range(len(Xsets))
    


def get_new_data(clf, X_test):
    res_leaf = clf.apply(X_test)
    leaf_node = []
    group_matrix = [[] for i in range(clf.get_n_leaves())]
    res = []
    for index in range(len(res_leaf)):
        if res_leaf[index] not in leaf_node:
            leaf_node.append(res_leaf[index])
        group_matrix[leaf_node.index(res_leaf[index])].append(index)
    #print(group_matrix)
    print(X_test)
    print(type(X_test))
    return res

    

for clf_key in clfs.keys():
    print('\nthe classifier is:', clf_key)
    clf = clfs[clf_key]
    #print(type(y_train.ravel()))
    #print(np.isnan(X_train).any())
    #print(np.count_nonzero(np.isnan(X_train)))
    #print(np.isnan(X_train))
    begin = time.process_time()
    clf.fit(X_train, y_train.ravel(), )
    elapsed = time.process_time() - begin
    #print(X_test)
    #print(y_test.ravel())
    prediction = np.divide((y_train == clf.predict(X_train)).sum(), y_train.size, dtype = float)
    print("the prediction is:", prediction)
#    res_leaf = clf.apply(X_test)
#    print(res_leaf[3])
#    print('the score is:', res_leaf)
#    print('Tree leafs: ', clf.get_n_leaves())
    print('the elapsed is:', elapsed)
    get_new_data(clf, X_test)
