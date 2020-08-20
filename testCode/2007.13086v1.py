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
from tqdm import trange, tqdm
import gc, os
path = "./Data/UCI/ConfLongDemo_JSI.txt"
df = pd.read_table(path, sep=',')

def preprocessing(df):
    cat_features = ['SequenceName', 'TagIdentificator']
    for col in cat_features:
        df[col] = df[col].astype('object')
    X_cat = df[cat_features]
    X_cat = pd.get_dummies(X_cat)
    #print(X_cat.head())

    scale_X = StandardScaler()
    num_features = ['Timestamp', 'x', 'y', 'z']
    X_num = scale_X.fit_transform(df[num_features])
    col_mean = np.nanmean(X_num, axis=0)
    inds = np.where(np.isnan(X_num))
    X_num[inds] = np.take(col_mean,inds[1])
     
    X_num = normalize(X_num, norm='l2')
    X_num = pd.DataFrame(data=X_num, columns=num_features, index=df.index)
    #print(X_num.head())

    X = pd.concat([X_cat, X_num], axis=1, ignore_index=False)
    y = df['activity']
    #print(X.head())
    #print(X.shape)
    #print(y.shape)

    return X, y

X, y = preprocessing(df)


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.10, shuffle=True)
#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)
k = 100

print(type(y_test))
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
    for i in range(len(Xsets)):
        temp_dist = 0
        for j in range(len(Xsets)):
            temp_dist = temp_dist + np.linalg.norm(Xsets[i] - Xsets[j])
        DistanceList.append(temp_dist)
    return DistanceList.index(min(DistanceList))
    


def get_new_data(clf, X, y):
    res_leaf = clf.apply(X)
    leaf_node = []
    group_matrix = [[] for i in range(clf.get_n_leaves())]
    res = []
    res_label = []
    represent_point_ind = []
    for ind in range(len(res_leaf)):
        if res_leaf[ind] not in leaf_node:
            leaf_node.append(res_leaf[ind])
        group_matrix[leaf_node.index(res_leaf[ind])].append(ind)
    #print(X.to_numpy()[0])
    #print(group_matrix)
    print("Get Represent Point Stage End")
    for ind in tqdm(range(clf.get_n_leaves())):
        typical_node = []
        for res_node in group_matrix[ind]:
            typical_node.append(X.to_numpy()[res_node])
        temp_index = find_point(typical_node)
        represent_point_ind.append(temp_index)
    print("Finish Point selection")
#    f = open("tempX", "w")
    file_name = "tempX"
    for ind in tqdm(range(clf.get_n_leaves())):
        for i in tqdm(range(len(group_matrix[ind]))):
            temp_list = []
            temp_list.append(X.to_numpy()[represent_point_ind[ind]])
            temp_pd_data = pd.DataFrame(temp_list, columns = X.columns.values)#, columns = X.columns())
            if (ind == 0 & i == 0):
                  temp_pd_data.to_csv(file_name, mode='a+', index=False)
            else:
                  temp_pd_data.to_csv(file_name, mode='a+', index=False, header=False)
            #res.append(X.to_numpy()[represent_point_ind[index]])
            res_label.append(y.to_numpy()[represent_point_ind[ind]])
        #gc.collect()
    print("Finish Data Rebuild")
#    f.close()
    #print(X_test)
    #print(X_test.columns)
    #print(len(X_test.values))
    #print(X_test.values[0])
    
    #print(np.array(res))
    #res_DataFrame = pd.DataFrame(data = res, columns = X.columns())
    res_DataFrame = pd.read_csv(file_name)
    os.remove(file_name)
    res_Label = pd.Series(res_label)
    return res_DataFrame, res_Label


def PredictionTest(X_train, X_test, y_train, y_test):
    clf_tree = tree.DecisionTreeClassifier(min_samples_leaf=k)
    clf_tree.fit(X_train, y_train.ravel(), )
    predict = np.divide((y_train == clf.predict(X_train)).sum(), y_train.size, dtype = float)
    print("The New Prediction:", predict)


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
#    print(X_test.to_numpy())
    print("Begin Create!")
    X_new_train, y_new_train = get_new_data(clf, X_test, y_test)
    print("Create End!")
    PredictionTest(X_new_train, X_test, y_new_train, y_test)

    #print(len(get_new_data(clf, X_test)))
