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
import os, random


def preprocessing(df):
#    cat_features = ['SequenceName', 'TagIdentificator']
#    for col in cat_features:
#        df[col] = df[col].astype('object')
#    X_cat = df[cat_features]
#    X_cat = pd.get_dummies(X_cat)    #one-hot encoding
    #print(X_cat.head())

    scale_X = StandardScaler()    #standardization
    num_features = ['Timestamp', 'x', 'y', 'z']
    X_num = scale_X.fit_transform(df[num_features])
#    X_num = df[num_features]
    col_mean = np.nanmean(X_num, axis=0)
    inds = np.where(np.isnan(X_num))
    X_num[inds] = np.take(col_mean, inds[1])   #nan insert value
    new_X = scale_X.inverse_transform(X_num)
#    print(new_X[0])
    X_num = normalize(X_num, norm='l2')
    X_num = pd.DataFrame(data=X_num, columns=num_features, index=df.index)
    #print(X_num.head())
    X = X_num
#    X = pd.concat([X_cat, X_num], axis=1, ignore_index=False)
    #print(scale_X.inverse_transform())
    y = df['activity']
    #print(X.head())
    #print(X.shape)
    #print(y.shape)

    return X, y


#print(X_train.shape, y_train.shape)
#print(X_test.shape, y_test.shape)


#clfs = {
#        'K_neighbor': neighbors.KNeighborsClassifier(),
#        'decision_tree': tree.DecisionTreeClassifier(min_samples_leaf=k),
#        'naive_gaussian': naive_bayes.GaussianNB(),
#        'svm': svm.SVC(),
#        'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5),
#        'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),
#        'random_forest': RandomForestClassifier(n_estimators=50),
#        'adaboost': AdaBoostClassifier(n_estimators=50),
#        'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
#        }

def find_point(Xsets):
    DistanceList = []
    for i in range(len(Xsets)):
        temp_dist = 0
        for j in range(len(Xsets)):
            temp_dist = temp_dist + np.linalg.norm(Xsets[i] - Xsets[j])
        DistanceList.append(temp_dist)
    return DistanceList.index(min(DistanceList))
    
def find_random_point(Xsets):
    return random.randint(0, len(Xsets) - 1)

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
    #    temp_index =
        if (len(group_matrix[ind]) == 0):
            continue
        for res_node in group_matrix[ind]:
            typical_node.append(X.to_numpy()[res_node])
#        if (len(typical_node) == 0):
#            print("fail")
        temp_index = find_point(typical_node)
#        temp_index = find_random_point(typical_node)
        represent_point_ind.append(temp_index)
    print("Finish Point selection")
    file_name = "tempX"
    if os.path.exists(file_name):
        os.remove(file_name)
    for ind in tqdm(range(clf.get_n_leaves())):
    #    for i in range(len(group_matrix[ind])):
        temp_list = []
        temp_list.append(X.to_numpy()[represent_point_ind[ind]])
        temp_pd_data = pd.DataFrame(temp_list, columns = X.columns.values)#, columns = X.columns())
        #if ((ind == 0) & (i == 0)):
        if (ind == 0):
            temp_pd_data.to_csv(file_name, mode='a+', index=False)
        else:
            temp_pd_data.to_csv(file_name, mode='a+', index=False, header=False)
            #res.append(X.to_numpy()[represent_point_ind[index]])
        #res_label.append(y.to_numpy()[represent_point_ind[ind]])
            #print(y.to_numpy()[represent_point_ind[ind]])
        res_label.append(clf.predict(temp_pd_data)[0])
            #print(clf.predict(temp_pd_data)[0])
            #print(type(y.to_numpy()[represent_point_ind[ind]]))
        #gc.collect()
    print("Finish Data Rebuild")
    print("new data:", len(res_label))
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


def PredictionTest(X_train, X_test, y_train, y_test, k):
    clf_tree = tree.DecisionTreeClassifier(min_samples_leaf=k)
    clf_tree.fit(X_train, y_train.ravel(), )
    predict = np.divide((y_test == clf_tree.predict(X_test)).sum(), y_test.size, dtype = float)
    return predict, clf_tree

if __name__ == '__main__':
    path = "./Data/UCI/ConfLongDemo_JSI.txt"
    df = pd.read_table(path, sep=',')
    k = 50
    X, y = preprocessing(df)
    print("NUM: ", len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.01, shuffle=True)
    print('\nthe classifier is:', 'decision_tree')
    prediction, clf = PredictionTest(X_train, X_test, y_train, y_test, k)
    #print("the prediction is:", prediction)
    print("Begin Create!")
    #if (os.path.exists('tempX'):
    #    
    #else:
    #    X_new_train, y_new_train = get_new_data(clf, X_train, y_train)
    X_new_train, y_new_train = get_new_data(clf, X_train, y_train)
    print("Create End!")
    X_select_train, X_rest_train, y_select_train, y_rest_train = train_test_split(X_train, y_train, random_state = None, test_size = 0.015, shuffle=True)
    print("The number of new train:", len(y_new_train))
    print("The number of old train:", len(y_rest_train))
    new_prediction, new_clf = PredictionTest(X_new_train, X_test, y_new_train, y_test, 5)
    old_prediction, old_clf = PredictionTest(X_rest_train, X_test, y_rest_train, y_test, 5)
    print("the old prediction is:", old_prediction)
    print("the new prediction is:", new_prediction)
#    print("the new score is: ", new_clf.score(X_test, y_test))
# for clf_key in clfs.keys():
#     print('\nthe classifier is:', clf_key)
#     clf = clfs[clf_key]
#     #print(type(y_train.ravel()))
#     #print(np.isnan(X_train).any())
#     #print(np.count_nonzero(np.isnan(X_train)))
#     #print(np.isnan(X_train))
#     begin = time.process_time()
#     clf.fit(X_train, y_train.ravel(), )
#     elapsed = time.process_time() - begin
#     #print(X_test)
#     #print(y_test.ravel())
#     prediction = np.divide((y_train == clf.predict(X_train)).sum(), y_train.size, dtype = float)
# #    res_leaf = clf.apply(X_test)
# #    print(res_leaf[3])
# #    print('the score is:', res_leaf)
# #    print('Tree leafs: ', clf.get_n_leaves())
# #    print('the elapsed is:', elapsed)
# #    print(X_test.to_numpy())
#     print("Begin Create!")
#     X_new_train, y_new_train = get_new_data(clf, X_train, y_train)
#     print("Create End!")
#     PredictionTest(X_new_train, X_test, y_new_train, y_test)
#     print("the prediction is:", prediction)

    #print(len(get_new_data(clf, X_test)))
