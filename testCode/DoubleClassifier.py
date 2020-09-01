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
    cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
    for col in cat_features:
        df[col] = df[col].astype('object')
    X_cat = df[cat_features]
    X_cat = pd.get_dummies(X_cat)    #one-hot encoding
    #print(X_cat.head())

    scale_X = StandardScaler()    #standardization
    num_features = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
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
    y = df['weights']
    #print(X.head())
    #print(X.shape)
    #print(y.shape)

    return X, y
def find_point(Xsets):
    DistanceList = []
    for i in range(len(Xsets)):
        temp_dist = 0
        for j in range(len(Xsets)):
            temp_dist = temp_dist + np.linalg.norm(Xsets[i] - Xsets[j])
        DistanceList.append(temp_dist)
    return DistanceList.index(min(DistanceList))
    
def find_random_point(Xsets, upper_limit):
    res_index = []
    if ((len(Xsets) < upper_limit) | (len(Xsets) == upper_limit)):
        return range(len(Xsets))
    while(len(res_index) < upper_limit):
        temp_value = random.randint(0, len(Xsets) - 1)
        if temp_value not in res_index:
             res_index.append(temp_value)
#    print("select Num:", len(res_index))
    return res_index

def get_new_data(clf, X, y):
    res_leaf = clf.apply(X)
    leaf_node = []
#        time.sleep(1000)
    group_matrix = [[] for i in range(clf.get_n_leaves())]
    res = []
    res_label = []
    represent_point_ind = []
    for ind in range(len(res_leaf)):
        if res_leaf[ind] not in leaf_node:
            leaf_node.append(res_leaf[ind])
        group_matrix[leaf_node.index(res_leaf[ind])].append(ind)
    #print(X.to_numpy()[0])
#    print(group_matrix[0])
#    time.sleep(1000)
    print("Get Represent Point Stage End")
    for ind in tqdm(range(clf.get_n_leaves())):
        typical_node = []
    #    temp_index =
        if (len(group_matrix[ind]) == 0):
            continue
        for gm_indx in range(len(group_matrix[ind])):
            typical_node.append(X.to_numpy()[group_matrix[ind][gm_indx]])
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
        #for indx in represent_point_ind[ind]:
        for num in range(len(group_matrix[ind])):
             indx = represent_point_ind[ind]
             temp_list.append(X.to_numpy()[group_matrix[ind][indx]])
             res_label.append(y.to_numpy()[group_matrix[ind][indx]])
        #temp_list.append(X.to_numpy()[represent_point_ind[ind]])
        temp_pd_data = pd.DataFrame(temp_list, columns = X.columns.values)#, columns = X.columns())
        #if ((ind == 0) & (i == 0)):
        if (ind == 0):
            temp_pd_data.to_csv(file_name, mode='a+', index=False)
            #print(temp_pd_data)
        else:
            temp_pd_data.to_csv(file_name, mode='a+', index=False, header=False)
            #res.append(X.to_numpy()[represent_point_ind[index]])
        #res_label.append(y.to_numpy()[represent_point_ind[ind]])
            #print(y.to_numpy()[represent_point_ind[ind]])
        #for indexN in range(len(temp_pd_data)):
        #temp_res = clf.predict(temp_pd_data)
        #for indexItem in range(len(temp_res)):
        #    res_label.append(temp_res[indexItem])
        #res_label.append(clf.predict(temp_pd_data))
        #print(res_label)
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

def get_new_data_from_model(clf, X):
    res_label = clf.predict(X)
    res_Label = pd.Series(res_label)
    return X, res_Label


def PredictionTest(X_train, X_test, y_train, y_test, k):
    clf_tree = tree.DecisionTreeClassifier(min_samples_leaf=k)
    clf_tree.fit(X_train, y_train.ravel(), )
    predict = np.divide((y_test == clf_tree.predict(X_test)).sum(), y_test.size, dtype = float)
    return predict, clf_tree

def ModelPredictionTest(X_train, X_test, y_train, y_test, clf):
    #clf_forest = RandomForestClassifier(n_estimators = 50)
    clf.fit(X_train, y_train)
    score = clf.score(X_test, y_test)
    return score

def BuildModel(X_train, y_train, clf):
    #clf_forest = RandomForestClassifier(n_estimators = 50)
    clf.fit(X_train, y_train)
    return clf



if __name__ == '__main__':
    path = "./Data/UCI/adult.data"
    df = pd.read_table(path, sep=',')
    k_list = [10, 20, 50, 100, 200, 500]
    k = 5
    # print(df[''])
    X, y = preprocessing(df)
    print("NUM: ", len(y))
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.01, shuffle=True)
    clfs = {
#       'K_neighbor': neighbors.KNeighborsClassifier(),
       'decision_tree': tree.DecisionTreeClassifier(min_samples_leaf=k),
#       'naive_gaussian': naive_bayes.GaussianNB(),
#       'svm': svm.SVC(),
#       'bagging_knn': BaggingClassifier(neighbors.KNeighborsClassifier(), max_samples=0.5,max_features=0.5),
#       'bagging_tree': BaggingClassifier(tree.DecisionTreeClassifier(), max_samples=0.5,max_features=0.5),
       'random_forest': RandomForestClassifier(n_estimators=50),
#       'adaboost': AdaBoostClassifier(n_estimators=50),
       'gradient_boost': GradientBoostingClassifier(n_estimators=50, learning_rate=1.0,max_depth=1, random_state=0)
    }
    fp = open('outputR_new.txt', 'a+')
    for clf_index in clfs.keys():
        for k in k_list:
            #print(type(clf_index))
            print("the classifier is:", clf_index)
            fp.write("the classifier is:" + clf_index + "\n")
            now_clf = clfs.get(clf_index) 
            
            print("------------Using prediction result-----------")
            clf_n = BuildModel(X_train, y_train, now_clf)
            X_cre_train, y_cre_train = get_new_data_from_model(clf_n, X_train)
        
            prediction, clf = PredictionTest(X_cre_train, X_test, y_cre_train, y_test, k)
            #print("the prediction is:", prediction)
        
            print("Begin Create!")
            #if (os.path.exists('tempX'):
            #    
            #else:
            #    X_new_train, y_new_train = get_new_data(clf, X_train, y_train)
            X_new_train, y_new_train = get_new_data(clf, X_cre_train, y_cre_train)
            print("Create End!")
            #X_select_train, X_rest_train, y_select_train, y_rest_train = train_test_split(X_train, y_train, random_state = None, test_size = 0.50, shuffle=True)
            print("The number of new train:", len(y_new_train))
            print("The number of old train:", len(y_cre_train))
        #    new_prediction, new_clf = PredictionTest(X_new_train, X_test, y_new_train, y_test, 5)
        #    old_prediction, old_clf = PredictionTest(X_train, X_test, y_train, y_test, 5)
        #    print("the old prediction is:", old_prediction)
        #    print("the new prediction is:", new_prediction)
            old_score = ModelPredictionTest(X_train, X_test, y_train, y_test, now_clf)
            new_score = ModelPredictionTest(X_new_train, X_test, y_new_train, y_test, now_clf)
            print("old score: ", old_score)
            print("new score: ", new_score)
            fp.write("The number of new train:" + str(len(y_new_train)) + "\n")
            fp.write("The number of old train:" + str(len(y_train))+ "\n")
            fp.write("old score: " + str(old_score) + "\n")
            fp.write("new score: " + str(new_score) + "\n")
    fp.close()
