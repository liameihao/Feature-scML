from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from joblib import dump
from sklearn.linear_model import LogisticRegression


def model_train(X_train,
                y_train,
                X_test,
                y_test,
                njob,
                get_model=False,
                method=None,
                filename=None):
    X_train,  X_test = scale_data(X_train, X_test)
    if method=="svm":
        clf = SVM(njob)
    elif method=="rf":
        clf = RF(njob)
    elif method=="gnb":
        clf = GNB(njob)
    else:
        clf = LR(njob)
    # train
    clf.fit(X_train, y_train)
    # get best accuracy of train data
    train_accuracy = clf.best_score_
    # predict testdata
    predict_test = clf.predict(X_test)
    # get accuracy of test data
    test_accuracy = accuracy_score(y_test, predict_test)
    # get model
    if get_model:
        model_name = "{}_{}.joblib".format(filename, method)
        dump(clf, model_name)
    print("feature number: {}".format(X_train.shape[1]))
    print("train accuracy: {:.4f}\ntest_accuracy: {:.4f}\nbest parameters: {}".format(train_accuracy, test_accuracy, clf.best_params_))
    return train_accuracy, test_accuracy, clf.best_params_
    

def SVM(njob):
    # classifier
    lpf_svm = SVC(decision_function_shape="ovo", 
                  probability=True, 
                  random_state=0)

    # parameters selection
    C = []
    for i in range(-5, 15 + 1, 2):
        C.append(2**i)
    gamma = []
    for i in range(-15, 3 + 1, 2):
        gamma.append(2**i)
    hp = {'C': C, 'gamma': gamma}
    # grid search
    clf = GridSearchCV(lpf_svm,
                       hp,
                       cv=5,
                       scoring="accuracy",
                       return_train_score=False,
                       n_jobs=njob)
    return clf


def RF(njob):
    # classifier
    lpf_rf = RandomForestClassifier(random_state=0, n_jobs=njob)
    # parameters selection
    parameters = {'criterion': ['gini', 'entropy'],
                  'min_samples_split': [2, 4, 8],
                  'min_samples_leaf': [1, 2, 4],
                  'n_estimators': [10, 50, 100, 150]}
    # grid search
    clf = GridSearchCV(lpf_rf, parameters, cv=5, scoring="accuracy",
                       return_train_score=False)
    return clf


def GNB(njob):
    # classifier
    lpf_gnb = GaussianNB()
    # parameters selection
    parameters = {'var_smoothing': [0.0001, 0.001, 0.01, 0.1, 1]}
    # grid search
    clf = GridSearchCV(lpf_gnb, parameters, cv=5, scoring="accuracy",
                       return_train_score=False, n_jobs=njob)
    
    return clf


def LR(njob):
    # classifier
    reg = LogisticRegression(solver='lbfgs', multi_class="ovr", max_iter=3000)
    # parameters selection
    parameters = {'C': [0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]}
    # grid search
    clf = GridSearchCV(reg, parameters, cv=5, scoring="accuracy",
                       return_train_score=False, n_jobs=njob)
    return clf


def scale_data(X_train, X_test):
    min_max_scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = min_max_scaler.fit(X_train)
    X_train_ = scaler.transform(X_train)
    X_test_ = scaler.transform(X_test)
    return X_train_, X_test_

