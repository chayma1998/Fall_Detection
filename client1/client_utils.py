import os
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import train_test_split
import pickle
from sklearn import metrics

NON_FALL_PATH = r".\nonfall\*"
FALL_PATH = r".\fall\*"
NON_FALL_FOLDER = r".\nonfall"
FALL_FOLDER = r".\fall"
AXIS = 'var'
TEST_SIZE = 0.2
SHUFFLE = True
RANDOM_STATE = None
XGB = r".\xgb_model.pkl"
SVM = r".\svm_model.pkl"


def get_model_params(model):
    """Returns the paramters of a model"""
    params = [
        model.coef_,
    ]
    return params


def set_model_params(model, params):
    """Sets the parameters of a model"""
    if params != []: 
        for i, p in enumerate(np.asarray(params).flatten()):
            if model.__class__.__name__ == "OneClassSVM": 
                model.support_vectors_[0][i] = p
            else:
                model.coef_[i] = p
    return model


def get_f1_score(model, X_test, y_test):
    print('pred: ', model.predict(X_test))
    print('y_test: ', y_test)
    f1 = metrics.f1_score(y_test, model.predict(X_test))
    print("Local F1-score: ", f1)
    return f1

def get_loss(model, X_test, y_test):
    if model.__class__.__name__ == "OneClassSVM":
        labels = np.array([1, -1])
    else:
        labels = model.classes_
    loss = metrics.log_loss(y_test, model.predict(X_test), labels=labels)
    return loss



def prepare_model():
    if not os.listdir(NON_FALL_FOLDER) or not os.listdir(FALL_FOLDER):
        model = pickle.load(open(SVM, 'rb'))
    else:
        model = pickle.load(open(XGB, 'rb'))
    return model


def load_datasets():
    try:
        fall_path = glob(FALL_PATH)
        non_fall_path = glob(NON_FALL_PATH)

        fall_list = []
        non_fall_list = []  

        for fpath in fall_path:
            df_fall = pd.read_csv(fpath)
            fall_list.append(np.asarray(df_fall[AXIS], dtype=np.float64))

        for nfpath in non_fall_path:
            df_nfall = pd.read_csv(nfpath)
            non_fall_list.append(np.asarray(df_nfall[AXIS], dtype=np.float64))

        X = fall_list + non_fall_list
        X = np.asarray(X)

        if not os.listdir(NON_FALL_FOLDER) and os.listdir(FALL_FOLDER):
            y = np.tile(1, len(fall_list))
            y = np.concatenate((y, np.tile(-1, len(non_fall_list)))).astype("uint8")

        if os.listdir(NON_FALL_FOLDER) and not os.listdir(FALL_FOLDER):
            y = np.tile(-1, len(fall_list))
            y = np.concatenate((y, np.tile(1, len(non_fall_list)))).astype("uint8")

        if os.listdir(NON_FALL_FOLDER) and os.listdir(FALL_FOLDER):
            y = np.tile(1, len(fall_list))
            y = np.concatenate((y, np.tile(0, len(non_fall_list)))).astype("uint8")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=SHUFFLE)
        print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}, y_test shape: {y_test.shape}") 

        return X_train, y_train, X_test, y_test, len(fall_list), len(non_fall_list)
    except:
        return [], [], [], [], 0, 0
