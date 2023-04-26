import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop, SGD
from tensorflow.keras.datasets import fashion_mnist, mnist, cifar10
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import pyplot
from PIL import Image, ImageFont, ImageDraw
from typing_extensions import Counter

from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix, accuracy_score, fbeta_score, roc_auc_score, roc_curve,auc, make_scorer, f1_score
from sklearn.neighbors import NearestNeighbors, KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,cross_validate,KFold

import random
import heapq
import math
import time
import sys

from imblearn.over_sampling import SMOTE

def train_clf(clf, param_grid, X_train, y_train, X_test, y_test):

    # define scoring metrics
    scorers = {
        'f1_score': make_scorer(f1_score, average='macro'),
        'roc_auc_score': make_scorer(roc_auc_score),
        'g_mean': make_scorer(balanced_accuracy_score)
    }

    # use GridSearchCV to search for best parameter values
    grid_search = GridSearchCV(clf, param_grid=param_grid, scoring=scorers, refit='f1_score', cv=5)
    grid_search.fit(X_train, y_train)

    # use best estimator to predict test set and calculate ROC AUC score
    y_pred = grid_search.best_estimator_.predict_proba(X_test)[:,1]
    score = calculate_score(y_test, y_pred)
    return score

def train_and_get_score(X_train, y_train, X_test, y_test):
    knn = KNeighborsClassifier()
    param_knn = {'n_neighbors': [3, 5, 7, 9]}
    score_knn = train_clf(knn, param_knn, X_train, y_train, X_test, y_test)

    svc = svm.SVC(kernel='rbf', probability=True, class_weight='balanced')
    param_svc = {'C': [0.001, 0.01, 0.1, 1, 10], 'gamma': [0.1, 0.3, 0.5, 0.7, 1, 2, 5]}
    score_svc = train_clf(svc, param_svc, X_train, y_train, X_test, y_test)

    rf = RandomForestClassifier(criterion='gini')
    param_rf = {'n_estimators': [20, 30, 50, 80, 100], 'max_depth': [5, 10, 13, 15, 20], 'max_features': [int(math.sqrt(X_train.shape[1]))]}
    score_rf = train_clf(rf, param_rf, X_train, y_train, X_test, y_test)

    return (score_knn, score_svc, score_rf)

def get_config(dataset):
    if dataset == 'fmnist' or dataset == 'mnist':
        IMAGE_W = 28
        IMAGE_H = 28
        IMAGE_C = 1
    elif dataset == 'cifar2':
        IMAGE_W = 32
        IMAGE_H = 32
        IMAGE_C = 3
    else:
        IMAGE_W = 128
        IMAGE_H = 128
        IMAGE_C = 3
    return IMAGE_H, IMAGE_W,IMAGE_C

def mean_score(score_clf):
    mean_score_arr = []
    
    for score in score_clf:
        mean_score = []
        score_df = pd.DataFrame(score)
        G_Mean = np.mean(score_df[0].tolist())
        F_Measure = np.mean(score_df[1].tolist())
        Auc = np.mean(score_df[2].tolist())
        bal_acc = np.mean(score_df[3].tolist())
        Precision = np.mean(score_df[4].tolist())
        Sensitivity = np.mean(score_df[5].tolist())
        Specificity = np.mean(score_df[6].tolist())
        
        Acc = np.mean(score_df[7].tolist())
        mean_score.extend([G_Mean, F_Measure, Auc, bal_acc, Precision, Sensitivity, Specificity, Acc])
        mean_score_arr.append(mean_score)
    return mean_score_arr

def print_score(score):
    score_df = pd.DataFrame(score)
    G_Mean_arr = score_df[0].tolist()
    F_Measure_arr = score_df[1].tolist()
    Auc_arr = score_df[2].tolist()
    bal_acc_arr = score_df[3].tolist()
    Precision_arr = score_df[4].tolist()
    Sensitivity_arr = score_df[5].tolist()
    Specificity_arr = score_df[6].tolist()
    Acc_arr = score_df[7].tolist()

    print('G_Mean_arr: mean=%.3f std=%.3f' % (np.mean(G_Mean_arr)*100, np.std(G_Mean_arr)*100))
    print('F_Measure_arr: mean=%.3f std=%.3f' % (np.mean(F_Measure_arr)*100, np.std(F_Measure_arr)*100))
    print('Auc_arr: mean=%.3f std=%.3f' % (np.mean(Auc_arr)*100, np.std(Auc_arr)*100))
    print('bal_acc_arr: mean=%.3f std=%.3f' % (np.mean(bal_acc_arr)*100, np.std(bal_acc_arr)*100))
    print('Precision_arr: mean=%.3f std=%.3f' % (np.mean(Precision_arr)*100, np.std(Precision_arr)*100))
    print('Sensitivity_arr: mean=%.3f std=%.3f' % (np.mean(Sensitivity_arr)*100, np.std(Sensitivity_arr)*100))
    print('Specificity_arr: mean=%.3f std=%.3f' % (np.mean(Specificity_arr)*100, np.std(Specificity_arr)*100))
    print('Acc_arr: mean=%.3f std=%.3f' % (np.mean(Acc_arr)*100, np.std(Acc_arr)*100))

def print_score_clf(score_clf):
    for score in score_clf:
        score_df = pd.DataFrame(score)
        G_Mean_arr = score_df[0].tolist()
        F_Measure_arr = score_df[1].tolist()
        Auc_arr = score_df[2].tolist()
        bal_acc_arr = score_df[3].tolist()
        Precision_arr = score_df[4].tolist()
        Sensitivity_arr = score_df[5].tolist()
        Specificity_arr = score_df[6].tolist()
        
        Acc_arr = score_df[7].tolist()

        print('G_Mean_arr: mean=%.3f std=%.3f' % (np.mean(G_Mean_arr)*100, np.std(G_Mean_arr)*100))
        print('F_Measure_arr: mean=%.3f std=%.3f' % (np.mean(F_Measure_arr)*100, np.std(F_Measure_arr)*100))
        print('Auc_arr: mean=%.3f std=%.3f' % (np.mean(Auc_arr)*100, np.std(Auc_arr)*100))
        print('bal_acc_arr: mean=%.3f std=%.3f' % (np.mean(bal_acc_arr)*100, np.std(bal_acc_arr)*100))
        print('Precision_arr: mean=%.3f std=%.3f' % (np.mean(Precision_arr)*100, np.std(Precision_arr)*100))
        print('Sensitivity_arr: mean=%.3f std=%.3f' % (np.mean(Sensitivity_arr)*100, np.std(Sensitivity_arr)*100))
        print('Specificity_arr: mean=%.3f std=%.3f' % (np.mean(Specificity_arr)*100, np.std(Specificity_arr)*100))
        print('Acc_arr: mean=%.3f std=%.3f' % (np.mean(Acc_arr)*100, np.std(Acc_arr)*100))

def calculate_score(y_true, y_pred):
    score = list()

    y_pred = np.where(y_pred < 0.5, 0, 1)
    
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
    Acc=(TP+TN)/(TP+TN+FP+FN)
    Precision=0.0
    if (TP+FP) != 0:
        Precision = TP/(TP+FP)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure = 0.0
    if (Sensitivity+Precision) != 0:
      F_Measure=2*Sensitivity*Precision/(Sensitivity+Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)

    score.extend([G_Mean, F_Measure, auc, bal_acc, Precision, Sensitivity, Specificity, Acc])
    return score

def load_dataset(path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C):
    if dataset == 'fmnist':
        return make_fmnist_im()
    elif dataset == 'mnist':
        return make_mnist_im()
    elif dataset == 'breakhis':
        return load_breakhis(path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C)
    elif dataset == 'ChestXray':
        return load_chestxray(path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C)
    elif dataset == 'cifar2':
        return make_cifar2_im()
    elif dataset == 'isic2018':
        return load_isic2018(path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C)

def load_breakhis(path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C):
    df1 = pd.read_pickle(path+dataset+"_train_"+dataset_name+".pkl")
    X_train = df1.loc[:, df1.columns[0:IMAGE_W*IMAGE_H*IMAGE_C]].to_numpy()
    X_train = X_train.reshape(-1,IMAGE_W,IMAGE_H,IMAGE_C)
    y_train = df1.loc[:, df1.columns == 'y_train'].to_numpy()
    y_train = y_train.reshape(-1)

    df1 = pd.read_pickle(path+dataset+"_val_"+dataset_name+".pkl")
    X_val = df1.loc[:, df1.columns[0:IMAGE_W*IMAGE_H*IMAGE_C]].to_numpy()
    X_val = X_val.reshape(-1,IMAGE_W,IMAGE_H,IMAGE_C)
    y_val = df1.loc[:, df1.columns == 'y_val'].to_numpy()
    y_val = y_val.reshape(-1)

    y_train[y_train==0] = 2
    y_train[y_train==1] = 0
    y_train[y_train==2] = 1
    y_val[y_val==0] = 2
    y_val[y_val==1] = 0
    y_val[y_val==2] = 1
    return (X_train, y_train), (X_val, y_val)

def load_chestxray(path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C):
    df1 = pd.read_pickle(path+dataset+"_train_"+dataset_name+".pkl")
    X_train = df1.loc[:, df1.columns[0:IMAGE_W*IMAGE_H*IMAGE_C]].to_numpy()
    X_train = X_train.reshape(-1,IMAGE_W,IMAGE_H,IMAGE_C)
    y_train = df1.loc[:, df1.columns == 'y_train'].to_numpy()
    y_train = y_train.reshape(-1)

    df1 = pd.read_pickle(path+dataset+"_val_"+dataset_name+".pkl")
    X_val = df1.loc[:, df1.columns[0:IMAGE_W*IMAGE_H*IMAGE_C]].to_numpy()
    X_val = X_val.reshape(-1,IMAGE_W,IMAGE_H,IMAGE_C)
    y_val = df1.loc[:, df1.columns == 'y_val'].to_numpy()
    y_val = y_val.reshape(-1)

    y_train[y_train==0] = 2
    y_train[y_train==1] = 0
    y_train[y_train==2] = 1
    y_val[y_val==0] = 2
    y_val[y_val==1] = 0
    y_val[y_val==2] = 1
    return (X_train, y_train), (X_val, y_val)

def load_isic2018(path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C):
    df1 = pd.read_pickle(path+dataset+"_train_"+dataset_name+".pkl")
    #X_train = df1.loc[:, df1.columns != 'y_train'].to_numpy()
    X_train = df1.loc[:, df1.columns[0:IMAGE_W*IMAGE_H*IMAGE_C]].to_numpy()
    X_train = X_train.reshape(-1,IMAGE_W,IMAGE_H,IMAGE_C)
    y_train = df1.loc[:, df1.columns == 'y_train'].to_numpy()
    # y_train = to_categorical(y_train)
    y_train = y_train.reshape(-1)

    df1 = pd.read_pickle(path+dataset+"_val_"+dataset_name+".pkl")
    #X_val = df1.loc[:, df1.columns != 'y_val'].to_numpy()
    X_val = df1.loc[:, df1.columns[0:IMAGE_W*IMAGE_H*IMAGE_C]].to_numpy()
    X_val = X_val.reshape(-1,IMAGE_W,IMAGE_H,IMAGE_C)
    y_val = df1.loc[:, df1.columns == 'y_val'].to_numpy()
    # y_val = to_categorical(y_val)
    y_val = y_val.reshape(-1)

    X_train = X_train[np.logical_or(y_train==4,y_train==5)]
    y_train = y_train[np.logical_or(y_train==4,y_train==5)]
    X_val = X_val[np.logical_or(y_val==4,y_val==5)]
    y_val = y_val[np.logical_or(y_val==4,y_val==5)]

    y_train[y_train==4] = 1
    y_train[y_train==5] = 0
    y_val[y_val==4] = 1
    y_val[y_val==5] = 0
    return (X_train, y_train), (X_val, y_val)

def make_fmnist_im():
  (X_train, y_train), (X_val, y_val) = fashion_mnist.load_data()
  X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
  X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)

  ma_idx = 0#random.randint(0, 9)
  mi_idx = 1#random.randint(0, 9)

  # To generate the indices of the data that we want. (Train)
  idx_train = np.concatenate(
      (
      np.where(y_train == ma_idx)[0],
      np.random.choice(np.where(y_train == mi_idx)[0], 150, replace=False)
      )
  )

  X_train = X_train[idx_train]
  y_train = y_train[idx_train]

  X_train, y_train = shuffle(X_train, y_train, random_state=42)

  # To generate the indices of the data that we want. (val)
  idx_val = np.concatenate(
      (
      np.where(y_val == ma_idx)[0],
      np.random.choice(np.where(y_val == mi_idx)[0], 25, replace=False)
      )
  )

  X_val = X_val[idx_val]
  y_val = y_val[idx_val]

  X_val, y_val = shuffle(X_val, y_val, random_state=42)
  return (X_train, y_train), (X_val, y_val)

def make_mnist_im():
  (X_train, y_train), (X_val, y_val) = mnist.load_data()
  X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
  X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2],1)

  ma_idx = 3#random.randint(0, 9)
  mi_idx = 8#random.randint(0, 9)

  # To generate the indices of the data that we want. (Train)
  idx_train = np.concatenate(
      (
      np.random.choice(np.where(y_train == ma_idx)[0], 5000, replace=False),
      np.random.choice(np.where(y_train == mi_idx)[0], 200, replace=False)
      )
  )

  X_train = X_train[idx_train]
  y_train = y_train[idx_train]

  X_train, y_train = shuffle(X_train, y_train, random_state=42)

  # To generate the indices of the data that we want. (val)
  idx_val = np.concatenate(
      (
      np.random.choice(np.where(y_val == ma_idx)[0], 1000, replace=False),
      np.random.choice(np.where(y_val == mi_idx)[0], 40, replace=False)
      )
  )

  X_val = X_val[idx_val]
  y_val = y_val[idx_val]

  X_val, y_val = shuffle(X_val, y_val, random_state=42)
  y_train[y_train==8] = 1
  y_train[y_train==3] = 0
  y_val[y_val==8] = 1
  y_val[y_val==3] = 0
  return (X_train, y_train), (X_val, y_val)

def make_cifar2():
  (X_train, y_train), (X_val, y_val) = cifar10.load_data()

  ma_idx = 0#random.randint(0, 9)
  mi_idx = 1#random.randint(0, 9)
  # print(ma_idx, mi_idx)

  # To generate the indices of the data that we want. (Train)
  idx_train = np.concatenate(
      (
      np.where(y_train == ma_idx)[0],
      np.where(y_train == mi_idx)[0]
      )
  )

  X_train = X_train[idx_train]
  y_train = y_train[idx_train]

  X_train, y_train = shuffle(X_train, y_train, random_state=42)

  # To generate the indices of the data that we want. (val)
  idx_val = np.concatenate(
      (
      np.where(y_val == ma_idx)[0],
      np.where(y_val == mi_idx)[0]
      )
  )

  X_val = X_val[idx_val]
  y_val = y_val[idx_val]

  X_val, y_val = shuffle(X_val, y_val, random_state=42)
  return (X_train, y_train), (X_val, y_val)

def make_cifar2_im():
  (X_train, y_train), (X_val, y_val) = cifar10.load_data()

  ma_idx = 0#random.randint(0, 9)
  mi_idx = 1#random.randint(0, 9)

  # To generate the indices of the data that we want. (Train)
  idx_train = np.concatenate(
      (
      np.where(y_train == ma_idx)[0],
      np.random.choice(np.where(y_train == mi_idx)[0], 250, replace=False)
      )
  )

  X_train = X_train[idx_train]
  y_train = y_train[idx_train]

  X_train, y_train = shuffle(X_train, y_train, random_state=42)
  y_train = y_train.reshape(-1)

  # To generate the indices of the data that we want. (val)
  idx_val = np.concatenate(
      (
      np.where(y_val == ma_idx)[0],
      np.random.choice(np.where(y_val == mi_idx)[0], 50, replace=False)
      )
  )

  X_val = X_val[idx_val]
  y_val = y_val[idx_val]

  X_val, y_val = shuffle(X_val, y_val, random_state=42)
  y_val = y_val.reshape(-1)
  return (X_train, y_train), (X_val, y_val)

def biased_get_class(X, y, c):
    
    xbeg = X[y == c]
    ybeg = y[y == c]
    
    return xbeg, ybeg
    #return xclass, yclass

def join_data(X_train,y_train,resx1,resy1):
  X_train = X_train.reshape(X_train.shape[0], -1)
  resx1 = resx1.reshape(resx1.shape[0],-1)
  X_train = np.vstack((resx1,X_train))
  y_train = np.hstack((resy1,y_train))
  # y_train = to_categorical(y_train)
  return X_train, y_train

# scale pixels
def prep_pixels(train, test, isResnet=True):
  # convert from integers to floats
  train_norm = prep_X(train, isResnet)
  test_norm = prep_X(test, isResnet)
  # return normalized images
  return train_norm, test_norm

# scale pixels
def prep_X(X, isResnet=True):
    # convert from integers to floats
    X = X.astype('float32')
    if isResnet:
        X = preprocess_input(X)
    else:
        X /= 255.0
    return X

def Euclidean_Metric(a,b):
      dis = np.linalg.norm(a - b)
      return dis

def create_pairs(x, digit_indices, num_classes):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
            
    return np.array(pairs), np.array(labels)


def create_pairs_on_set(X, labels, num_classes):
    
    digit_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    pairs, y = create_pairs(X, digit_indices, num_classes)
    y = y.astype('float32')
    
    return pairs, y


def show_image(image):
    plt.figure()
    plt.imshow(image)
    plt.colorbar()
    plt.grid(False)
    plt.show()

def initialize_base_network(IMAGE_H, IMAGE_W,IMAGE_C):
    input_shape = (IMAGE_H, IMAGE_W,IMAGE_C)
    input = Input(shape=input_shape, name="base_input")
    x = Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3))(input)
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.2)(x)
    x = Flatten(name="flatten_input")(x)
    x = Dense(128, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    x = Dense(128, activation='relu', name="third_base_dense")(x)

    return Model(inputs=input, outputs=x)

def initialize_base_network_dense(num_features):
    # Define the input layer
    inputs = Input(shape=(num_features,))

    # Define the hidden layers
    x = Dense(64, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    x = Dense(16, activation='relu')(x)

    return Model(inputs=inputs, outputs=x)

def initialize_base_network_resnet(IMAGE_H, IMAGE_W,IMAGE_C):
  target_shape = (IMAGE_W, IMAGE_H)
  base_cnn = ResNet50(
      weights="imagenet", input_shape=target_shape + (3,), include_top=False
  )

  flatten = Flatten()(base_cnn.output)
  # dense1 = Dense(512, activation="relu")(flatten)
  # dense1 = BatchNormalization()(dense1)
  # dense2 = Dense(256, activation="relu")(dense1)
  # dense2 = BatchNormalization()(dense2)
  output = Dense(256)(flatten)

  embedding = Model(base_cnn.input, output, name="Embedding")
  trainable = False
  for layer in base_cnn.layers:
      if layer.name == "conv5_block1_out":
          trainable = True
      layer.trainable = trainable
  return embedding

def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss_with_margin(margin):
    def contrastive_loss(y_true, y_pred):
        '''Contrastive loss from Hadsell-et-al.'06
        http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
        '''
        square_pred = K.square(y_pred)
        margin_square = K.square(K.maximum(margin - y_pred, 0))
        return (y_true * square_pred + (1 - y_true) * margin_square)
    return contrastive_loss

def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

def G_SM_smote(X, y,n_to_sample,cl, k = 5):
    nn = NearestNeighbors(n_neighbors=k, n_jobs=1)
    nn.fit(X)
    dist, ind = nn.kneighbors(X)
    # generating samples
    base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    neighbor_indices = np.random.choice(list(range(1, k)),n_to_sample)

    X_base = X[base_indices]
    X_neighbor = X[ind[base_indices, neighbor_indices]]
    
    samples = X_base + np.multiply(np.random.rand(n_to_sample,1), X_neighbor - X_base)
    return X_base, samples, [cl]*n_to_sample

def SMOTE_Data(X_train, y_train, one_hot = False):
  if one_hot:
    y_train = np.argmax(y_train, axis=1)
  
  #oversampling
  resx = []
  resy = []
  
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)# 0 is majority class, 1 is minority class
  i = 0 # 0 is minority class
  xclass, yclass = biased_get_class(X_train, y_train, i)
  xsamp, ysamp = G_SM_smote(xclass,yclass,n_to_sample,i)
  ysamp = np.array(ysamp)
  resx.append(xsamp)
  resy.append(ysamp)
  
  resx1 = np.vstack(resx)
  resy1 = np.hstack(resy)
  resx1 = resx1.reshape(resx1.shape[0],-1)
  X_train = np.vstack((resx1,X_train))
  y_train = np.hstack((resy1,y_train))
  
  
  return X_train, y_train

def G_SM_sia(All_X,samples_Y, n_to_sample, cl, maxdist_from_base=1):
    Minority_X=All_X[samples_Y == 1] # 1 is Minority class
    #Populate distance matrix
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i],All_X[j])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    #noise filtering
    pair_indices=[] # d = noise, minority class which its nearest neihbor is majority class
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        if(samples_Y[min_index[0]]==0): 
            pair = (i,min_index[0])
            pair_indices.append(pair)
    
    pair_indices = random.choices(pair_indices,k=n_to_sample)
    
    base_indices = []
    neighbor_indices = []

    for tup in pair_indices:
        base_indices.append(tup[0])
        neighbor_indices.append(tup[1])
    base_indices = np.array(base_indices)
    neighbor_indices = np.array(neighbor_indices)
    X_base = Minority_X[base_indices]
    X_neighbor = All_X[neighbor_indices]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1)*maxdist_from_base, X_neighbor - X_base)

    return X_base, samples, [cl]*n_to_sample

def G_SM_no_outlier(All_X,samples_Y, n_to_sample, cl, maxdist_from_base=1):
    Minority_X=All_X[samples_Y == 1] # 1 is Minority class
    #Populate distance matrix
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i],All_X[j])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    #noise filtering
    pair_indices=[] # d = noise, minority class which its nearest neihbor is majority class
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        if(samples_Y[min_index[0]]==0): 
            pair = (i,min_index[0])
            pair_indices.append(pair)
    
    pair_indices = random.choices(pair_indices,k=n_to_sample)
    
    base_indices = []
    neighbor_indices = []

    for tup in pair_indices:
        base_indices.append(tup[0])
        neighbor_indices.append(tup[1])
    base_indices = np.array(base_indices)
    neighbor_indices = np.array(neighbor_indices)
    X_base = Minority_X[base_indices]
    X_neighbor = All_X[neighbor_indices]

    samples = X_base + np.multiply(np.random.rand(n_to_sample,1)*maxdist_from_base, X_neighbor - X_base)

    return X_base, samples, [cl]*n_to_sample

def G_SM_sia_old(All_X,samples_Y, n_to_sample, cl, maxdist_from_base=1):
    g_index=0
    
    Minority_X=All_X[samples_Y == 1] # 1 is Minority class
    #Populate distance matrix
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i],All_X[j])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    #noise filtering
    base_indices=[] # d = noise, minority class which its nearest neihbor is majority class
    neighbor_indices=[]
    #print(Minority_X.shape[0])
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        #print(min_index)
        if(samples_Y[min_index[0]]==0): 
            base_indices.append(i)
            neighbor_indices.append(min_index[0])
    # Minority_X=np.delete(Minority_X,d,axis=0)
    # dis_matrix = np.array(dis_matrix)
    
    # base_indices = np.random.choice(list(range(len(X))),n_to_sample)
    base_indices = np.random.choice(base_indices,n_to_sample)
    neighbor_indices = np.random.choice(neighbor_indices,n_to_sample)

    X_base = Minority_X[base_indices]
    X_neighbor = All_X[neighbor_indices]
    samples = X_base + np.multiply(np.random.rand(n_to_sample,1)*maxdist_from_base, X_neighbor - X_base)

    #use 10 as label because 0 to 9 real classes and 1 fake/smoted = 10
    return X_base, samples, [cl]*n_to_sample

def sia_smote(X_train, y_train,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold, one_hot = False, maxdist_from_base=0.5, isResnet = True):
  if one_hot:
    y_train = np.argmax(y_train, axis=1)
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)

  xbase = xbase.reshape(xbase.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xsamp = xsamp.reshape(xsamp.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xbase_prep, xsamp_prep = prep_pixels(xbase, xsamp, isResnet)
  y_pred = sia_model.predict([xbase_prep, xsamp_prep])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp[pred]
  ysamp_sia = ysamp[pred]

  num_over = X_train[y_train == 0].shape[0]-xsamp_sia.shape[0]
  if num_over < X_train[y_train == 1].shape[0]:
    num_over = X_train[y_train == 1].shape[0]
  
  # SMOTE oversampling to balance dataset in the inner samples
  i = 1 # 1 is minority class
  xclass, yclass = biased_get_class(X_train, y_train, i)
  xbase, xsamp, ysamp = G_SM_smote(xclass,yclass,num_over,i)
  ysamp = np.array(ysamp)

  xbase = xbase.reshape(xbase.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xsamp = xsamp.reshape(xsamp.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xbase_prep, xsamp_prep = prep_pixels(xbase, xsamp, isResnet)
  y_pred = sia_model.predict([xbase_prep, xsamp_prep])
  pred = y_pred.ravel() < threshold
  xsamp_smote = xsamp[pred]
  ysamp_smote = ysamp[pred]

  X_train = X_train.reshape(X_train.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)
  X_train,y_train = join_data(X_train,y_train,xsamp_smote,ysamp_smote)
  return X_train,y_train

def sia_smote_balance(X_train, y_train,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold, one_hot = False, maxdist_from_base=0.5, isResnet = True, k = 5):
  if one_hot:
    y_train = np.argmax(y_train, axis=1)
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)

  xbase = xbase.reshape(xbase.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xsamp = xsamp.reshape(xsamp.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xbase_prep, xsamp_prep = prep_pixels(xbase, xsamp, isResnet)
  y_pred = sia_model.predict([xbase_prep, xsamp_prep])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp[pred]
  ysamp_sia = ysamp[pred]

  num_over = X_train[y_train == 0].shape[0]-xsamp_sia.shape[0]
  if num_over < X_train[y_train == 1].shape[0]:
    num_over = X_train[y_train == 1].shape[0]
  
  # SMOTE oversampling to balance dataset in the inner samples
  i = 1 # 1 is minority class
  xclass, yclass = biased_get_class(X_train, y_train, i)
  _, xsamp_smote, ysamp_smote = G_SM_smote(xclass,yclass,num_over,i, k)
  ysamp_smote = np.array(ysamp_smote)
  xsamp_smote = xsamp_smote.reshape(xsamp_smote.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  

  X_train = X_train.reshape(X_train.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)
  X_train,y_train = join_data(X_train,y_train,xsamp_smote,ysamp_smote)
  return X_train,y_train

def sia_smote_then_smote_sia(X_train, y_train,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold, one_hot = False, maxdist_from_base=0.5, isResnet = True, k = 5):
  if one_hot:
    y_train = np.argmax(y_train, axis=1)
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)

  xbase = xbase.reshape(xbase.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xsamp = xsamp.reshape(xsamp.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xbase_prep, xsamp_prep = prep_pixels(xbase, xsamp, isResnet)
  y_pred = sia_model.predict([xbase_prep, xsamp_prep])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp[pred]
  ysamp_sia = ysamp[pred]

  num_over = X_train[y_train == 0].shape[0]-xsamp_sia.shape[0]
  if num_over < X_train[y_train == 1].shape[0]:
    num_over = X_train[y_train == 1].shape[0]
  
  # SMOTE oversampling to balance dataset in the inner samples
  i = 1 # 1 is minority class
  xclass, yclass = biased_get_class(X_train, y_train, i)
  xbase, xsamp_smote, ysamp_smote = G_SM_smote(xclass,yclass,num_over,i, k)
  ysamp_smote = np.array(ysamp_smote)
  xsamp_smote = xsamp_smote.reshape(xsamp_smote.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xbase = xbase.reshape(xbase.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)

  xbase_prep, xsamp_prep = prep_pixels(xbase, xsamp_smote, isResnet)
  y_pred = sia_model.predict([xbase_prep, xsamp_prep])
  pred = y_pred.ravel() < threshold
  xsamp_smotesia = xsamp_smote[pred]
  ysamp_smotesia = ysamp_smote[pred]

  X_train = X_train.reshape(X_train.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)
  X_train,y_train = join_data(X_train,y_train,xsamp_smotesia,ysamp_smotesia)
  return X_train,y_train

def smote_then_sia(X_train, y_train,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold, one_hot = False, maxdist_from_base=0.5, isResnet = True, k = 5):
  if one_hot:
    y_train = np.argmax(y_train, axis=1)
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  
  # SMOTE oversampling to balance dataset in the inner samples
  i = 1 # 1 is minority class
  xclass, yclass = biased_get_class(X_train, y_train, i)
  xbase, xsamp_smote, ysamp_smote = G_SM_smote(xclass,yclass,n_to_sample,i, k)
  ysamp_smote = np.array(ysamp_smote)
  xsamp_smote = xsamp_smote.reshape(xsamp_smote.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xbase = xbase.reshape(xbase.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)

  xbase_prep, xsamp_prep = prep_pixels(xbase, xsamp_smote, isResnet)
  y_pred = sia_model.predict([xbase_prep, xsamp_prep])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp_smote[pred]
  ysamp_sia = ysamp_smote[pred]

  X_train = X_train.reshape(X_train.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)
  return X_train,y_train

def sia_asn_smote_dense(X_train, y_train, sia_model, threshold, maxdist_from_base=0.5, k = 5):
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)

  y_pred = sia_model.predict([xbase, xsamp])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp[pred]
  ysamp_sia = ysamp[pred]

  # num_over = X_train[y_train == 0].shape[0]-xsamp_sia.shape[0]
  # if num_over < X_train[y_train == 1].shape[0]:
  #   num_over = X_train[y_train == 1].shape[0]
  
  # asn smote
  kdata=pd.DataFrame(np.column_stack((X_train ,y_train)))
  g_sample=generate_x(kdata,100,k)
  xsamp_smote,ysamp_smote = g_sample[:,0:-1], g_sample[:,-1]
  
  print(f'shape of originil {X_train.shape[0]} sia smote: {xsamp_sia.shape[0]} asn smote: {xsamp_smote.shape[0]}')

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)
  X_train,y_train = join_data(X_train,y_train,xsamp_smote,ysamp_smote)
  return X_train,y_train

def sia_smote_balance_dense(X_train, y_train, sia_model, threshold, maxdist_from_base=0.5, k = 5):
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)

  y_pred = sia_model.predict([xbase, xsamp])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp[pred]
  ysamp_sia = ysamp[pred]

  num_over = X_train[y_train == 0].shape[0]-xsamp_sia.shape[0]
  if num_over < X_train[y_train == 1].shape[0]:
    num_over = X_train[y_train == 1].shape[0]
  
  # SMOTE oversampling to balance dataset in the inner samples
  i = 1 # 1 is minority class
  xclass, yclass = biased_get_class(X_train, y_train, i)
  _, xsamp_smote, ysamp_smote = G_SM_smote(xclass,yclass,num_over,i, k)
  ysamp_smote = np.array(ysamp_smote)

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)
  X_train,y_train = join_data(X_train,y_train,xsamp_smote,ysamp_smote)
  return X_train,y_train

def sia_smote_then_smote(X_train, y_train,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold, one_hot = False, maxdist_from_base=0.5, isResnet = True, k = 5):
  if one_hot:
    y_train = np.argmax(y_train, axis=1)
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)

  xbase = xbase.reshape(xbase.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xsamp = xsamp.reshape(xsamp.shape[0],IMAGE_W,IMAGE_H,IMAGE_C)
  xbase_prep, xsamp_prep = prep_pixels(xbase, xsamp, isResnet)
  y_pred = sia_model.predict([xbase_prep, xsamp_prep])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp[pred]
  ysamp_sia = ysamp[pred]  

  X_train = X_train.reshape(X_train.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)

  sm = SMOTE(k_neighbors=k)
  X_train,y_train = sm.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
  return X_train,y_train

def sia_smote_then_smote_dense(X_train, y_train, sia_model, threshold, maxdist_from_base=0.5, k = 5):
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)

  y_pred = sia_model.predict([xbase, xsamp])
  pred = y_pred.ravel() < threshold
  xsamp_sia = xsamp[pred]
  ysamp_sia = ysamp[pred]

  X_train,y_train = join_data(X_train,y_train,xsamp_sia,ysamp_sia)

  sm = SMOTE(k_neighbors=k)
  X_train,y_train = sm.fit_resample(X_train.reshape(X_train.shape[0], -1), y_train)
  return X_train,y_train

def SIAMESE_SMOTE_Data(X_train, y_train, one_hot = False, maxdist_from_base=1):
  X_train = X_train.reshape(X_train.shape[0], -1)
  if one_hot:
    y_train = np.argmax(y_train, axis=1)
  n_to_sample = np.count_nonzero(y_train == 0) - np.count_nonzero(y_train == 1)
  xbase, xsamp, ysamp = G_SM_sia(X_train,y_train,n_to_sample,1, maxdist_from_base)
  ysamp = np.array(ysamp)
  return xbase, xsamp, ysamp

def generate_x(samples,N,k):
    #n=int(N/10)
    time_start=time.time()
    g_index=0
    wrg=0
    samples_X=samples.iloc[:,0:-1]
    samples_Y=samples.iloc[:,-1]
    Minority_sample=samples[samples.iloc[:,-1].isin([1])] # 1 is Minority class
    Minority_sample_X=Minority_sample.iloc[:,0:-1]
    Minority_X=np.array(Minority_sample_X)
    All_X=np.array(samples_X)
    n1=All_X.shape[0]-2*Minority_X.shape[0]

    #Populate distance matrix
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i,:],All_X[j,:])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    
    #noise filtering
    d=[] # d = noise, minority class which its nearest neihbor is majority class
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        if(samples_Y[min_index[0]]==0): 
            d.append(i)
    Minority_X=np.delete(Minority_X,d,axis=0)

    n=int((n1)/Minority_X.shape[0])
    synthetic = np.zeros(((Minority_X.shape[0])*n,Minority_X.shape[1]),dtype=float)
    for i in range(Minority_X.shape[0]):

        # Filter in only neihgbours within safe radius
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(k, dis_matrix[i])))
        best_index={}
        best_f=0
        for h in range(len(min_index)):
            
            if(samples_Y[min_index[h]]==0): # 0 is Majority class
               best_index[best_f]=min_index[h]
               best_f+=1
               break # safe radius has been reached
            else:
                best_index[best_f]=min_index[h]
                best_f+=1

        # syntesize samples by interpolating base samples and safe neihgbours
        for j in range(0,n):
            nn=random.randint(0,len(best_index)-1)
            dif=All_X[best_index[nn]]-Minority_X[i]
            gap=random.random()
            synthetic[g_index]=Minority_X[i]+gap*dif
            g_index+=1
            
    # synthetic=synthetic[0:synthetic.shape[0]-,:]
    labels=np.ones(synthetic.shape[0])
    synthetic=np.insert(synthetic,synthetic.shape[1],values=labels,axis=1)
    examples=np.concatenate((samples,synthetic),axis=0)
    time_end=time.time()
    del(dis_matrix)
    return examples

def generate_x_sia2(samples,N,k, sia_model, threshold):
    #n=int(N/10)
    time_start=time.time()
    g_index=0
    wrg=0
    samples_X=samples.iloc[:,0:-1]
    samples_Y=samples.iloc[:,-1]
    Minority_sample=samples[samples.iloc[:,-1].isin([1])] # 1 is Minority class
    Minority_sample_X=Minority_sample.iloc[:,0:-1]
    Minority_X=np.array(Minority_sample_X)
    All_X=np.array(samples_X)
    n1=All_X.shape[0]-2*Minority_X.shape[0]

    #Populate distance matrix
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i,:],All_X[j,:])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    
    #noise filtering
    d=[] # d = noise, minority class which its nearest neihbor is majority class
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        if(samples_Y[min_index[0]]==0): 
            d.append(i)
    Minority_X=np.delete(Minority_X,d,axis=0)

    n=int((n1)/Minority_X.shape[0])
    base_samples = np.zeros(((Minority_X.shape[0])*n,Minority_X.shape[1]),dtype=float)
    synthetic = np.zeros(((Minority_X.shape[0])*n,Minority_X.shape[1]),dtype=float)
    for i in range(Minority_X.shape[0]):

        # Filter in only neihgbours within safe radius
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(k, dis_matrix[i])))
        best_index={}
        best_f=0
        for h in range(len(min_index)):
            
            if(samples_Y[min_index[h]]==0): # 0 is Majority class
               best_index[best_f]=min_index[h]
               best_f+=1
               break # safe radius has been reached
            else:
                best_index[best_f]=min_index[h]
                best_f+=1

        # syntesize samples by interpolating base samples and safe neihgbours
        for j in range(0,n):
            nn=random.randint(0,len(best_index)-1)
            dif=All_X[best_index[nn]]-Minority_X[i]
            gap=random.random()
            base_samples[g_index]=Minority_X[i]
            synthetic[g_index]=Minority_X[i]+gap*dif
            g_index+=1
    # new samples selection
    y_pred = sia_model.predict([base_samples, synthetic])
    pred = y_pred.ravel() < threshold
    synthetic = synthetic[pred]

    labels=np.ones(synthetic.shape[0])
    synthetic=np.insert(synthetic,synthetic.shape[1],values=labels,axis=1)
    examples=np.concatenate((samples,synthetic),axis=0)
    time_end=time.time()
    del(dis_matrix)
    return examples

def generate_x_sia(samples,N,k, sia_model, threshold):
    #n=int(N/10)
    time_start=time.time()
    g_index=0
    wrg=0
    samples_X=samples.iloc[:,0:-1]
    samples_Y=samples.iloc[:,-1]
    Minority_sample=samples[samples.iloc[:,-1].isin([1])] # 1 is Minority class
    Minority_sample_X=Minority_sample.iloc[:,0:-1]
    Minority_X=np.array(Minority_sample_X)
    All_X=np.array(samples_X)
    n1=All_X.shape[0]-2*Minority_X.shape[0]

    #Populate distance matrix
    dis_matrix=np.zeros((Minority_X.shape[0],All_X.shape[0]),dtype=float)
    for i in range(0,Minority_X.shape[0]):
        for j in range(0,All_X.shape[0]):
            dis_matrix[i,j]=Euclidean_Metric(Minority_X[i,:],All_X[j,:])
            if(dis_matrix[i,j]==0):
                dis_matrix[i,j]=999999
    dis_matrix=dis_matrix.tolist()
    
    #noise filtering
    d=[] # d = noise, minority class which its nearest neihbor is majority class
    for i in range(Minority_X.shape[0]):
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(1, dis_matrix[i])))
        if(samples_Y[min_index[0]]==0): 
            d.append(i)
    Minority_X=np.delete(Minority_X,d,axis=0)

    n=int((n1)/Minority_X.shape[0])
    base_samples = np.zeros(((Minority_X.shape[0])*n,Minority_X.shape[1]),dtype=float)
    synthetic = np.zeros(((Minority_X.shape[0])*n,Minority_X.shape[1]),dtype=float)
    for i in range(Minority_X.shape[0]):

        # Filter in only neihgbours within safe radius
        min_index=list(map(dis_matrix[i].index, heapq.nsmallest(k, dis_matrix[i])))
        best_index={}
        best_f=0
        for h in range(len(min_index)):
            best_index[best_f]=min_index[h]
            best_f+=1

        # syntesize samples by interpolating base samples and safe neihgbours
        for j in range(0,n):
            nn=random.randint(0,len(best_index)-1)
            dif=All_X[best_index[nn]]-Minority_X[i]
            gap=random.random()
            base_samples[g_index]=Minority_X[i]
            synthetic[g_index]=Minority_X[i]+gap*dif
            g_index+=1
    # new samples selection
    y_pred = sia_model.predict([base_samples, synthetic])
    pred = y_pred.ravel() < threshold
    synthetic = synthetic[pred]

    labels=np.ones(synthetic.shape[0])
    synthetic=np.insert(synthetic,synthetic.shape[1],values=labels,axis=1)
    examples=np.concatenate((samples,synthetic),axis=0)
    time_end=time.time()
    del(dis_matrix)
    return examples

def asn_smote(X_train, y_train, IMAGE_W, IMAGE_H, IMAGE_C, k):
    kdata=pd.DataFrame(np.column_stack((X_train,y_train)))
    g_sample=generate_x(kdata,100,k)
    X_train, y_train = g_sample[:,0:-1], g_sample[:,-1]
    return X_train, y_train
def RandomforClassifier(xtrain,ytrain,xtest,ytest):
    transfer = StandardScaler()
    xtrain = transfer.fit_transform(xtrain)
    xtest = transfer.transform(xtest)
    #选用随机森林模型
    rfc=RandomForestClassifier(
                                criterion='gini',
                                n_estimators=100,
                                min_samples_split=2,
                                min_samples_leaf=2,
                                max_depth=15,
                                random_state=6)
    #score_pre = cross_val_score(rfc,xtrain,ytrain,scoring='roc_auc',cv=10).mean()
    #scores = cross_val_score(rfc,xtrain,ytrain,cv=10,scoring='roc_auc')
    #print(scores)
    #print('mean CV-Scores: %.6f' % score_pre)
    rfc=rfc.fit(xtrain,ytrain)
    # #测试评估
    #result=rfc.score(xtest,ytest)
    AUC=roc_auc_score(ytest,rfc.predict_proba(xtest)[:,1])
    cm=confusion_matrix(ytest,rfc.predict(xtest))
    TN=cm[0][0]
    FP=cm[0][1]
    FN=cm[1][0]
    TP=cm[1][1]
    Acc=(TP+TN)/(TP+TN+FP+FN)
    Pos_Precision=TP/(TP+FP)
    #print("%.3f" %(Pos_Precision))
    #Neg_Precision=TN/(TN+FN)
    Sensitivity=TP/(TP+FN)
    Specificity=TN/(TN+FP)
    F_Measure=2*Sensitivity*Pos_Precision/(Sensitivity+Pos_Precision)
    G_Mean=np.sqrt(Sensitivity*Specificity)
    #print("F_Measure=%.6f" % F_Measure)
    #print("G_Mean=%.6f" %G_Mean)
    #print("AUC=%.6f" %AUC)
    #print("Acc=%.6f" % Acc)
    bal_acc = balanced_accuracy_score(ytest,rfc.predict(xtest))
    return F_Measure,G_Mean,AUC,Acc,bal_acc

def define_model(IMAGE_H, IMAGE_W,IMAGE_C):
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=(IMAGE_H, IMAGE_W, IMAGE_C)))
  model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Dropout(0.2))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Dropout(0.2))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
  model.add(MaxPooling2D((2, 2)))
  model.add(Dropout(0.2))
  model.add(Flatten())
  model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
  model.add(Dropout(0.2))
  model.add(Dense(1, activation='sigmoid'))
  # compile model
  opt = SGD(learning_rate=0.001, momentum=0.9)
  model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
  return model

def define_model_resnet(IMAGE_H, IMAGE_W,IMAGE_C):
  input_shape = (IMAGE_H, IMAGE_W, IMAGE_C)
  input_tensor = Input(shape=input_shape)
  x = ResNet50(input_shape=(input_shape), weights='imagenet', include_top=False)(input_tensor, training=False)
  x = GlobalAveragePooling2D()(x)
  # x = Flatten()(x)
  # x = Dropout(0.2)(x)
  # x = Dense(1024, activation='relu')(x)
  # x = Dropout(0.2)(x)
  # x = Dense(512, activation='relu')(x)
  predictions = Dense(1, activation='sigmoid')(x)
  # predictions = Dense(2, activation='softmax')(x)
  model = Model(inputs=input_tensor, outputs=predictions)
  opt = SGD(learning_rate=0.001, momentum=0.9)
  model.compile(optimizer = opt , loss = "binary_crossentropy", metrics=['accuracy'])
  return model

def train_siamese(X_train, y_train, X_test, y_test, isResnet, num_classes, IMAGE_H, IMAGE_W,IMAGE_C, epochs_siamese, batch_size):
    print('creating pairs')
    tr_pairs, tr_y = create_pairs_on_set(prep_X(X_train, isResnet), y_train, num_classes)
    ts_pairs, ts_y = create_pairs_on_set(prep_X(X_test, isResnet), y_test, num_classes)
    print(Counter(tr_y))
    print(Counter(ts_y))
    print('init base network')
    if isResnet:
        base_network = initialize_base_network_resnet(IMAGE_H, IMAGE_W,IMAGE_C)
    else:
        base_network = initialize_base_network(IMAGE_H, IMAGE_W,IMAGE_C)
    # create the left input and point to the base network
    input_shape = (IMAGE_H, IMAGE_W,IMAGE_C)
    input_a = Input(shape=input_shape, name="left_input")
    vect_output_a = base_network(input_a)
    # create the right input and point to the base network
    input_b = Input(shape=input_shape, name="right_input")
    vect_output_b = base_network(input_b)
    # measure the similarity of the two vector outputs
    output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])
    # specify the inputs and output of the model
    sia_model = Model([input_a, input_b], output)

    print('training siamese network...')
    rms = RMSprop()
    sia_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
    history = sia_model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=epochs_siamese, batch_size=batch_size, validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y), verbose=0)
    print('evaluate')
    loss = sia_model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)
    # loss = 0.0
    print('predict train')
    y_pred_train = sia_model.predict([tr_pairs[:,0], tr_pairs[:,1]])
    print('compute train acc')
    train_accuracy = compute_accuracy(tr_y, y_pred_train)
    print('predict val')
    y_pred_test = sia_model.predict([ts_pairs[:,0], ts_pairs[:,1]])
    print('compute val acc')
    test_accuracy = compute_accuracy(ts_y, y_pred_test)
    print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
    del(ts_pairs)
    del(tr_pairs)
    return sia_model

def train_siamese_dense(X_train, y_train, X_test, y_test, num_classes, num_features, epochs_siamese, batch_size):
    # print('creating pairs')
    tr_pairs, tr_y = create_pairs_on_set(X_train, y_train, num_classes)
    ts_pairs, ts_y = create_pairs_on_set(X_test, y_test, num_classes)
    # print(Counter(tr_y))
    # print(Counter(ts_y))
    # print('init base network')
    base_network = initialize_base_network_dense(num_features)
    # create the left input and point to the base network
    input_shape = (num_features)
    input_a = Input(shape=input_shape, name="left_input")
    vect_output_a = base_network(input_a)
    # create the right input and point to the base network
    input_b = Input(shape=input_shape, name="right_input")
    vect_output_b = base_network(input_b)
    # measure the similarity of the two vector outputs
    output = Lambda(euclidean_distance, name="output_layer", output_shape=eucl_dist_output_shape)([vect_output_a, vect_output_b])
    # specify the inputs and output of the model
    sia_model = Model([input_a, input_b], output)

    # print('training siamese network...')
    rms = RMSprop()
    sia_model.compile(loss=contrastive_loss_with_margin(margin=1), optimizer=rms)
    history = sia_model.fit([tr_pairs[:,0], tr_pairs[:,1]], tr_y, epochs=epochs_siamese, batch_size=batch_size, validation_data=([ts_pairs[:,0], ts_pairs[:,1]], ts_y), verbose=0)
    # print('evaluate')
    loss = sia_model.evaluate(x=[ts_pairs[:,0],ts_pairs[:,1]], y=ts_y)
    # loss = 0.0
    # print('predict train')
    y_pred_train = sia_model.predict([tr_pairs[:,0], tr_pairs[:,1]])
    # print('compute train acc')
    train_accuracy = compute_accuracy(tr_y, y_pred_train)
    # print('predict val')
    y_pred_test = sia_model.predict([ts_pairs[:,0], ts_pairs[:,1]])
    # print('compute val acc')
    test_accuracy = compute_accuracy(ts_y, y_pred_test)
    print("Loss = {}, Train Accuracy = {} Test Accuracy = {}".format(loss, train_accuracy, test_accuracy))
    del(ts_pairs)
    del(tr_pairs)
    return sia_model