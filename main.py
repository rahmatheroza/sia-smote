import sys
import myfunction as fu
from typing_extensions import Counter
import pandas as pd
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Dense, Dropout, Lambda, Conv2D, GlobalAveragePooling2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import StratifiedKFold
from imblearn.over_sampling import SMOTE, RandomOverSampler

exp_name="siamese"
dataset_name="128px_raw"
data_path = "/home/rh22708/data/"
num_classes = 2
epochs_siamese = 100
epochs_classifier = 20
batch_size = 32 # CHANGEDTO CHESTXRAY
num_folds = 5
num_repeat=3
# siamese hyperparameter
threshold = [0.02, 0.05, 0.1, 0.3, 0.5]
# other method hyerparameter
ks = [3,5,7,9]

no_exp = int(sys.argv[1])
dataset = sys.argv[2]
if dataset == 'fmnist' or dataset == 'mnist':
  isResnet = False
else:
  isResnet = True
if len(sys.argv) == 4:
  n_neigh = int(sys.argv[3])
  if n_neigh == 0:
    n_neigh = 5
else:
  n_neigh = 5

if len(sys.argv) == 5:
  max_dist = float(sys.argv[4])
  if max_dist == 0:
    max_dist = 0.5
else:
  max_dist = 0.5
print('argv: ',no_exp, dataset, n_neigh, max_dist)
# coba smote ke after oversample by sia smote
# coba sia smote dengan interpolate semua minority ke semua majority

# exp
rand_ov_exp = 0
smote_exp = (28,31)
asn_smote_exp = (32,35)
sia_smote_exp = (36,40)
sia_smote_2_exp = (41,45)
smote_then_sia_exp = (46,50)
siasmote_then_smotesia_exp = (51,55)

IMAGE_H, IMAGE_W,IMAGE_C = fu.get_config(dataset)
#LOAD DATASET
print('loading dataset')
(X_train, y_train), (X_test, y_test) = fu.load_dataset(data_path, dataset, dataset_name, IMAGE_H, IMAGE_W,IMAGE_C)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print('Counter train data: ', Counter(y_train))
print('Counter val data: ', Counter(y_test))

print('Start exp')
X_test = fu.prep_X(X_test, isResnet) #BEWARE
is_finished = True
while True:
  print(f'start exp no: {no_exp}')
  val_score, test_score = list(), list()
  kfold = StratifiedKFold(num_folds, shuffle=True)
  i_fold = 0
  for train_ix, test_ix in kfold.split(np.zeros(y_train.shape[0]),y_train):
    print('i_fold: ',i_fold)
    X_trainsplit, y_trainsplit, X_val_split, y_val_split = X_train[train_ix], y_train[train_ix], X_train[test_ix], y_train[test_ix]
    if no_exp == sia_smote_exp[0] or no_exp == sia_smote_2_exp[0] or no_exp == smote_then_sia_exp[0] or no_exp == siasmote_then_smotesia_exp[0]:
      sia_model = fu.train_siamese(X_trainsplit, y_trainsplit, X_val_split, y_val_split, isResnet, num_classes, IMAGE_H, IMAGE_W,IMAGE_C, epochs_siamese, batch_size)
    for iteration in range(num_repeat):  
      if no_exp == rand_ov_exp:
        # define oversampling strategy
        oversample = RandomOverSampler(sampling_strategy='minority')
        # fit and apply the transform
        X_trainsplit_ov, y_trainsplit_ov = oversample.fit_resample(X_trainsplit.reshape(X_trainsplit.shape[0], -1), y_trainsplit)
        X_trainsplit_ov = X_trainsplit_ov.reshape(X_trainsplit_ov.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)
      # elif no_exp == 1:
      elif no_exp >= smote_exp[0] and no_exp <= smote_exp[1]:
        k = ks[no_exp-smote_exp[0]]
        print("SMOTE with k:",k)
        sm = SMOTE(k_neighbors=k)
        X_trainsplit_ov, y_trainsplit_ov = sm.fit_resample(X_trainsplit.reshape(X_trainsplit.shape[0], -1), y_trainsplit)
        X_trainsplit_ov = X_trainsplit_ov.reshape(X_trainsplit_ov.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

        if i_fold == num_folds-1 and iteration == num_repeat-1:
          if no_exp == smote_exp[1]:
            is_finished = True
          else :
            is_finished = False
            no_exp +=1
      # elif no_exp == 2:
      elif no_exp >= asn_smote_exp[0] and no_exp <= asn_smote_exp[1]:
        # 4 exp for 4 different k 3,5,7,9
        k = ks[no_exp-asn_smote_exp[0]]
        print("ASN smote with k:",k)
        X_trainsplit_ov, y_trainsplit_ov = fu.asn_smote(X_trainsplit.reshape(X_trainsplit.shape[0], -1), y_trainsplit, IMAGE_W, IMAGE_H, IMAGE_C, k=k)
        X_trainsplit_ov = X_trainsplit_ov.reshape(X_trainsplit_ov.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

        if i_fold == num_folds-1 and iteration == num_repeat-1:
          if no_exp == asn_smote_exp[1]:
            is_finished = True
          else :
            is_finished = False
            no_exp +=1
      elif no_exp >= sia_smote_exp[0] and no_exp <= sia_smote_exp[1]:
        #SIA-SMOTE
        print('SIA SMOTE balance dist 0.5 with threshold:',threshold[no_exp-sia_smote_exp[0]])
        X_trainsplit_ov, y_trainsplit_ov = fu.sia_smote_balance(X_trainsplit.reshape(X_trainsplit.shape[0], -1), y_trainsplit,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold[no_exp-sia_smote_exp[0]], one_hot = False, maxdist_from_base = max_dist, isResnet = isResnet, k=n_neigh)
        X_trainsplit_ov = X_trainsplit_ov.reshape(X_trainsplit_ov.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

        if i_fold == num_folds-1 and iteration == num_repeat-1:
          if no_exp == sia_smote_exp[1]:
            is_finished = True
          else :
            is_finished = False
            no_exp +=1
      elif no_exp >= sia_smote_2_exp[0] and no_exp <= sia_smote_2_exp[1]:
        #SIA-SMOTE
        print('SIA SMOTE then SMOTE dist 0.5 with threshold:',threshold[no_exp-sia_smote_2_exp[0]])
        X_trainsplit_ov, y_trainsplit_ov = fu.sia_smote_then_smote(X_trainsplit.reshape(X_trainsplit.shape[0], -1), y_trainsplit,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold[no_exp-sia_smote_2_exp[0]], one_hot = False, maxdist_from_base = max_dist, isResnet = isResnet, k=n_neigh)
        X_trainsplit_ov = X_trainsplit_ov.reshape(X_trainsplit_ov.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

        if i_fold == num_folds-1 and iteration == num_repeat-1:
          if no_exp == sia_smote_2_exp[1]:
            is_finished = True
          else :
            is_finished = False
            no_exp +=1
      elif no_exp >= smote_then_sia_exp[0] and no_exp <= smote_then_sia_exp[1]:
        #SIA-SMOTE
        print('SMOTE then SIA dist 0.5 with threshold:',threshold[no_exp-smote_then_sia_exp[0]])
        X_trainsplit_ov, y_trainsplit_ov = fu.smote_then_sia(X_trainsplit.reshape(X_trainsplit.shape[0], -1), y_trainsplit,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold[no_exp-smote_then_sia_exp[0]], one_hot = False, maxdist_from_base = max_dist, isResnet = isResnet, k=n_neigh)
        X_trainsplit_ov = X_trainsplit_ov.reshape(X_trainsplit_ov.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

        if i_fold == num_folds-1 and iteration == num_repeat-1:
          if no_exp == smote_then_sia_exp[1]:
            is_finished = True
          else :
            is_finished = False
            no_exp +=1
      elif no_exp >= siasmote_then_smotesia_exp[0] and no_exp <= siasmote_then_smotesia_exp[1]:
        #SIA-SMOTE
        print('SIASMOTE 4 with threshold:',threshold[no_exp-siasmote_then_smotesia_exp[0]])
        X_trainsplit_ov, y_trainsplit_ov = fu.sia_smote_then_smote_sia(X_trainsplit.reshape(X_trainsplit.shape[0], -1), y_trainsplit,IMAGE_W,IMAGE_H,IMAGE_C, sia_model, threshold[no_exp-siasmote_then_smotesia_exp[0]], one_hot = False, maxdist_from_base = max_dist, isResnet = isResnet, k=n_neigh)
        X_trainsplit_ov = X_trainsplit_ov.reshape(X_trainsplit_ov.shape[0], IMAGE_W, IMAGE_H, IMAGE_C)

        if i_fold == num_folds-1 and iteration == num_repeat-1:
          if no_exp == siasmote_then_smotesia_exp[1]:
            is_finished = True
          else :
            is_finished = False
            no_exp +=1
      else:
        pass
      
      X_trainsplit_ov, X_val_split_prep = fu.prep_pixels(X_trainsplit_ov, X_val_split, isResnet)

      if isResnet:
        model = fu.define_model_resnet(IMAGE_H, IMAGE_W,IMAGE_C)
      else:
        model = fu.define_model(IMAGE_H, IMAGE_W,IMAGE_C)
      # fit model
      print(f'model fitting fold: {i_fold}, iteration: {iteration}')
      history = model.fit(X_trainsplit_ov, y_trainsplit_ov, epochs=epochs_classifier, batch_size=batch_size, validation_data=(X_val_split_prep, y_val_split), verbose=0)
      y_val_pred = model.predict(X_val_split_prep)
      val_score.append(fu.calculate_score(y_val_split, y_val_pred))
      y_test_pred = model.predict(X_test)
      test_score.append(fu.calculate_score(y_test, y_test_pred))    
    # end of iteration
    i_fold=i_fold+1
  # end of cv
  print('finish exp no:',no_exp)
  print('printing val score')
  fu.print_score(val_score)
  print('printing test score')
  fu.print_score(test_score)
  if is_finished:
    break
print('Done')
  
