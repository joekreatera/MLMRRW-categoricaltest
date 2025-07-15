import pandas as pd
import numpy as np
from  mlmrrw.MLMRRWPredictor import MLMRRWPredictor
from tqdm import tqdm
import warnings
from mlmrrw.DownloadHelper import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, classification_report, multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score
from random import random
from sklearn.model_selection import GridSearchCV
#from sklearn.metrics import roc_auc_score
#from mlmrrw.roc_auc_reimplementation import roc_auc as roc_auc_score
from sklearn.metrics import make_scorer
from sklearn.model_selection import ParameterGrid
from random import sample 
from mlmrrw.utils import multilabel_train_test_split,multilabel_kfold_split, generate_compatibility_matrix_counting_0s, generate_cosine_distance_based_compatibility,generate_compatibility_matrix, transform_multiclass_to_multilabel, get_features, save_report
# from mlmrrw.utils import get_stats

import sys
from sklearn.preprocessing import StandardScaler, QuantileTransformer, Normalizer

warnings.filterwarnings("ignore")


if __name__ == '__main__':
    ds_name = sys.argv[1]

    ds_configs = {
    "emotions":6, # multilabel
    "yeast_multi":14, # multilabel
    "birds":19,
    'flags':7,
    'cal500':174,
    'slashdot':21,
    'enron':53,
    'water':14,
    'medical':45,
    'foodtruck':12,
    'plantgo':12,
    'eukaryote':22,
    'genbase':27,
    'gonegative': 8,
    'humanpse': 14,
    'plantpse':12,
    '3sourcesbbc':6,
    'chd49':6,
    'mediamill':101,
    'scene':6,
    'image':5,
    'flagscat':7,
    'fifa':26,
    'amphibians':7,
    'divorce':10,
    'nursery':3,
    }
    # getFromOpenML will convert automatically the classes found to a mutually exclusive multilabel
    dataset = getFromOpenML(ds_name,version="active",ospath='datasets/', download=False, save=False)
    #dataset = getFromOpenML(ds_name+'-train',version="active",ospath='datasets/', download=False, save=False)
    # test_dataset = getFromOpenML(ds_name+'-test',version="active",ospath='datasets/', download=False, save=False)
    

    # multilabel_dataset = transform_multiclass_to_multilabel(dataset, "label_0") # will expand label_0 to the unique values , mutually exclusive as labels
    label_columns = [f"label_{i}" for i in range(0,ds_configs[ds_name])]  # for iris (3) ,for yeast(10) for ecoli(8), satimage(6)
    


    # identify categorical columns and dummy encode them. 
    dataset = pd.get_dummies(dataset)*1
    print(dataset.shape)
    # exit()

    gamma_A_collection = [0.001, 0.01, 0.1 , 1, 5] # 5,10,50,100
    gamma_I_collection = [0.1,0.5,1,5]# 10,20,50,100
    XI_v1_collection = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9] # 0.1,0.2,0.3,0.4,0.5,
    XI_v2_collection = list(reversed(XI_v1_collection))
    # gamma_collection = [1,5,10,15,20,30,50,70,100] # altamente dependiente del dataset ooooo , es necesario escalarlo todo
    gamma_collection = [0.5,1,5] # misssin all but 50 , # ,10,20
    eps_collection  = [0.2,0.4,0.6,0.8] # groups


    parameters = {
    'gamma_A':gamma_A_collection ,
    'gamma_I':gamma_I_collection ,
    'XI_v1': XI_v1_collection, 
    # 'XI_v2': XI_v2_collection, as it is 1 - XI_v1
    'gamma':gamma_collection,
    'eps': eps_collection,
    'scaler':[Normalizer(), None]
    }
    param_grid = ParameterGrid(parameters)

    final_list = []
    best_lsap_avg = 0
    i = 0
    best_param_combination = None
    unlabeled_ratio = .1    
    
    train_set,test_set = multilabel_train_test_split(dataset, test_size=.30, random_state=180, stratify=dataset[label_columns]) # .05 for the CLUS test as it was with train and test datasets
    train_set.reset_index(drop=True, inplace=True)
    test_set.reset_index(drop=True , inplace=True)

    instance_columns = get_features(train_set, label_columns)


    param_grid = sample( list(param_grid) , 150 )
    for param_combination in  tqdm(list(param_grid)):
        i += 1
        k_fold=10
        
        # print(param_combination)
        folds = multilabel_kfold_split(n_splits=k_fold, shuffle=True, random_state= 180)
        lsap = 0
        fold = 0

        auprc_curve = 0
        label_rank_average_precision= 0
        average_precision= 0
        auc_micro= 0
        auc_macro= 0
        hamming_loss = 0
        accuracy= 0 
            

        for train_index, test_index in folds.split(train_set[instance_columns], train_set[label_columns] ):
        # for rep in range(0,1):

            k_train_set = train_set.loc[train_index]
            k_test_set = train_set.loc[test_index] 

            #train_set,test_set = multilabel_train_test_split(dataset, test_size=.30, random_state=rep, stratify=dataset[label_columns])
            
            labeled_instances, unlabeled_instances =  multilabel_train_test_split(k_train_set, test_size=unlabeled_ratio, random_state=141, stratify=k_train_set[label_columns] ) # simulate unlabeled instances
            
            X = k_train_set[instance_columns]
            y = k_train_set[label_columns]

            param_combination["XI_v2"] = 1-param_combination["XI_v1"]
            predictor = MLMRRWPredictor(
                                unlabeledIndex=unlabeled_instances.index,
                                tag=ds_name,
                                hyper_params_dict = param_combination,
                                )
            
        
            scaler = None
            if(param_combination["scaler"] is not None):
                scaler = param_combination["scaler"]
                X = pd.DataFrame(data=scaler.fit_transform(X), index=X.index, columns=X.columns)
            predictor.fit(X,y)

            y_true = k_test_set[label_columns].to_numpy()
            x_test = k_test_set[instance_columns]

            if(scaler is not None):
                x_test = pd.DataFrame(data=scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
            predictions, probabilities = predictor.predict_with_proba(x_test, y_true = y_true)    
            nm = f"test1_{fold}_{ds_name}_{i}" 
            # % (i, param_combination["gamma"], param_combination["M_groups"] ,  param_combination["N_attr"] ,param_combination["leaf_relative_instance_quantity"],param_combination["trees_quantity"])
            output = save_report('./output/', nm, y_true = y_true, y_predicted = predictions, y_predicted_proba = probabilities, do_output = True, parameters = param_combination)
            
            lsap += (output["label_score_avg_precision"])
            auprc_curve += output["micro_avg_precision_AUPRC"]
            label_rank_average_precision+= output["label_score_avg_precision"]
            average_precision+= output["score_avg_precision"]
            auc_micro+= output["score_auc_micro"]
            auc_macro+= output["score_auc_macro"]
            hamming_loss+= output["score_hamming"]
            accuracy+= output["score_accuracy"] 

            final_list.append(output)
            fold += 1
        
        lsap_avg = lsap/k_fold
        
        if lsap_avg > best_lsap_avg:
            best_lsap_avg = lsap_avg
            best_param_combination = param_combination


    # grid_search = GridSearchCV(n_jobs=2,estimator = predictor, param_grid=parameters, scoring=make_scorer(custom_precision_score,needs_proba=True,estimator=predictor), cv=2, refit=False)
    # grid_search.fit(X,y)

    # use itertools to calculate all!!! Paremeter grid is for this also!

    # we should do this 5 times

    for it in range(0,5):
        labeled_instances, unlabeled_instances =  multilabel_train_test_split(train_set, test_size=unlabeled_ratio, random_state=141, stratify=train_set[label_columns] ) # simulate unlabeled instances
        
        X = train_set[instance_columns]
        y = train_set[label_columns]
        
        predictor = MLMRRWPredictor(
                                    unlabeledIndex=unlabeled_instances.index,
                                    tag=ds_name,
                                    hyper_params_dict = best_param_combination,
                                    )
        scaler = None
        if(best_param_combination["scaler"] is not None):
            scaler = best_param_combination["scaler"]
            X = pd.DataFrame(data=scaler.fit_transform(X), index=X.index, columns=X.columns)
        predictor.fit(X,y)
        y_true = test_set[label_columns].to_numpy()
        x_test = test_set[instance_columns]

        if(scaler is not None):
                    x_test = pd.DataFrame(data=scaler.transform(x_test), index=x_test.index, columns=x_test.columns)
        predictions, probabilities = predictor.predict_with_proba(x_test, y_true = y_true)    
        nm = f"final_{it}_{ds_name}_{i}" 
        # % (i, param_combination["gamma"], param_combination["M_groups"] ,  param_combination["N_attr"] ,param_combination["leaf_relative_instance_quantity"],param_combination["trees_quantity"])
        output = save_report('./', nm, y_true = y_true, y_predicted = predictions, y_predicted_proba = probabilities, do_output = True, parameters = param_combination)
        final_list.append(output)       

    df = pd.DataFrame(data=final_list)
    df.to_csv(f'{ds_name}_{unlabeled_ratio}_all_params1_s.csv')