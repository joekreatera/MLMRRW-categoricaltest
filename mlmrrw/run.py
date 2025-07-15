import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from  MLSSVRForestPredictor import MLSSVRForestPredictor
from DownloadHelper import *
from pandas.plotting import scatter_matrix
from random import random

from sklearn.datasets import fetch_openml
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, hamming_loss, classification_report, multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score

# for roc curve
from itertools import cycle
from sklearn.metrics import roc_curve, auc
from scipy import interp
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from math import isnan
import datetime

# from sklearn.metrics import roc_auc_score
from roc_auc_reimplementation import roc_auc as roc_auc_score


import sys
# https://stackoverflow.com/questions/54721497/scipy-sparse-indicator-matrix-from-arrays
# https://pandas.pydata.org/pandas-docs/stable/user_guide/basics.html


print(__doc__)

def custom_precision_score(y_true,y_pred):
    print('scoring')
    #np.savetxt( dataset_root + '/y_true.csv',y_true, delimiter=',')
    # print(y_pred)

    """
    s1 = average_precision_score(y_true, y_pred)
    sc = accuracy_score(y_true,y_pred)
    score_hamming = hamming_loss(y_true, y_pred)
    #print(f'Average precision score {sc}')
    print(f'Accuracy score {sc} \t\t {s1} \t\t {score_hamming}')

    if( isnan(s1)):
        # this was return sc
        return 10000
    return s1
    """
    y_true = y_true.to_numpy()
    r = roc_auc_score(y_true, y_pred, average='micro')
    s1 = average_precision_score(y_true, y_pred)
    # sc = accuracy_score(y_true,y_pred)
    print(f"AUC {r} avg prc:{s1} acc_sc{0} ")
    return r



def do_process(dataset_root,dataset_filename,datatest_size,unlabeled_size,experiment_name,test_train_dif, complete_ss = True ):
    """
        function to be called from outside.
        dataset_root: base folder directory
        dataset_filename: name of dataset file
        datatest_size: size of testing dataset in percentage
        unlabeled_size: size of the instances that will have labels removed, in percentage
        experiment_name: experiment name as str
        test_train_dif whether bootstrap has to divide the testing and training (false), or is already divided (true). dataset for this is named <dataset_filename>-{training|test}.csv

    """

    """
    Label columns are on label_columns list
    instance attributes are on instance_columns list
    """
    #print('Number of arguments:', len(sys.argv), 'arguments.')
    #print('Argument List:', str(sys.argv))
    print(f"Starting {dataset_root} {dataset_filename} : {experiment_name}")
    """
    dataset_root = sys.argv[1]
    dataset_filename = sys.argv[2]
    datatest_size = float(sys.argv[3])
    unlabeled_size = float(sys.argv[4])
    experiment_name =  sys.argv[5]
    test_train_dif = sys.argv[6]
    """

    if( test_train_dif == "true"):
        test_train_dif = True
    else:
        test_train_dif = False

    label_columns_file = open(dataset_root + '/' + 'label_columns.cols','r')
    label_columns_str = label_columns_file.read()
    label_columns_str = label_columns_str.rstrip()
    label_columns = label_columns_str.split(",")
    label_columns_size = len(label_columns) # assumes they are at the end of the end

    # print ( label_columns )

    originalTime = ( time.time() )
    dataset = None
    train_set  =None
    test_set = None

    if( not test_train_dif):
        dataset = getFromOpenML(dataset_filename,version=4,ospath=dataset_root+'/', download=False, save=True)
        train_set,test_set = train_test_split(dataset, test_size=datatest_size, random_state=43)
    else:
        train_set = getFromOpenML(dataset_filename+('-')+'train',version=4,ospath=dataset_root+'/', download=False, save=True)
        test_set = getFromOpenML(dataset_filename+('-')+'test',version=4,ospath=dataset_root+'/', download=False, save=True)

    labeled_instances, unlabeled_instances =  train_test_split(train_set, test_size=unlabeled_size, random_state=43) # simulate unlabeled instances
    # will be done internally by predictor,simulate semisupervised dataset
    #train_set.loc[unlabeled_instances.index, label_columns] = -1

    # get instance columns
    # get instance labels on Y
    instance_columns = train_set.columns.to_list()
    for col in label_columns:
        #print('col:  ' + col )
        instance_columns.remove(col)

    """
    print("---TRAINING SET----")
    print( train_set.info())
    print("---TEST SET ----")
    print( test_set.info())
    print("------")
    """


    compatibility_matrix_A = train_set.loc[labeled_instances.index, label_columns]
    compatibility_matrix_A_T = compatibility_matrix_A.transpose(copy=True)

    """

    # intersection of A*At (transpose). As the labels of an instance are a 0-1 Vector, the multiplication of them
    # becomes a intersection (multiply 0 or 1 per 0 or 1 is the same as AND)
    # doubt/question!!!  : if the unsupervised elements have a -1 on label columns:
    # 1) one labeled and one un labeled with a 1 on label, would set a -1 on label, but should be 0
    # 2) one labeled and one unlabeled with a 0 on label, would set a 0 on label, it is ok
    # 3) one unlabeled and one unlabeled would set a 1, its ok? they are actually the same, we dont know what they are
    #           on the other hand , we cannot say they are the same. Just 50% chance they will be the same in that label

    Option, take out all of the indices that are actually not labeled.
    Generate the process, add the rest of the table generating a second dot product with all ON -1

    Problem: when taking compatibility.... do not take -1?

    """

    intersection = compatibility_matrix_A.dot(compatibility_matrix_A_T)

    # compatibility_A_T is the tranpose of just label columns. Agg counts and sets a row with all the sums per instance of all labels
    union = compatibility_matrix_A_T.agg(['sum'])
    # transpose to generate a table of instances vs total number of 1's in labels
    union = union.transpose()
    #insert ones to operate after, a matrix multiplication
    union.insert(loc=0, column="ones", value= int(1))
    # get the values to numpy array to thrash indices
    union_transpose_np_matrix = union.values
    #generate new dataframe with reordered rows, to be able to calculate union before intersection
    union = pd.DataFrame({'sum':union['sum'] , 'ones':union['ones']   })
    #transpose to be able to multiply both matrices
    union = union.transpose()
    # get numpy 2d matrix
    union_np_matrix = union.values

    # indices to return the compatibility matrix to pandas dataframe
    colIndex = union.columns
    rowIndex = union.columns

    # magic moment obtaining union
    union_before_intersection_np_matrix = union_transpose_np_matrix.dot(union_np_matrix)
    # dataframe going to be final
    union_before_intersection = pd.DataFrame(data=union_before_intersection_np_matrix, index = rowIndex, columns = colIndex)
    # A+B-intersectionofAB
    union_minus_intersection = (union_before_intersection - intersection) + 0.00000001
    # probably add a very small epsilon to avoid 0 on union. Case in which an instance does not have any labels at all.
    # compatibility defined as the intersection/union
    compatibility_matrix = intersection/union_minus_intersection

    # check the:
    # https://towardsdatascience.com/17-types-of-similarity-and-dissimilarity-measures-used-in-data-science-3eb914d2681
    # where it says Sorensen-Dice. Could be another method:  2times the intersection dividied by (amount of p plus amount of q)

    # another:;
    # Recently, fractional norms have been brought into light for high-dimensiona data [5].
    # Charu C. Aggarwal, Alexander Hinneburg, and Daniel A. Keim. On the surprising
    # behavior of distance metrics in high dimensional space. Lecture Notes in Computer
    # Science, 1973:420–434, 2001.

    # others:
    # similarity ratio
    # percentage similarity / czekanowski coefficient
    # https://www.umass.edu/landeco/teaching/multivariate/readings/McCune.and.Grace.2002.chapter6.pdf
    #

    # another idea: instead of  getting the average similarity index, try to set a parameter to know know much of unlabeled data
    # should count.

    # another idea levenshtein distance applied to binary strings = > Hamming distance. What's the error between two strings of
    # labels. If we divide hamming distance by the amount on 1's in both, it becomes jaccard... i think.


    # add the missing unlabeled data matrix
    unlabeled_index = unlabeled_instances.index

    # add unlabeled columns
    for i in unlabeled_index :
        compatibility_matrix[i] = -1 #per column

    # add rows
    for i in unlabeled_index :
        compatibility_matrix.loc[i] = -1 # per row
    # ----

    #saveFile(compatibility_matrix,"compMatrix_v2_with_unlabeled.csv")

    finalTime = time.time()
    took = finalTime - originalTime
    print("Time : " + str(took) )

    # removes original 199-237 lines
    X = train_set[instance_columns]
    y = train_set[label_columns]



    #options  0.0,0.3,0.5,0.7
    # parameters = {'leaf_relative_instance_quantity':[0.02,0.05] ,'unlabeledIndex':[unlabeled_instances.index] , 'compatibilityMatrix':[compatibility_matrix] }
    # use when testing changes:  parameters = {'alpha':[0.7],'leaf_relative_instance_quantity':[0.05] ,'unlabeledIndex':[unlabeled_instances.index] , 'compatibilityMatrix':[compatibility_matrix] }

    # apparently complete_ss is making things worse on actual test dataset
    parameters = {'trees_quantity':[10,50,100],'division_op':['min','max'], 'complete_ss':[True] ,'alpha':[0.6,0.3,0.0,1],'leaf_relative_instance_quantity':[0.05] ,'unlabeledIndex':[unlabeled_instances.index] , 'compatibilityMatrix':[compatibility_matrix] }
    #parameters = {'trees_quantity':[10],'complete_ss':[True,False] ,'alpha':[1],'leaf_relative_instance_quantity':[0.02] ,'unlabeledIndex':[unlabeled_instances.index] , 'compatibilityMatrix':[compatibility_matrix] }
    parameters = {'trees_quantity':100,'division_op':'max', 'complete_ss':True ,'alpha':0,'leaf_relative_instance_quantity':0.05 ,'unlabeledIndex':unlabeled_instances.index , 'compatibilityMatrix':compatibility_matrix  }
    

    # the problem is CV=2 , it is too small. Question, when having train dataset, cv=1?
    # previous: scoring=make_scorer(custom_precision_score) , this should have never happened...
    do_gs = False 
    gs = None
    if( do_gs):
        predictor = MLSSVRForestPredictor(compatibilityMatrix = compatibility_matrix, unlabeledIndex=unlabeled_instances.index, bfr=dataset_root ,
        tag=experiment_name, complete_ss = complete_ss , trees_quantity = 2)
        
        gs = GridSearchCV( estimator = predictor, param_grid=parameters, scoring=make_scorer(custom_precision_score,needs_proba=True), cv=2, refit=False)
        #print(y_train.shape)
        gs.fit(train_set[instance_columns], train_set[label_columns])
        print('results!!')
        sorted(gs.cv_results_.keys())
        besties =gs.best_params_
        print("Best params " + str(besties["division_op"])  + " \t\t"  +  str(besties["alpha"]) + " \t\t" + str(besties["leaf_relative_instance_quantity"]) + " \t\t" + str(besties["complete_ss"]) + " \t\t" + str(besties["trees_quantity"]) )

        """
        Compatibility matrix now has :
        -1  on all rowsxcolumns that are not labeled
        X is the same as before
        Y has -1 in all the labels that are simulated unlabeled
        """
        predictor = MLSSVRForestPredictor(leaf_relative_instance_quantity = gs.best_params_['leaf_relative_instance_quantity'],
                                    compatibilityMatrix = compatibility_matrix,
                                    unlabeledIndex=unlabeled_instances.index,
                                    bfr=dataset_root ,
                                    tag=experiment_name,
                                    alpha = gs.best_params_['alpha'],
                                    complete_ss = gs.best_params_['complete_ss'],
                                    trees_quantity =  gs.best_params_['trees_quantity'],
                                    division_op = gs.best_params_['division_op']
                                    )
        predictor.fit(X,y)
        """
        y is modified in place!
        """
    else:
        predictor = MLSSVRForestPredictor(leaf_relative_instance_quantity = parameters['leaf_relative_instance_quantity'],
                                    compatibilityMatrix = compatibility_matrix,
                                    unlabeledIndex=unlabeled_instances.index,
                                    bfr=dataset_root ,
                                    tag=experiment_name,
                                    alpha = parameters['alpha'],
                                    complete_ss = parameters['complete_ss'],
                                    trees_quantity = parameters['trees_quantity'],
                                    division_op = parameters['division_op'],
                                    )
        predictor.fit(X,y)
        
    y_true = test_set[label_columns]
    instancesToTest = test_set[instance_columns]

    #remember: joe divided the train set to unlabel some of the instances.That's why previous simple efforts did not function to get the avg prediction
    #y_true = train_set.loc[labeled_instances.index, label_columns]
    #instancesToTest = train_set.loc[labeled_instances.index, instance_columns]

    # the other method is predict() that outputs 1 or 0 <<----- this might be needed for comparing one last time.
    predictions, probabilities = predictor.predict_with_proba(instancesToTest, print_prediction_log = True)

    # print(predictions)
    #prediction_arr = []
    #for i in predictions:
    #    arr = i[1].values.tolist()
    #    prediction_arr.append(arr[:])
        # test: https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_scoret

    prediction_arr = predictions
    probabilty_arr = probabilities

    y_arr = y_true.to_numpy()

    print(y_arr.shape)
    print(len(prediction_arr) , len(prediction_arr[0]) )


    results = open(dataset_root + '/' + experiment_name +".txt",'w')
    if( gs is not None):
        results.write(f'Hyperparams selected {gs.best_params_}\n')
    # score_avg_precision = average_precision_score(y_arr, prediction_arr)

    score_avg_precision = average_precision_score(y_arr, prediction_arr)
    results.write(f'Avg precision :\t {score_avg_precision}\n')


    score_avg_precision = label_ranking_average_precision_score(y_arr, probabilty_arr)
    results.write(f'Label rank Avg precision :\t {score_avg_precision}\n')

    score_accuracy = accuracy_score(y_arr, prediction_arr)
    results.write(f'Accuracy:\t{score_accuracy}\n')

    score_hamming = hamming_loss(y_arr, prediction_arr)
    results.write(f'Haming:\t{score_hamming}\n')

    score_auc = roc_auc_score(y_arr, probabilty_arr, average='micro')
    results.write(f'AUC:\t{score_auc}\n')

    report= classification_report(y_arr, prediction_arr)
    results.write(report)

    """
    TN	FP
    FN	TP
    """
    ml_confussion_matrix = multilabel_confusion_matrix(y_arr, prediction_arr)
    results.write(f'TN\tFN\tTP\tFP\n')
    for i in ml_confussion_matrix:
        results.write(f'{i[0,0]}\t{i[1,0]}\t{i[1,1]}\t{i[0,1]}\n')

    # results.close()

    # saveFile(y,"filled_out_all.csv", idx = True)



    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_test = y_arr
    y_score = probabilty_arr


    for i in range(0,label_columns_size):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    results.write(f'AUC_SK_BINARY:\t{roc_auc["micro"]}\n')
    results.close()

    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(0,label_columns_size)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(0,label_columns_size):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= label_columns_size

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    """
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(0,label_columns_size), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))
    """

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Some extension of Receiver operating characteristic to multi-class')
    plt.legend(loc="lower right")
    plt.savefig(dataset_root + '/' + experiment_name)


    finalTime = time.time()
    took = finalTime - originalTime
    print("Time : " + str(took) )



    """
    Was just a test
    # random predictor
    y_random = []

    def ran():
        if random() > 0.5:
            return 1
        return 0

    for index, row in instancesToTest.iterrows():
        inst = [ran(),ran(),ran(),ran(),ran(),ran(),ran(),ran(),ran(),ran(),ran(),ran(),ran(),ran()]
        y_random.append(inst)

    score_avg_precision = average_precision_score(y_arr, y_random)
    print(score_avg_precision)
    #score_accuracy  = accuracy_score(y_arr, prediction_arr)


    #print(score_accuracy)
    """



# for backwards compatibility
if __name__ == "__main__":
    dataset_root = sys.argv[1]
    dataset_filename = sys.argv[2]
    datatest_size = float(sys.argv[3])
    unlabeled_size = float(sys.argv[4])
    experiment_name =  sys.argv[5]
    test_train_dif = sys.argv[6]
    # this setting will complete semi supervised always

    do_process(dataset_root,
    dataset_filename,
    datatest_size,
    unlabeled_size,
    experiment_name,
    test_train_dif)
