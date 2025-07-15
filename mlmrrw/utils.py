import pandas as pd
import uuid
from .roc_auc_reimplementation import roc_auc as roc_auc_score
from sklearn.metrics import average_precision_score, accuracy_score, hamming_loss, classification_report, multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import precision_recall_curve
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
from sklearn.model_selection import train_test_split
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sklearn.utils import indexable, _safe_indexing
from sklearn.utils.validation import _num_samples
from sklearn.model_selection._split import _validate_shuffle_split
from itertools import chain

def multilabel_kfold_split(n_splits=5, shuffle=True, random_state= 180):
    return MultilabelStratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

def multilabel_train_test_split(*arrays,
                                test_size=None,
                                train_size=None,
                                random_state=None,
                                shuffle=True,
                                stratify=None):
    """
    Train test split for multilabel classification. Uses the algorithm from: 
    'Sechidis K., Tsoumakas G., Vlahavas I. (2011) On the Stratification of Multi-Label Data'.
    """
    if stratify is None:
        return train_test_split(*arrays, test_size=test_size,train_size=train_size,
                                random_state=random_state, stratify=None, shuffle=shuffle)
    
    assert shuffle, "Stratified train/test split is not implemented for shuffle=False"
    
    n_arrays = len(arrays)
    arrays = indexable(*arrays)
    n_samples = _num_samples(arrays[0])
    n_train, n_test = _validate_shuffle_split(
        n_samples, test_size, train_size, default_test_size=0.25
    )
    cv = MultilabelStratifiedShuffleSplit(test_size=n_test, train_size=n_train, random_state=123)
    train, test = next(cv.split(X=arrays[0], y=stratify))

    return list(
        chain.from_iterable(
            (_safe_indexing(a, train), _safe_indexing(a, test)) for a in arrays
        )
    )

def save_report(root, experiment_name, y_true = None, y_predicted = None, y_predicted_proba = None, do_output = False, parameters = None):
    results = open(root + '/' + experiment_name + f"_{uuid.uuid4()}.txt",'w')
    # results.write(f'Hyperparams selected {gs.best_params_}\n')
    # score_avg_precision = average_precision_score(y_arr, prediction_arr)

    # print("*******   RESULTS " , y_true.shape )
    # print("*******   RESULTS 2 " , y_predicted.shape )
    if(y_predicted_proba is not None):
        score_avg_precision = average_precision_score(y_true, y_predicted_proba)
        results.write(f'Avg precision :\t {score_avg_precision}\n')
        
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(y_predicted_proba.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_predicted_proba[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], y_predicted_proba[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_predicted_proba.ravel()
        )
        average_precision["micro"] = average_precision_score(y_true, y_predicted_proba, average="micro")
        avg_pr_micro = average_precision["micro"]
        results.write(f'Micro Avg precision AUCPR :\t {avg_pr_micro}\n')
        """
        display = PrecisionRecallDisplay(
            recall=recall["micro"],
            precision=precision["micro"],
            average_precision=average_precision["micro"],
        )
        
        display.plot()
        _ = display.ax_.set_title("Micro-averaged over all classes")
        plt.show()
        """

    if(y_predicted_proba is not None):
        label_score_avg_precision = label_ranking_average_precision_score(y_true, y_predicted_proba)
        results.write(f'Label rank Avg precision :\t {label_score_avg_precision}\n')

    if(y_predicted is not None):
        score_accuracy = accuracy_score(y_true, y_predicted)
        results.write(f'Accuracy:\t{score_accuracy}\n')

    if(y_predicted is not None):
        score_hamming = hamming_loss(y_true, y_predicted)
        results.write(f'Haming:\t{score_hamming}\n')

    if(y_predicted_proba is not None):
        score_auc_micro = roc_auc_score(y_true, y_predicted_proba, average='micro')
        results.write(f'AUC.micro:\t{score_auc_micro}\n')

    if(y_predicted_proba is not None):    
        score_auc_macro = roc_auc_score(y_true, y_predicted_proba, average='macro')
        results.write(f'AUC.macro:\t{score_auc_macro}\n')

    if(y_predicted is not None):
        report = classification_report(y_true, y_predicted)
        # print(report)
        results.write(report)

    if(y_predicted is not None):    
        conf = multilabel_confusion_matrix(y_true, y_predicted)

    if(y_predicted is not None):    
        results.write(f'TN\tFN\tTP\tFP\n')
        for i in conf:
            results.write(f'{i[0,0]}\t{i[1,0]}\t{i[1,1]}\t{i[0,1]}\n')
    
    
    if( parameters is not None):
        for i in parameters:
            results.write(f'{i}: {parameters[i]}\n')
        
        
    results.close()
    if(do_output):
        return {
        'score_avg_precision':score_avg_precision,
        'micro_avg_precision_AUPRC':avg_pr_micro,
        'label_score_avg_precision':label_score_avg_precision,
        'score_accuracy':score_accuracy,
        'score_hamming':score_hamming,
        'score_auc_micro':score_auc_micro,
        'score_auc_macro':score_auc_macro,
        'report':report,
        'parameters':parameters
        }

def get_features(train_set, labels):
    instance_columns = train_set.columns.to_list()
    for col in labels:
        #print('col:  ' + col )
        instance_columns.remove(col)
    return instance_columns

def transform_multiclass_to_multilabel(o_dataframe, class_column):
    
    unique_vals = o_dataframe[ class_column ].unique()
    dataframe = o_dataframe.drop(columns=[ class_column ])
    counter = 0
    for i in unique_vals:
        dataframe[f"label_{counter}"] = 0
        dataframe.loc[ o_dataframe[class_column] == i , f"label_{counter}" ] = 1
        counter+=1
    return dataframe


def generate_multiclass_compatibility_matrix(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set
    
    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")
    
    compatibility_matrix_A = train_set.loc[labeled_instances.index, label_columns]
    compatibility_matrix_A_T = compatibility_matrix_A.transpose()

    intersection = compatibility_matrix_A.dot(compatibility_matrix_A_T)
    
    return intersection


def generate_feature_weights(feature_set):
    """
    will output feature weights according to algorithm on 
    A Consolidated Decision Tree-Based Intrusion Detection System for Binary and Multiclass Imbalanced Datasets
    and 
    Infinite Feature Selection:A Graph-based Feature Filtering Approach
    """
    feature_weights = np.zeros(feature_set.shape[1])
    # print(feature_weights.shape)
    alpha =0.5
    normalized_data = (feature_set - feature_set.min())/( feature_set.max() -  feature_set.min())
    std_devs = normalized_data.std(ddof=0)
    feature_counter= 0

    
    A = np.zeros(shape=(feature_weights.shape[0], feature_weights.shape[0]))
    I = np.eye(feature_weights.shape[0])
    for feature_i, feature_column_i in feature_set.items():
        fi_vs_fj = np.zeros(feature_weights.shape[0])
        counter = 0
        
        for feature_j, feature_column_j in feature_set.items():
            
            std_i_j = max( std_devs[feature_i], std_devs[feature_j] )
            
            # print(f'{feature_i} {feature_j}')
            corr_i_j = 1 - np.abs( feature_column_i.corr(feature_column_j, method='spearman')  ) 
            # print(f'comparing {feature_column_i} {feature_column_j}')
            
            fi_vs_fj[counter] = alpha*std_i_j + (1-alpha)*corr_i_j # is not counter but the actual correlation score
            counter += 1
        # yeah i know that this is going to calculate the same thing twice
        # feature_weights[feature_counter] =  fi_vs_fj.mean()
        A[feature_counter, :] = fi_vs_fj
        # check eigen values
        
        feature_counter += 1
    
    # print(A)
    rho_A = np.max( np.linalg.eig(A)[0] )
    # print(rho_A)
    inv_minus_I = np.linalg.inv(I - 0.9/rho_A*A ) - I
    S = np.ones(feature_weights.shape[0])*inv_minus_I
    # print(S) #lets use this one to select the N best features on each tree. 
    # select one at random and then all the most important and significant ones. 
    return S
    """
    print("-------------")
    feature_weights = S.mean(axis = 0)
    print(feature_weights)
    print("**********")
    print( np.argsort(feature_weights) )
    """
    
    
def generate_compatibility_matrix_counting_0s(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set

    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")

    compatibility_matrix_A = train_set.loc[labeled_instances.index, label_columns]
    compatibility_matrix_A_T = compatibility_matrix_A.transpose(copy=True) # case for counting equal ones
    intersection = compatibility_matrix_A.dot(compatibility_matrix_A_T) # instance vs instance matrix

    compatibility_matrix_inv_A = 1 - train_set.loc[labeled_instances.index, label_columns] #inverse, 0 x 1 and viceversa 
    compatibility_matrix_inv_A_T = compatibility_matrix_inv_A.transpose(copy=True) # case for counting equal ones
    intersection_inv = compatibility_matrix_inv_A.dot(compatibility_matrix_inv_A_T) # instance vs instance matrix
    
    equal_labels = (intersection + intersection_inv)/len(label_columns)
    
    if ( unlabeled_instances is not None):
        # add the missing unlabeled data matrix
        unlabeled_index = unlabeled_instances.index

        # add unlabeled columns
        for i in unlabeled_index :
            equal_labels[i] = -1 #per column

        # add rows
        for i in unlabeled_index :
            equal_labels.loc[i] = -1 # per row
            
            
    return equal_labels

def generate_cosine_distance_based_compatibility(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set

    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")

    labels_orig = train_set.loc[labeled_instances.index, label_columns]
    #print(labels_orig)
    labels_sum = (train_set.loc[labeled_instances.index, label_columns].sum(axis='columns') )**(0.5)
    labels_sum_t = pd.DataFrame()
    
    for l in label_columns:
        labels_sum_t = labels_sum_t.append( labels_sum.transpose() , ignore_index = True )
    labels_sum = 1/labels_sum_t.transpose()
    labels_sum.columns = label_columns
    #print(labels_sum)
    
    labels_normalized = labels_orig.mul(labels_sum)
    labels_normalized_t = labels_normalized.transpose(copy=True)
    compatibility_matrix = labels_normalized.dot(labels_normalized_t)
    
    if ( unlabeled_instances is not None):
        # add the missing unlabeled data matrix
        unlabeled_index = unlabeled_instances.index

        # add unlabeled columns
        for i in unlabeled_index :
            compatibility_matrix[i] = -1 #per column

        # add rows
        for i in unlabeled_index :
            compatibility_matrix.loc[i] = -1 # per row
            
        
    return compatibility_matrix

    
def generate_compatibility_matrix(train_set, labeled_instances = None, unlabeled_instances = None, label_columns = []):
    if( labeled_instances is None):
        labeled_instances = train_set
        
    if( unlabeled_instances is None):
        # copy one instance and set as the unlabeled instance
        labeled_instances = train_set
    
    if(label_columns is None or len(label_columns)  == 0 ):
        print("Error, label columns should have the labels column names")
    
    compatibility_matrix_A = train_set.loc[labeled_instances.index, label_columns]
    compatibility_matrix_A_T = compatibility_matrix_A.transpose(copy=True)

    intersection = compatibility_matrix_A.dot(compatibility_matrix_A_T) # instance vs instance matrix

    # compatibility_A_T is the tranpose of just label columns. Agg counts and sets a row with all the sums per instance of all labels
    union = compatibility_matrix_A_T.agg(['sum']) # sum of ones 
    # transpose to generate a table of instances vs total number of 1's in labels
    union = union.transpose()
    #insert ones to operate after, a matrix multiplication
    union.insert(loc=0, column="ones", value= int(1)) # CA on paper
    # get the values to numpy array to thrash indices
    union_transpose_np_matrix = union.values
    #generate new dataframe with reordered rows, to be able to calculate union before intersection
    union = pd.DataFrame({'sum':union['sum'] , 'ones':union['ones']   }) # CB on paper
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


    if ( unlabeled_instances is not None):
        # add the missing unlabeled data matrix
        unlabeled_index = unlabeled_instances.index

        # add unlabeled columns
        for i in unlabeled_index :
            compatibility_matrix[i] = -1 #per column

        # add rows
        for i in unlabeled_index :
            compatibility_matrix.loc[i] = -1 # per row
            
        
    return compatibility_matrix

def pairwise_distances(X,Y=None,metric='euclidean'):
    if(metric=='euclidean'):
        return euclidean_distances(X,Y=Y,squared=False)
    #if(metric=='gower'):
        #print(X)
        #return gower.gower_matrix(X)
    if(metric=='cosine'):
        return cosine_distances(X,Y=Y)
    
    return None


def get_metrics( y_true = None, y_predicted = None, y_predicted_proba = None):
    results = dict()
    if(y_predicted_proba is not None):
        score_avg_precision = average_precision_score(y_true, y_predicted_proba)
        results["average_precision"] = score_avg_precision

        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(y_predicted_proba.shape[1]):
            precision[i], recall[i], _ = precision_recall_curve(y_true[:, i], y_predicted_proba[:, i])
            average_precision[i] = average_precision_score(y_true[:, i], y_predicted_proba[:, i])

        # A "micro-average": quantifying score on all classes jointly
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true.ravel(), y_predicted_proba.ravel()
        )
        average_precision["micro"] = average_precision_score(y_true, y_predicted_proba, average="micro")
        avg_pr_micro = average_precision["micro"]
        results["auprc_curve"] = avg_pr_micro
        
    if(y_predicted_proba is not None):
        label_score_avg_precision = label_ranking_average_precision_score(y_true, y_predicted_proba)
        results["label_rank_average_precision"] = label_score_avg_precision
        
    if(y_predicted is not None):
        score_accuracy = accuracy_score(y_true, y_predicted)
        results["accuracy"] = score_accuracy
        
    if(y_predicted is not None):
        score_hamming = hamming_loss(y_true, y_predicted)
        results["hamming_loss"]= score_hamming
        
    if(y_predicted_proba is not None):
        score_auc_micro = roc_auc_score(y_true, y_predicted_proba, average='micro')
        results["auc_micro"]= score_auc_micro
        
    if(y_predicted_proba is not None):    
        score_auc_macro = roc_auc_score(y_true, y_predicted_proba, average='macro')
        results["auc_macro"]= score_auc_macro

    return results