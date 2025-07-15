import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from DownloadHelper import *
from pandas.plotting import scatter_matrix
from random import random
from DTNode import DecisionTreeNode, DecisionTree
from sklearn.datasets import fetch_openml
import time
print(__doc__)


MAX_VARIANCE = 0
MAX_VARIANCE_TOTAL = 0
MAX_VARIANCE_ROW = 1
MAX_VARIANCE_ROW_INDEX = 2
LEFT_SIDE = 1
RIGHT_SIDE = 2
COMPATIBILITY_AVG = 3
COLUMN = 4


TRUE_VALUE = 1
FALSE_VALUE = 0

ONE = 1
ZERO = 0

def getSumAndSumSquared(data):
    label = data.columns[1]
    simpleAverage = 0
    squaredAverage = 0
    for index,row in data.iterrows():
        v= row[label]
        simpleAverage = simpleAverage  + v
        squaredAverage = squaredAverage + v*v

    return simpleAverage, squaredAverage

def heuristicFunction(leftTotal, leftSimpleSum, leftSquaredSum, rightTotal, rightSimpleSum, rightSquaredSum, totalVariance):

    total = leftTotal + rightTotal
    if( leftTotal <= 1):
        return -np.inf
    if( rightTotal <= 1):
        return -np.inf

    tVar = totalVariance
    lVar = (leftSquaredSum-(leftSimpleSum*leftSimpleSum)/leftTotal)/leftTotal  # second part is divided two times by left total as it is the squared!!
    rVar = (rightSquaredSum-rightSimpleSum*rightSimpleSum/rightTotal)/rightTotal
    return tVar - (leftTotal/total*lVar + rightTotal/total*rVar)
"""
receives subset of column and original index, in that order, already sorted
"""
def varianceReduction(subset):
    rowPosition = 1
    totalRows = subset.shape[0]

    # this will hold the maxHeur and the row position that made it possible

    averageSum, averageSumSq = getSumAndSumSquared(subset)
    variance = (averageSumSq - averageSum*averageSum/totalRows)/totalRows
    maxHeur = [-np.inf,rowPosition, -1]


    leftSideTotal = 1
    rightSideTotal = totalRows-1
    leftSimpleSum = 0
    leftSquaredSum = 0
    rightSimpleSum = averageSum # left originally has 0 elements, leftSideTotal has 1 to start with something
    rightSquaredSum = averageSumSq # first part, right side and total are the same
    #1 get total average
    for index,row in subset.iterrows():
        actualItem = row[subset.columns[1]] # actual item to be left out of right and summed to left
        leftSimpleSum = leftSimpleSum + actualItem
        leftSquaredSum = leftSquaredSum + actualItem*actualItem
        rightSimpleSum = rightSimpleSum - actualItem
        rightSquaredSum = rightSquaredSum - actualItem*actualItem

        heur = heuristicFunction( leftSideTotal , leftSimpleSum , leftSquaredSum, rightSideTotal , rightSimpleSum , rightSquaredSum , variance)
        if( heur > maxHeur[0]):
            #print("was better " , heur , " than " ,  maxHeur[0] , " on " ,  rowPosition)
            maxHeur[0] = heur
            maxHeur[1] = rowPosition
            maxHeur[2] = index
        #print(" Heur " ,  heur)
        rowPosition = rowPosition+1
        leftSideTotal = leftSideTotal + 1
        rightSideTotal = rightSideTotal - 1


    # now, with maxHeur row position, gather left side and right side
    leftSide = list()
    rightSide = list()
    #print("MAX HEUR " , maxHeur)
    for i in range(0,maxHeur[1]):
        leftSide.append( subset.iloc[i,0])

    for i in range(maxHeur[1],subset.shape[0]):
        rightSide.append( subset.iloc[i,0])

    # warning this is returning a 5 space vector.
    return [maxHeur,leftSide, rightSide]


# must get a better algorithm than gettint N*row.mean
def getCompatibilityAverage(nodesList, compatibility):
    total = 0.0
    #print(compatibility)
    #print(nodesList)

    #for j in compatibility:
     #   print( j )

    for i in nodesList:
        row = compatibility.loc[i,nodesList]
        total = total + (row.mean())
    return (total/len(nodesList))


# as this just checks the first column, we are implying that a unlabeled instance is all unlabeled. Right now there is no suppoort por partial unlabeled data. Have not decided what to do with that (20-11-2019 , comment is before actual writing date cause it was that finding date)



def totalSetLabeled(subset):
    #print("---------%%%%%%%%%%%%%%%%%%----------- " , subset)
    totalLabeledData = 0
    labeledDataList = []
    unlabeledInstances = []
    for index,row in subset.iterrows():
        for i in range(0, row.count()  ): # row.count returns column count
            thisLabeledData = 0

            # initalized list to eventually be same size as row.columns
            if( len(labeledDataList) < (i+1) ):
                labeledDataList.append( [0,0] )

            if( row[i] ==TRUE_VALUE):
                if(  labeledDataList[i][ONE] == -1):
                     labeledDataList[i][ONE] = 0
                thisLabeledData = 1
                labeledDataList[i][ONE] = labeledDataList[i][ONE] + 1

            if( row[i] == FALSE_VALUE):
                if(  labeledDataList[i][ZERO] == -1):
                     labeledDataList[i][ZERO] = 0
                thisLabeledData = 1
                labeledDataList[i][ZERO] = labeledDataList[i][ZERO] + 1

        # just check if this row was labeled
        if( thisLabeledData > 0 ):
            totalLabeledData = totalLabeledData + 1
        else:
            unlabeledInstances.append(index)
    return totalLabeledData,labeledDataList, unlabeledInstances


# Load a multi-label dataset from https://www.openml.org/d/40597
"""
Multi-label dataset. The yeast dataset (Elisseeff and Weston, 2002) consists of micro-array
expression data, as well as phylogenetic profiles of yeast, and includes 2417 genes and
103 predictors. In total, 14 different labels can be assigned to a gene, but only 13 labels
were used due to label sparsity.

Since one gene can have many functional classes this is a multi-label problem: one gene is
associated to different edges. We then have Q=14 and the average number of labels for
all genes in the learning set is 4.2

http://papers.nips.cc/paper/1964-a-kernel-method-for-multi-labelled-classification.pdf
"""


originalTime = ( time.time() )
hamming_loss_counter = 0
yeast = getFromOpenML('yeast',version=4,ospath='online_download/', download=False, save=True)

toClassify = yeast.iloc[2001:2201]
yeast = yeast.iloc[0:2000]
redoCompatibilityMatrix = False

#yeast.info()
#(yeast.describe(include='all')).to_csv(getFullURI("data_analysis.csv"))


# will not focus on labels not SET!!!!! -> this will change

# convert this to fit _ transform
# should convert 1 and 0 ?
# this still needs the index for labels column
columnNames = []
amountOfLabels = 14

labelColumnIndex = len(list(yeast.columns))-amountOfLabels
labelColumns = []
= list(yeast.columns)

labelsDone = False
totalTrueLabelsDataFrame = pd.DataFrame(0,index=yeast.index, columns=["total_true_labels"])

indexCol = 0
labelColumnsByIndex = []

for col in originalLabelColumns:
    if indexCol >= labelColumnIndex :
        print(col)
        labelColumnsByIndex.append(indexCol)
        labelColumns.append(col)
        yeast[col] = yeast[col].astype(int) # convert to ones and zeros
    indexCol = indexCol + 1
# ended list of columns

for index,row in yeast.iterrows():
    t = (index)
    columnNames.append(t)

    totalTrue = 0
    indexCol = 0

    #simulate unsupervised step, erase stuff from certain instances

    for column in row:
        if( indexCol >= labelColumnIndex ):

            """
            # ***************** next if should not be used when real life is applied
            if( index in [1,3,5,7,9,12,15]): # change this for a list of number to be used when trying to test unsupervised
                yeast.iloc[index, indexCol] = -1
            # ********************** end the false life.
            """
            if( column == TRUE_VALUE ):
                totalTrue = totalTrue + 1
        indexCol = indexCol + 1

    totalTrueLabelsDataFrame.iloc[index]["total_true_labels"] = totalTrue

print( totalTrueLabelsDataFrame )





if redoCompatibilityMatrix :

    compatibilityMatrix = pd.DataFrame(0,index=columnNames,columns=columnNames)
    compatibilityMatrix["total_true_labels"] = int(0)

    # change numbers to float
    compatibilityMatrix.loc[1:,0:len(list(compatibilityMatrix.columns))]  = np.float64(0)

    ## heym, index and  [access] is different!!!
    for index,row in compatibilityMatrix.iterrows():
            row["total_true_labels"] = int(totalTrueLabelsDataFrame.iloc[index]["total_true_labels"])

    print(compatibilityMatrix)

    # algorithm start

    compatMatrixColumnCount = len(list( compatibilityMatrix.columns))
    startIndex = -1
    # itertuples should be faster than iterrows
    for row in compatibilityMatrix.itertuples():
        rowIndex = row[0]
        startIndex = startIndex+1 #trick for making the method triangular
        for col in range(startIndex, compatMatrixColumnCount-1): # inclusive range
            # the cycle should do this only for columns that are not total_true_labels
            if col != "total_true_labels":
                union = 0.
                intersection = 0.
                for label in labelColumns:
                    if( yeast.iloc[rowIndex][label]  == TRUE_VALUE and  yeast.loc[col][label] ==  TRUE_VALUE  ):
                       intersection = intersection + 1.0
                    if( yeast.iloc[rowIndex][label]  == TRUE_VALUE or  yeast.loc[col][label] == TRUE_VALUE  ):
                       union = union + 1.0
                if( union > 0):
                    compatibilityMatrix.iloc[rowIndex][col] = (float(intersection)/float(union))
                    compatibilityMatrix.iloc[col][rowIndex] = (float(intersection)/float(union))

        print( "On row "  , rowIndex)
    saveFile(compatibilityMatrix,"compMatrix.csv")

else:
    # read and change header indices to ints!
    print("column names ")
    _tempColumnLabels = {'0':0}
    columnNames.append("total_true_labels")

    for i in range(0,len(columnNames)):
        nm = ""+str(i)+""
        _tempColumnLabels[nm] = i
    #columnLabels["total_true_labels"] = "total_true_labels"

    print( _tempColumnLabels )
    compatibilityMatrix = pd.read_csv('compMatrix.csv').rename(columns=_tempColumnLabels)

print("Will print comp matrix!");
print( compatibilityMatrix)
print( compatibilityMatrix.info())


# maintain the original ids given

yeast['original_index'] = yeast.index
indexCol = len(originalLabelColumns)
print(yeast)



"""
for each column, get the average of the compatibility
matrix for left side and right side
"""





def generateTree(instanceList, limit, level, totalInstances , amountStopPercentage,parentNode, tree):
    print("Level: " , level , ". Received instance list with ids: " , instanceList)

    bestSplitPoint = [[],[],[],-np.inf, 0]
    bestSplitPoint[MAX_VARIANCE].append(0)
    bestSplitPoint[MAX_VARIANCE].append(0)
    bestSplitPoint[MAX_VARIANCE].append(0)
    bestSplitPoint[LEFT_SIDE] = []
    bestSplitPoint[RIGHT_SIDE] = []
    bestSplitPoint[COMPATIBILITY_AVG] = -1
    bestSplitPoint[COLUMN] = 0

    totalLabeled = np.inf
    onesZerosMatrixCount = 0
    unlabeledInstances = 0
       #requirements, amount of total instances, method for checking labeled instances
    if(instanceList != 0):
        totalLabeled, onesZerosMatrixCount , unlabeledInstances = totalSetLabeled(yeast.iloc[instanceList, labelColumnsByIndex ])

    if( instanceList != 0 and (len(instanceList) < totalInstances*amountStopPercentage  or ( totalLabeled < 2) )):
    # if there is 20% or less instances than the total amount of instances, or all of the instances are unlabeled
            #assign mode of labels for each instance (met 1)
            #assign random weighted label for each instance (met 2)

        #print( "Amount  " , (len(instanceList) < totalInstances*amountStopPercentage)  , " //// "  ,  totalLabeled < 2 )
        print( onesZerosMatrixCount)

        print( yeast.iloc[instanceList, : ])
        print("Ended decision tree, [unsupervised step] fill the blanks")


        # big question: if the amount of unlabeled data is greater than the amount of labeled data, the nodes are left unchanged?
        # right now, it does not matter, i'llll doo it anyway

        # for each row that is unlabeled, for each column, set the weighted random
        if( totalLabeled > 0  ):
            for i in range(0, len(unlabeledInstances) ):
                row = yeast.iloc[unlabeledInstances[i]]
                print( "    Filling row  " , row )

                classificationLabelIndex = len(list(yeast.columns))-amountOfLabels-1
                k = 0
                for  j in range(classificationLabelIndex, classificationLabelIndex+amountOfLabels):
                    totalOnePercentage = onesZerosMatrixCount[k][ONE]/(onesZerosMatrixCount[k][ONE]+onesZerosMatrixCount[k][ZERO])
                    a  = 1 if random() > (1-totalOnePercentage) else 0
                    yeast.loc[unlabeledInstances[i] , row.index[j]] = a
                    k = k + 1
        # onesZerosMatrixCount has the amount of ones and zeros per column
        # right now, we are losing the unlabeled data count!
        return DecisionTreeNode(-1, -1, parentNode, onesZerosMatrixCount )

    else:
        # if there is 20% or labeled instances and there are at least 2 instances labeled
            #get variance champions
            # get compatibility tree
        print("Calculating champion for each column variance reduction" )
       # print( yeast.iloc[instanceList, : ])
        for col in range(0,len(originalLabelColumns)-amountOfLabels):

            # should obtain best split point tuple:column, rowIndex, splitValue
            #for col in range(0,1):
            thisColumnSubset = 0
            if( instanceList == 0):
                thisColumnSubset = (yeast.iloc[:,[indexCol,col]])
            else:
                thisColumnSubset = (yeast.iloc[instanceList,[indexCol,col]])

            # this should not be useful anymore.
            #if( thisColumnSubset.shape[0] <= limit):
            #    return

            #print(originalLabelColumns[col])
            # possible point of low efficiency
            thisColumnSubset = thisColumnSubset.sort_values(originalLabelColumns[col],axis=0)


            bestSplitForColumn = varianceReduction(thisColumnSubset)
            #print(bestSplitForColumn) .... the compatibility average is to test wether the nodes correspond to each other. optimization by andres, implemented by joe
            if( bestSplitForColumn[MAX_VARIANCE][MAX_VARIANCE_TOTAL] > bestSplitPoint[MAX_VARIANCE][MAX_VARIANCE_TOTAL]):
                leftAvg = getCompatibilityAverage(bestSplitForColumn[1], compatibilityMatrix)
                rightAvg = getCompatibilityAverage(bestSplitForColumn[2], compatibilityMatrix)
                prom = leftAvg*0.5+rightAvg*0.5 # average of both sides
                #print(prom , " :: " , bestSplitPoint[COMPATIBILITY_AVG])
                if( prom > bestSplitPoint[COMPATIBILITY_AVG] ):
                    bestSplitPoint[COLUMN] = col
                    bestSplitPoint[MAX_VARIANCE][MAX_VARIANCE_TOTAL] = bestSplitForColumn[MAX_VARIANCE][MAX_VARIANCE_TOTAL]
                    bestSplitPoint[MAX_VARIANCE][MAX_VARIANCE_ROW] = bestSplitForColumn[MAX_VARIANCE][MAX_VARIANCE_ROW]
                    bestSplitPoint[MAX_VARIANCE][MAX_VARIANCE_ROW_INDEX] = bestSplitForColumn[MAX_VARIANCE][MAX_VARIANCE_ROW_INDEX]
                    bestSplitPoint[LEFT_SIDE] = bestSplitForColumn[LEFT_SIDE]
                    bestSplitPoint[RIGHT_SIDE] = bestSplitForColumn[RIGHT_SIDE]
                    bestSplitPoint[COMPATIBILITY_AVG] = prom


        print("Achieved one best column in leve: " , level , "[SplitPoint:MAX_VARIANCE{value,row number on algorithm, idx},left idxs. right idxs, compatibility avg, column number ]")
        print( bestSplitPoint )
        # save best split point on data structure

        # double link node list (parent access children and children access parent)
        node = DecisionTreeNode(bestSplitPoint[COLUMN], yeast.iloc[bestSplitPoint[MAX_VARIANCE][MAX_VARIANCE_ROW_INDEX]][bestSplitPoint[COLUMN]], parentNode )
        # set all the needed stuff

        left_node = generateTree(bestSplitPoint[LEFT_SIDE], limit, level+1 , totalInstances,amountStopPercentage, node, tree)
        right_node =generateTree(bestSplitPoint[RIGHT_SIDE], limit, level+1 , totalInstances,amountStopPercentage , node ,tree)

        node.set_left_node(left_node)
        node.set_right_node(right_node)

        if( parentNode is None):
            tree.add_root_node(node)

        return node

def classify(row, tree):
    global hamming_loss_counter,true_positive_sum,false_positive_sum, true_negative_sum,false_negative_sum
    node = tree.classify(row)
    prev_node = 0
    #print(row)
    while( True ):
        prev_node=node
        node = node.decide(row)
        if( node == prev_node):
            break
    #print( node.print_node(0) )
    node.check_original_assignment(row)

    true_positive_sum = true_positive_sum + node.true_positive
    false_positive_sum = false_positive_sum + node.false_positive
    true_negative_sum = true_negative_sum + node.true_negative
    false_negative_sum  = false_negative_sum + node.false_negative
    hamming_loss_counter  = hamming_loss_counter + node.true_negative + node.false_positive
    return node



decision_tree = DecisionTree()
generateTree(0, 3 , 0, yeast.shape[0], 0.2, None , decision_tree )
print( yeast)
# 601 - 620
decision_tree.print_tree()

print( (time.time()-originalTime) )
total_nodes_classified = 199 # as we decided this for a certain amount of testing
true_positive_sum = 0
false_positive_sum = 0
true_negative_sum = 0
false_negative_sum = 0



for i in range(0,199) :
    classify( toClassify.loc[2001+i]  , decision_tree)
    print( (time.time()-originalTime) )

hamming_loss = 1.0/(total_nodes_classified*amountOfLabels)*hamming_loss_counter
precision = true_positive_sum/(true_positive_sum+false_positive_sum)
recall = true_positive_sum/(true_positive_sum+true_negative_sum)
accuracy =  (true_positive_sum + true_negative_sum) / (true_positive_sum +false_positive_sum + true_negative_sum + false_negative_sum )
f1_score = 2*precision*recall/(precision+recall)

print("Hamming Loss:" + str(hamming_loss) )
print("precision:" + str(precision) )
print("recall Loss:" + str(recall) )
print("accuracy:" + str(accuracy) )
print("f1_score:" + str(f1_score) )


saveFile(yeast,"yeastRefilled.csv")
