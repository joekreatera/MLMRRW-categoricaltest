import numpy
import pandas

def roc_auc(y_true, y_probs, **kwargs):
    """
    y_true: ground truth
    y_probs: assignment probability
    """

    order_index = numpy.flip( y_probs.argsort(axis=1) , axis = 1 )

    tps = numpy.zeros( y_probs.shape[1] )

    fps = numpy.zeros( y_probs.shape[1] )


    for j in range(0, y_probs.shape[1] ):
        # computed_y(:) = 0
        numpy_array = numpy.zeros(y_probs.shape )

        for i in range(0, y_probs.shape[0]):
            for k in range(0, j+1 ):
                numpy_array[i,order_index[i,k] ] = 1

            # print(f'{j} {i} {numpy_array[i]}')

        tp = ((y_true*numpy_array).sum(axis=0)).sum()
        fp = (( (1-y_true)*numpy_array).sum(axis=0)).sum()
        tn = (( (1-y_true)*(1-numpy_array) ).sum(axis=0)).sum()
        fn = (( y_true*(1-numpy_array) ).sum(axis=0)).sum()


        tps[j] = tp/(tp+fn)
        fps[j] = fp/(fp+tn)

    # print(tps)
    false_positives = fps
    true_positives = tps

    auc = 0



    for i in range(y_probs.shape[1]-1):

        x_diff = abs(false_positives[i+1]-false_positives[i])

        y_sum = true_positives[i+1]+true_positives[i]
        to_add = 0.5*x_diff*y_sum
        auc = auc + to_add
        # print(auc)

    # should be
    # "birds_vrssml__tt_u0p1"    "vrssml__tt_u0p1"    "0.60479"
    return auc
