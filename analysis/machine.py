from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from astropy.table import Table, vstack
import numpy as np
import matplotlib.pyplot as plt
import pdb

def whiten(data):
    '''
    data is 2D array
    mean & std for each column used to normalize that column
    '''
    # Now we have a training sample. "Whiten" data.
    for col in range(data.shape[1]):
        mean = np.nanmean(data[:,col])
        std = np.nanstd(data[:,col])
        data[:,col] = (data[:,col]-mean)/std 
    return data

def extract_training(data, keys=['M20', 'C', 'elipt', 'A', 'G']):
    '''
    INPUTS:
        astropy Table (or dictionary?)
    
    PURPOSE:
        This function isolates those rows which have well-measured
        morphological features that can be fed into the machine classifiers, 
        i.e. no nans or infs
    
    RETURNS:
        original astropy Table with only those entries that pass selection
        whitened np.array of the morphological parameters    
    '''
    morph = np.array([data[k] for k in keys], dtype='float32').T

    # remove those which don't have morph parameters measured
    training = ((~np.isnan(morph).any(1)) & (~np.isinf(morph).any(1)))

    return data[training], whiten(morph[training])

def runKNC(X_train, y_train, X_test, N=10, weights='uniform', **kwargs):
    # initialize the classifier
    model = KNeighborsClassifier(N, weights=weights, **kwargs)

    # train the classifier with training sample
    knc = model.fit(X_train, y_train)

    # predict classifications for test sample
    predictions = knc.predict(X_test)

    # obtain probabilities for those predictions
    probabilities = knc.predict_proba(X_test)

    return predictions, probabilities, knc

def runRNC(X_train, y_train, X_test, R=1.0, weights='uniform', outlier=None):
    # initialize the classifier
    model = RadiusNeighborsClassifier(R, weights=weights, outlier_label=outlier)
    rnc = model.fit(X_train, y_train)
    predictions = rnc.predict(X_test)

    return predictions

def runRF(X_train, y_train, X_test, depth=None):
    # Initialize the classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=depth)
    RF = model.fit(X_train, y_train)
    predictions = RF.predict(X_test)
    probabilities = RF.predict_proba(X_test)

    return predictions, probabilities

def completeness_contamination(predicted, true):
    """Compute the completeness and contamination values
    Parameters
    ----------
    predicted_value, true_value : array_like
        integer arrays of predicted and true values.  This assumes that
        'false' values are given by 0, and 'true' values are nonzero.
    Returns
    -------
    completeness, contamination : float or array_like
        the completeness and contamination of the results.  shape is
        np.broadcast(predicted, true).shape[:-1]
    """

    predicted = np.asarray(predicted)
    true = np.asarray(true)

    outshape = np.broadcast(predicted, true).shape[:-1]

    predicted = np.atleast_2d(predicted)
    true = np.atleast_2d(true)

    matches = (predicted == true)

    tp = np.sum(matches & (true != 0), -1)
    tn = np.sum(matches & (true == 0), -1)
    fp = np.sum(~matches & (true == 0), -1)
    fn = np.sum(~matches & (true != 0), -1)

    tot = (tp + fn)
    tot[tot == 0] = 1
    completeness = tp * 1. / tot #same thing as Sensitivity or TPR

    tot = (tp + fp)
    tot[tot == 0] = 1
    contamination = fp * 1. / tot #same thing as False Discovery Rate

    tot = (tn + fp)
    tot[tot == 0] = 1
    fpr = fp * 1. / tot #False Positive Rate (FPR) or Fall-Out

    completeness[np.isnan(completeness)] = 0
    contamination[np.isnan(contamination)] = 0
    fpr[np.isnan(fpr)] = 0

    return [completeness.reshape(outshape), contamination.reshape(outshape), 
            fpr.reshape(outshape)]

# Need some plotting functions -- ROC plots, etc.

def plot_ROC(tpr, fpr, labels=None, filename=None):

    # ROC plot
    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(111)
    
    ls = ['-','--',':','-.']

    # ensure they are the same size
    if tpr.shape == fpr.shape and len(tpr) == len(labels):
        print "in the first if"

        pdb.set_trace()
        # if ndim > 1; we have multiple things to plot
        if tpr.ndim > 1:

            for idx, l in enumerate(labels):
                if idx > 3:
                    ax.plot(tpr[idx], fpr[idx], ls=[idx-4], lw=2, 
                            color='purple', label=l)
                else:
                    ax.plot(tpr[idx], fpr[idx], ls=[idx], lw=2, color='purple', 
                            label=l)
        else:
            ax.plot(tpr, fpr, ls=ls[0], lw=2, color='purple', label=labels)
            
        ax.plot([0.,1.], [0.,1.], 'k-', lw=2, label='Random Guess')

        ax.set_xlabel('False Positive Rate', fontsize=16)
        ax.set_ylabel('True Positive Rate (Completeness)', fontsize=16)

        ax.legend(loc='best')

        #plt.savefig('ROC_%s.png'%filename)
        #plt.show()
        #plt.close()
        return
    else: 
        print "TPR and FPR differenct sizes!"
        return
    
