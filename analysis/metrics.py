import numpy as np

def array_equal(a1, a2):
    return bool(np.asarray(a1 == a2).all())

def _binary_clf_curve(y_true, y_score, pos_label=None, sample_weight=None):
    """Calculate true and false positives per binary classification threshold.
    Parameters
    ----------
    y_true : array, shape = [n_samples]
        True targets of binary classification
    y_score : array, shape = [n_samples]
        Estimated probabilities or decision function
    pos_label : int, optional (default=None)
        The label of the positive class
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    Returns
    -------
    fps : array, shape = [n_thresholds]
        A count of false positives, at index i being the number of negative
        samples assigned a score >= thresholds[i]. The total number of
        negative samples is equal to fps[-1] (thus true negatives are given by
        fps[-1] - fps).
    tps : array, shape = [n_thresholds <= len(np.unique(y_score))]
        An increasing count of true positives, at index i being the number
        of positive samples assigned a score >= thresholds[i]. The total
        number of positive samples is equal to tps[-1] (thus false negatives
        are given by tps[-1] - tps).
    thresholds : array, shape = [n_thresholds]
        Decreasing score values.
    """
    #check_consistent_length(y_true, y_score)
    if len(y_true) == len(y_score):
        #y_true = column_or_1d(y_true)
        #y_score = column_or_1d(y_score)
        #if sample_weight is not None:
        #    sample_weight = column_or_1d(sample_weight)

        # ensure binary classification if pos_label is not specified
        classes = np.unique(y_true)
        if (pos_label is None and
            not (array_equal(classes, [0, 1]) or
                 array_equal(classes, [-1, 1]) or
                 array_equal(classes, [0]) or
                 array_equal(classes, [-1]) or
                 array_equal(classes, [1]))):
            raise ValueError("Data is not binary and pos_label is not specified")
        elif pos_label is None:
            pos_label = 1.

        # make y_true a boolean vector
        y_true = (y_true == pos_label)

        # sort scores and corresponding truth values
        desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
        y_score = y_score[desc_score_indices]
        y_true = y_true[desc_score_indices]
        #if sample_weight is not None:
        #    weight = sample_weight[desc_score_indices]
        #else:
        weight = 1.

        # y_score typically has many tied values. Here we extract
        # the indices associated with the distinct values. We also
        # concatenate a value for the end of the curve.
        # We need to use isclose to avoid spurious repeated thresholds
        # stemming from floating point roundoff errors.
        distinct_value_indices = np.where(np.logical_not(np.isclose(
            np.diff(y_score), 0)))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

        # accumulate the true positives with decreasing threshold
        tps = (y_true * weight).cumsum()[threshold_idxs]
        if sample_weight is not None:
            fps = weight.cumsum()[threshold_idxs] - tps
        else:
            fps = 1 + threshold_idxs - tps

    return fps, tps, y_score[threshold_idxs]


def compute_binary_metrics(fps, tps):
    """
    fps = False Positives (Number count along a moving threshold)
    tps = True Positives 
    
    Uses these to compute various metrics for a binary 
    confusion matrix:
    
        Precision, Recall (True Positive Rate), False Positive Rate, 
        False Discovery Rate, False Omission Rate, Specificity, 
        False Negative Rate, etc. 
    """
    
    # compute False Negatives and True Negatives
    fns, tns = tps[-1] - tps, fps[-1]-fps

    TPR = tps / tps[-1]  # True Positive Rate; Recall/Completeness 
    FPR = fps / fps[-1]  # False Positive Rate
    FNR = fns / fns[0]   # False Negative Rate
    TNR = tns / tns[0]   # True Negative Rate; Specificity

    PPV = tps / (tps + fps)   # Positive Predictive Value (Precision)
    FDR = fps / (tps + fps)   # Also known as Contamination
    FOR = fns / (fns + tns)   # "Contamination" of the other condition
    NPV = tns / (fns + tns)   # Negative Predictive Value 
    
    ACC = ( tps + tns ) / ( tps + fps + tns + fns )

    return [ACC, TPR, FPR, FNR, TNR, PPV, FDR, FOR, NPV]
