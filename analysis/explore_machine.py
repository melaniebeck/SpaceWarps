import os, subprocess
from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import pdb
import swap
from optparse import OptionParser

def extract_training(data, keys=['M20', 'C', 'E', 'A', 'G']):
    '''
    INPUTS:
        astropy Table (or dictionary?)
    
    PURPOSE:
        This function isolates those rows which have well-measured morphological 
        features that can be fed into the machine classifiers, i.e. no nans or infs
    
    RETURNS:
        original astropy Table with only those entries that pass selection
        whitened np.array of the morphological parameters    
    '''
    morph = np.array([data[k] for k in keys], dtype='float32').T

    # remove those which don't have morph parameters measured
    training = ((~np.isnan(morph).any(1)) & (~np.isinf(morph).any(1)))

    return data[training], whiten(morph[training])

def gz2truth(data):
    answers = []
    for d in data:
        if 'E' in d['gz2class']: answers.append(1.)
        else: answers.append(0.)

    truth = Table(data=(data['t01_smooth_or_features_a01_smooth_debiased',
                    't01_smooth_or_features_a02_features_or_disk_debiased']),
                  names=('smooth%', 'not%'))
    truth['truth'] = answers
    return truth

def matches(predictions, truth):
    matches = (predictions == truth)
    total_correct = np.sum(matches)
    fraction_correct = float(total_correct)/len(truth)
    return total_correct, fraction_correct

def matches_above_threshold(predictions, probabilities, truth, threshold=.9):
    match_smooth = ((predictions==1) & (truth==1) & 
                    (probabilities['smooth%']>=threshold))
    match_not = ((predictions==0) & (truth==0) & 
                 (probabilities['not%']>=threshold)) 
    fraction_smooth = match_smooth/np.sum(((predictions==1) & (truth==1)))
    fraction_not = match_not/np.sum(((predictions==0) & (truth==0)))

    return [match_smooth, match_not, fraction_smooth, fraction_not]

def completeness_contamination(prediction, truth):
    """Compute the completeness and contamination values
    Parameters
    ----------
    prediction, truth : array_like
        integer arrays of predicted and true values.  This assumes that
        'false' values are given by 0, and 'true' values are nonzero.
    Returns
    -------
    completeness, contamination : float or array_like
        the completeness and contamination of the results.  shape is
        np.broadcast(predicted, true).shape[:-1]
    """

    predicted = np.asarray(prediction)
    true = np.asarray(truth)

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



def main(options, args):

    # read in the machine's classifications
    try: config = options.configfile
    except: pdb.set_trace()

    tonights = swap.Configuration(config)
    survey = tonights.parameters['survey']

    if options.offline:
        # read in a slew of directories
        try:
            directories = subprocess.check_output("ls -d $PWD/sup_run4/%s*/"\
                                                  %survey,
                                                  stderr=subprocess.STDOUT,
                                                  shell=True).splitlines()
        except subprocess.CalledProcessError:
            print "No directories found for survey '%s'.\nAborting.\n"%survey
            return
    else:
        # read in the "current" night's files
        try:
            # These parameters will hold the locations of where SWAP just saved
            # the night's data; unlike "start" and "end" which will have already 
            # been updated to reflect the next night's run
            directories = [tonights.parameters['dir']+'/']
        except: 
            print "Output from SWAP not found!\nAborting.\n"
            return


    threshold = 0.9

    total, fraction = np.array([]), np.array([])
    completeness, contamination, false_positive_rate = [], [], []
    match_smooth_above_thresh,  match_not_above_thresh = [], []
    predict_smooth_above_thresh, predict_not_above_thresh = [], []
    match_smooth, match_not = [], []
    good_smooth_idx, good_not_idx = {}, {}
    night = 16
    for directory in directories[:8]:

        trunk = directory.split('/')[-2]

        # Read in tonight's test sample
        try: 
            answerfile = '%s%s_machine_testsample.fits'%(directory,trunk)
            print "Read in test sample: %s"%answerfile
            test_subjects = Table.read(answerfile, format='fits')
            truth = gz2truth(test_subjects)
        except:
            print "Cannot find test sample for %s!\nAborting.\n"%trunk
            return
                    
        # Read in tonight's machine output for the training sample
        try: 
            trainingfile = '%s%s_machine.txt'%(directory,trunk)
            machine = Table.read(trainingfile, format='ascii')
        except:
            print "Cannot find machine output for %s!\nAborting.\n"%trunk
            return

        print "Processing machine output from %s"%trainingfile

        
        tt, ff = matches(machine['prediction'], truth['truth'])
        total = np.append(total, tt)
        fraction = np.append(fraction,ff)
        
        predictions, answer = machine['prediction'], truth['truth']
        probabilities = machine['smooth%','not%']
        
        """------------------ MATCHES ABOVE THRESHOLD ----------------------"""
        msat = ((predictions==1) & (answer==1) &
                (probabilities['smooth%']>=threshold))
        match_smooth_above_thresh.append(np.sum(msat))

        mnat = ((predictions==0) & (answer==0) & 
                (probabilities['not%']>=threshold)) 
        match_not_above_thresh.append( np.sum(mnat))

        """---------------- PREDICTIONS ABOVE THRESHOLD --------------------"""
        psat = ((predictions==1)&(probabilities['smooth%']>=threshold))
        predict_smooth_above_thresh.append(np.sum(psat))

        pnat = ((predictions==0)&(probabilities['not%']>=threshold))
        predict_not_above_thresh.append(np.sum(pnat))

        """----------------------- TOTAL MATCHES ---------------------------"""
        match_smooth.append(np.sum(((predictions==1) & (answer==1)))*1.0/
                            len(predictions))
        match_not.append(np.sum(((predictions==0) & (answer==0)))*1.0/
                         len(predictions))

        # Indexes of SMOOTH MATCHES ABOVE THRESHOLD
        good_smooth_idx['night_%i'%night] = match_smooth
        # Indexes of NOT MATCHES ABOVE THRESHOLD
        good_not_idx['night_%i'%night] = match_not
        night+=1

        [comp, cont, fpr] = completeness_contamination(predictions, answer)
        completeness.append(comp)
        contamination.append(cont)
        false_positive_rate.append(fpr)
        #pdb.set_trace()

    
    # Ratio of MATCHES / PREDICTIONS (above threshold: AT)
    match_predict_AT_smooth = np.array(match_smooth_above_thresh)*1./\
                              np.array(predict_smooth_above_thresh)
    match_predict_AT_not = np.array(match_not_above_thresh)*1./\
                           np.array(predict_not_above_thresh)

    # Ratio of MATCHES AT / TOTAL MATCHES
    match_AT_totmatch_smooth = np.array(match_smooth_above_thresh)*1./\
                               np.array(match_smooth)
    match_AT_totmatch_not = np.array(match_not_above_thresh)*1./\
                            np.array(match_not)

    pdb.set_trace()

    fig = plt.figure(figsize=(10,10))
    ax = fig.add_subplot(211)
    
    ax.plot(completeness, 'k', label='completeness')
    ax.plot(contamination, 'r', label='contamination')
    ax.set_xlabel('Days in Sim')
    ax.set_ylabel('Per Cent')
    ax.legend(loc='best')
    
    ax2 = fig.add_subplot(212)
    ax2.plot(fraction, label='All matches')
    ax2.plot(match_smooth, 'b', label='SMOOTH matches')
    ax2.plot(match_not, 'r', label='NOT matches')
    ax2.set_xlabel('Days in Sim')
    ax2.set_ylabel('Fraction Correct')
    ax2.legend(loc='best')
    
    plt.tight_layout()
    plt.savefig('machine_output_sup_run4.png')
    plt.show()
    pdb.set_trace()


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", dest="configfile", default=None)
    parser.add_option("-o", "--offline", dest="offline", default=False,
                      action='store_true',
                      help="Run in offline mode; e.g. on existing SWAP output.")

    
    (options, args) = parser.parse_args()

    main(options, args)
