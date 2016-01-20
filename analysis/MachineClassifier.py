
import swap
from optparse import OptionParser
from astropy.table import Table, vstack
import pdb
import datetime
import numpy as np
import os, subprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import cPickle

'''
Workflow:
   access morphology database
   accept labels/data for training
   accept labels/data for testing
   "whiten" data (normalize)
   {reduce dimensions} (optional)
   train the machine classifier
   run machine classifier on test sample
'''

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

def runKNC(X_train, y_train, X_test, N=10, weight='uniform'):
    # initialize the classifier
    model = KNeighborsClassifier(N, weights=weight)

    # train the classifier with training sample
    model.fit(X_train, y_train)

    # predict classifications for test sample
    predictions = model.predict(X_test)

    # obtain probabilities for those predictions
    probabilities = model.predict_proba(X_test)

    return predictions, probabilities

def runRF(X_train, y_train, X_test, depth=None):
    # Initialize the classifier
    model = RandomForestClassifier(n_estimators=100, max_depth=depth)

    # Trainin classifier with training sample
    model.fit(X_train, y_train)

    # Predict classifications for test sample
    predictions = model.predict(X_test)

    # Obtain probabilities for thos predictions
    probabilities = model.predict_proba(X_test)

    return predictions, probabilities


def MachineClassifier(options, args):

    try: config = options.configfile
    except: pdb.set_trace()

    tonights = swap.Configuration(config)

    # Read the pickled random state file
    random_file = open(tonights.parameters['random_file'],"r");
    random_state = cPickle.load(random_file);
    random_file.close();
    np.random.set_state(random_state);

    survey = tonights.parameters['survey']
    subdir = 'sup_run4'

    # read in the FULL collection of subjects
    full_sample = swap.read_pickle(tonights.parameters['fullsamplefile'],
                                   'full_collection')
    # read in the SWAP collection
    sample = swap.read_pickle(tonights.parameters['samplefile'],'collection')


    train = full_sample[full_sample['MLsample']=='train']
    test = full_sample[full_sample['MLsample']=='test']
    train_data, train_sample = extract_training(train, 
                                            keys=['M20','C','elipt','A','G'])
    test_data, test_sample = extract_training(train, 
                                            keys=['M20','C','elipt','A','G'])
    labels = np.array([1 if p > 0.3 else 0 for p in train_data['label']])
 
                                        
    predictions, probabilities = runKNC(train_sample, labels, test_sample)
    # for subjects with probabilities > threshold, create an entry in the 
    # SWAP Collection
    
    pdb.set_trace()
                      

    # read out results of ML to file ... 
    # In order to be "Like SWAP", need to figure out which subjects can be
    # "retired" -- which are classified well enough. 
    output = Table(data=probabilities, names=('not%', 'smooth%'))
    output['prediction']=predictions
    Table.write(output,'%s%s_machine.txt'%(directory,trunk), format='ascii')
        
                                

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", "--config", dest="configfile", 
                      help="Name of config file")
    parser.add_option("-o", "--offline", dest="offline", default=False,
                      action='store_true',
                      help="Run in offline mode; e.g. on existing SWAP output.")

    (options, args) = parser.parse_args()
    MachineClassifier(options, args)


    """
    # I don't think I need to look through all the output files... 
    if options.offline:
        # read in a slew of directories
        try:
            directories = subprocess.check_output("ls -d -1 $PWD/%s/%s*/"\
                                                  %(subdir,survey),
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

    # Pull up all "test" subjects -- don't need this with new system
    subjects = Table.read('GZ2assets_Nair_Morph_zoo2Main.fits')

    # Read in the Catalogs (training data) -- "retired" and "candidate" catalogs
    # if offline, there will be many to do...
    for directory in directories:
        trunk = directory.split('/')[-2]
        print "Reading candidate/retired catalogs from %s"%trunk

        # read in candidates catalog
        try:
            candidates = Table.read('%s/%s_candidate_catalog.txt'\
                                    %(directory,trunk), format='ascii') 
            if len(candidates) > 0:
                # not all subjects have morphological parameters -- 
                # select only those rows that do
                candidates, candidates_morph = extract_training(candidates)
                candidate_objects = True
            else:
                print "There are no candidate subjects in %s to feed into "\
                    "Machine Classifier(s)"%trunk
                candidate_objects = False 
        except:
            print "There are no candidate subjects in %s to feed into Machine "\
                "Classifier(s)"%trunk
            candidate_objects = False

        # read in retired catalog
        try:
            retired = Table.read('%s/%s_retired_catalog.txt'\
                                 %(directory,trunk), format='ascii')
            if len(retired) > 0:
                retired, retired_morph = extract_training(retired)
                retired_objects = True
            else:
                print "There are no rejected subjects in %s to feed into "\
                    "Machine Classifier(s)."%trunk
                retired_objects = False 
        except:
            print "There are no rejected subjects in %s to feed into Machine "\
                "Classifier(s)."%trunk
            retired_objects = False
            
        if candidate_objects and retired_objects:
            # stack both candidates and retireds together
            training_subjects = vstack([candidates, retired])
            training_sample = np.vstack([candidates_morph, retired_morph])
            SWAP_probabilities = np.concatenate([np.array(candidates['P']), 
                                                  np.array(retired['P'])])
        elif retired_objects: # retired =  
            training_subjects = retired
            training_sample = retired_morph
            SWAP_probabilities = retired['P']
        elif candidate_objects:
            training_subjects = candidates
            training_sample = candidates_morph
            SWAP_probabilities = candidates['P']
        else:
            # something went wrong
            print "Something went wrong... no data?"
            return
            
        print "Training sample consists of %i subjects."%len(training_sample)

        # Add 'target' labels (SMOOTH: 1, NOT: 0)
        # Assuming that EVERYTHING seen by users is being fed to SWAP & Machine
        # SWAP assigns probability to subject; if > prior -> SMOOTH; else NOT
        labels = np.array([])
        for p in SWAP_probabilities:
            if p >= 0.3: 
                labels = np.append(labels, 1)
                #weights = np.append(weights, p)
            else: 
                labels = np.append(labels, 0)
                #weights = np.append(weights, 1.-p)

        # But what if we don't want to send EVERYTHING to Machine? 
        # Just send objects which have crossed rejection/detection thesholds!
        

        # Finally, test_sample contains EVERYTHING -- 
        # pull out those that are in the training sample -- don't overfit
        # find dr7objids in test_subjects that match zooids in training_subjects
        test_idx, train_idx = [], []
        for i,num in enumerate(test_subjects['dr7objid']):
            if num not in training_subjects['zooid']:
                test_idx.append(i)

        test_sample_final = test_sample[test_idx]
        print "Test sample consists of %i subjects"%len(test_sample_final)

        # save the abbreviated test sample for processing later with 
        # explore_machine.py
        outfile = '%s%s_machine_testsample.fits'%(directory,trunk)
        thingy = test_subjects[test_idx]
        Table.write(thingy, outfile, format='fits', overwrite=True)

    """
