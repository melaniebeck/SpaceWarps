
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

def extract_training(data, keys=['M20', 'C', 'elipt', 'A', 'G']):
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

def Nair_or_Not(subject):
    """
    subject must be an entry from the metadata file 
    should have stucture like a dictionary
    """
    if subject['JID'][0]=='J': 
        category = 'training'
        if subject['TType'] <= -2 and subject['dist'] <= 2: 
            flavor = 'lensing cluster'  # so that I don't have to
            kind = 'sim'                # change stuff elsewhere...
            truth = 'SMOOTH'
        elif subject['TType'] >= 1 and subject['flag'] != 2: 
            kind = 'dud' 
            flavor = 'dud'
            truth = 'NOT'
        else:
            kind = 'test'
            flavor = 'test'
            truth = 'UNKNOWN'
    else: 
        category = 'test'
        kind = 'test'
        flavor = 'test'
        truth = 'UNKNOWN'

    descriptors = category, kind, flavor, truth
    return descriptors[:]

def MachineClassifier(options, args):

    threshold = 0.9

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

    #----------------------------------------------------------------------
    # read in the metadata for all subjects (Test or Training sample?)
    subjects = swap.read_pickle(tonights.parameters['metadatafile'], 'metadata')

    #----------------------------------------------------------------------
    # read in the SWAP collection
    sample = swap.read_pickle(tonights.parameters['samplefile'],'collection')

    #----------------------------------------------------------------------
    # read in or create the ML collection (Should be empty first time)
    MLsample = swap.read_pickle(tonights.parameters['MLsamplefile'],
                                'MLcollection')

    #-----------------------------------------------------------------------    
    #        DETERMINE IF THERE IS A TRAINING SAMPLE TO WORK WITH 
    #-----------------------------------------------------------------------
    train = subjects[subjects['MLsample']=='train']
    #train = subjects[:100]
    if train:
        test = subjects[subjects['MLsample']=='test']
        #test = subjects[100:200]
        train_data, train_sample = extract_training(train)
        test_data, test_sample = extract_training(test)
        labels = np.array([1 if p > 0.3 else 0 for p in train_data['MLsample']])

        #---------------------------------------------------------------    
        #                 TRAIN THE MACHINE; GET PREDICTIONS 
        #---------------------------------------------------------------        
        predictions, probabilities = runKNC(train_sample, labels, test_sample)
        probs = np.array([p[0] for p in probabilities])

        #---------------------------------------------------------------    
        #                    PROCESS PREDICTIONS/PROBS
        #---------------------------------------------------------------
        for s,p,l in zip(test_data,probs,predictions):
            ID = str(s['id'])

            descriptions = Nair_or_Not(s)
            category, kind, flavor, truth = descriptions

            # LOAD EACH TEST SUBJECT INTO MACHINE COLLECTION
            # -------------------------------------------------------------
            try: test = MLsample.member[ID]
            except: MLsample.member[ID] = swap.Subject_ML(ID, str(s['name']), 
                            category, kind,truth,threshold,s['external_ref'])
                
            tstring = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            MLsample.member[ID].was_described(by='knn', as_being=1, withp=p, 
                                              at_time=tstring)

            # NOTE: if subject is Nair (training) it doesn't get flagged as 
            # inactive but it can be flagged as detected/rejected

            # IF MACHINE P >= THRESHOLD, INSERT INTO SWAP COLLECTION
            # -------------------------------------------------------------
            thresholds = {'detection':0.,'rejection':0.}
            if (p >= threshold) or (1-p >= threshold):
                print "BOOM! WE'VE GOT A MACHINE-CLASSIFIED SUBJECT:"
                print "Probability:",p
                # Initialize the subject in SWAP Collection
                sample.member[ID] = swap.Subject(ID, str(s['name']), category,
                                                 kind,flavor,truth,thresholds,
                                                 s['external_ref'],0.) 
                sample.member[ID].retiredby = 'machine'
                
                # Flag subject as 'INACTIVE' / 'DETECTED' / 'REJECTED'
                # ----------------------------------------------------------
                if p >= threshold:
                    sample.member[str(s['id'])].state = 'inactive'
                elif 1-p >= threshold:
                    sample.member[str(s['id'])].status = 'rejected' 


        #---------------------------------------------------------------    
        #                 SAVE MACHINE METADATA? 
        #---------------------------------------------------------------
        print "Size of SWAP sample:", sample.size()
        print "Size of ML sample:", MLsample.size()

      
        if tonights.parameters['report']:
            
            # Output list of subjects to retire, based on this batch of
            # classifications. Note that what is needed here is the ZooID,
            # not the subject ID:
            
            new_retirementfile = swap.get_new_filename(tonights.parameters,\
                                                   'retire_these', source='ML')
            print "SWAP: saving Machine-retired subject Zooniverse IDs..."
            N = swap.write_list(MLsample,new_retirementfile,
                                item='retired_subject', source='ML')
            print "SWAP: "+str(N)+" lines written to "+new_retirementfile
            
            # write catalogs of smooth/not over MLthreshold
            # ---------------------------------------------------------------
            catalog = swap.get_new_filename(tonights.parameters,
                                            'retired_catalog', source='ML')
            print "SWAP: saving catalog of Machine-retired subjects..."
            Nretired, Nsubjects = swap.write_catalog(MLsample,bureau,catalog,
                                        threshold,kind='rejected', source='ML')
            print "SWAP: From "+str(Nsubjects)+" subjects classified,"
            print "SWAP: "+str(Nretired)+" retired (with P < rejection) "\
                "written to "+catalog
            
            catalog = swap.get_new_filename(tonights.parameters,
                                            'detected_catalog', source='ML')
            print "SWAP: saving catalog of Machine detected subjects..."
            Ndetected, Nsubjects = swap.write_catalog(MLsample, bureau,catalog,
                                        threshold, kind='detected', source='ML')
            print "SWAP: From "+str(Nsubjects)+" subjects classified,"
            print "SWAP: "+str(Ndetected)+" detected (with P > MLthreshold) "\
                "written to "+catalog        

    # If is hasn't been done already, save the current directory
    # -----------------------------------------------------------------------
    tonights.parameters['dir'] = os.getcwd()+'/'+tonights.parameters['trunk']
    
    if not os.path.exists(tonights.parameters['dir']):
        os.makedirs(tonights.parameters['dir'])


    # Repickle all the shits
    # -----------------------------------------------------------------------
    if tonights.parameters['repickle']:

        new_samplefile = swap.get_new_filename(tonights.parameters,'collection')
        print "SWAP: saving SWAP subjects to "+new_samplefile
        swap.write_pickle(sample,new_samplefile)
        tonights.parameters['samplefile'] = new_samplefile
        
        new_samplefile=swap.get_new_filename(tonights.parameters,'MLcollection')
        print "SWAP: saving test sample subjects to "+new_samplefile
        swap.write_pickle(MLsample,new_samplefile)
        tonights.parameters['MLsamplefile'] = new_samplefile

        metadatafile = swap.get_new_filename(tonights.parameters,'metadata')
        print "SWAP: saving metadata to "+metadatafile
        swap.write_pickle(subjects,metadatafile)
        tonights.parameters['metadatafile'] = metadatafile
       

    # Update the time increment for SWAP's next run
    # -----------------------------------------------------------------------
    t2 = datetime.datetime.strptime(tonights.parameters['start'],
                                    '%Y-%m-%d_%H:%M:%S') + \
         datetime.timedelta(days=tonights.parameters['increment'])
    tstop = datetime.datetime.strptime(tonights.parameters['end'],
                                    '%Y-%m-%d_%H:%M:%S')
    if t2 == tstop: 
        plots = True
    else:
        tonights.parameters['start'] = t2.strftime('%Y-%m-%d_%H:%M:%S')
                

    # Update configfile to reflect Machine additions
    # -----------------------------------------------------------------------
    configfile = 'update.config'

    random_file = open(tonights.parameters['random_file'],"w");
    random_state = np.random.get_state();
    cPickle.dump(random_state,random_file);
    random_file.close();
    swap.write_config(configfile, tonights.parameters)

    #pdb.set_trace()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", "--config", dest="configfile", 
                      help="Name of config file")
    parser.add_option("-o", "--offline", dest="offline", default=False,
                      action='store_true',
                      help="Run in offline mode; e.g. on existing SWAP output.")

    (options, args) = parser.parse_args()
    MachineClassifier(options, args)

