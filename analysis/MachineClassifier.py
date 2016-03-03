
import swap
import machine as ml
from optparse import OptionParser
from astropy.table import Table
import pdb
import datetime
import numpy as np
import os, subprocess
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

    # read in or create the ML bureau for machine agents (history)
    MLbureau = swap.read_pickle(tonights.parameters['MLbureaufile'], 'MLbureau')

    #-----------------------------------------------------------------------    
    #        DETERMINE IF THERE IS A TRAINING SAMPLE TO WORK WITH 
    #-----------------------------------------------------------------------
    # TO DO: training sample should only select those which are NOT part of 
    # validation sample (Nair catalog objects) 2/22/16

    train_sample = subjects[subjects['MLsample']=='train']
    train_meta, train_features = ml.extract_training(train_sample)
    train_labels = np.array([1 if p > 0.3 else 0 for p in train_meta['label']])

    # Validation sample should remain constant 
    valid_sample = subjects[subjects['MLsample']=='valid']
    valid_meta, valid_features = ml.extract_training(valid_sample)
    valid_labels = []

    if len(train) > 10:
        test_sample = subjects[subjects['MLsample']=='test']
        test_meta, test_features = ml.extract_training(test_sample)
        
        # loop through different machines? 
        # Machine Name based on Metric Evaluation + Machine Algorithm? 
        # MAKE EVALUATION METRIC/CRITERION ARRAYS???? 
        # THEN WE CAN LOOP THROUGH THEM? 

        pdb.set_trace()
        # register an Agent for this Machine
        try: test = MLbureau.member[Name]
        except: MLbureau.member[Name] = swap.Agent_ML(Name, tonights.parameters)
        ##### ADD EVALUATION METRIC AND EVAL CRITERION TO TONIGHTS.PARAMS ####

        #---------------------------------------------------------------    
        #     TRAIN THE MACHINE; GET PREDICTIONS ON VALIDATION SAMPLE
        #---------------------------------------------------------------        
        predictions, probas, model = ml.runKNC(train_features, train_labels, 
                                               valid_features)

        # DETERMINE IF MACHINE TRAINS AT/ABOVE ==>CONDITION<== ??
        # 1. use predictions/probas to compute various metrics
        fps, tps, thresh = metrics._binary_clf_curve(truth, probas[:,1])

        # Should this be a function in the agent_ML.py object? 
        metrics = metrics.compute_binary_metrics(fps, tps)
        ACC, TPR, FPR, FNR, TNR, PPV, FDR, FOR, NPV = metrics

        # 2. record those metrics for each night (and each machine)
        #### GET DATE FROM TONIGHTS.PARAMETERS? #####
        MLbureau.member[Name].record(training_sample_size=len(train), 
                                     with_accuracy=ACC,
                                     smooth_completeness=TPR,
                                     feature_completeness=TNR, 
                                     smooth_contamination=PPV, 
                                     feature_contamination=FOR, 
                                     at_time=None)
        
        # 3. compare the metric of choice with the evaluation criterion to
        # see if this machine has sufficiently learned? 
        # ... what if my criterion is simply "Maximize Accuracy"? 
        # ... or minimize feature contamination? these require that we 
        # compare tonight's machine with the previous night's machine 
        # But if my criterion is simply "have feature contam less than 20%"
        # then it's easy.... 
        metrics = ml.analyse_performance(predictions, probas)     

        # 4. If the machine is sufficiently trained, send it on!
        if metrics > threshold: 
            condition = True

        pdb.set_trace()
        
        # IF TRAINED MACHINE PREDICTS WELL ON VALIDATION .... 
        if condition:
            #---------------------------------------------------------------    
            #                 APPLY MACHINE TO TEST SAMPLE
            #--------------------------------------------------------------- 
            # This requires that my runKNC function returns the Machine Object
            shitski=5
      
            #---------------------------------------------------------------    
            #                    PROCESS PREDICTIONS/PROBS
            #---------------------------------------------------------------
            for s,p,l in zip(test_meta, probas, predictions):
                ID = str(s['id'])

                descriptions = Nair_or_Not(s)
                category, kind, flavor, truth = descriptions

                # LOAD EACH TEST SUBJECT INTO MACHINE COLLECTION
                # -------------------------------------------------------------
                try: 
                    test = MLsample.member[ID]
                except: MLsample.member[ID] = swap.Subject_ML(ID,
                                            str(s['name']), category, kind,
                                            truth,threshold,s['external_ref'])
                
                tstring = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
                MLsample.member[ID].was_described(by='knn', as_being=1, 
                                                  withp=p, at_time=tstring)

                # NOTE: if subject is Nair (training) it doesn't get flagged as 
                # inactive but it can be flagged as detected/rejected


                # IF MACHINE P >= THRESHOLD, INSERT INTO SWAP COLLECTION
                # -------------------------------------------------------------
                thresholds = {'detection':0.,'rejection':0.}
                if (p >= threshold) or (1-p >= threshold):
                    print "BOOM! WE'VE GOT A MACHINE-CLASSIFIED SUBJECT:"
                    print "Probability:",p
                    # Initialize the subject in SWAP Collection
                    sample.member[ID] = swap.Subject(ID, str(s['name']), 
                                            category, kind,flavor,truth,
                                            thresholds, s['external_ref'],0.) 
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
                # -------------------------------------------------------------
                catalog = swap.get_new_filename(tonights.parameters,
                                            'retired_catalog', source='ML')
                print "SWAP: saving catalog of Machine-retired subjects..."
                Nretired, Nsubjects = swap.write_catalog(MLsample,bureau,
                                                catalog, threshold,
                                                kind='rejected', source='ML')
                print "SWAP: From "+str(Nsubjects)+" subjects classified,"
                print "SWAP: "+str(Nretired)+" retired (with P < rejection) "\
                    "written to "+catalog
            
                catalog = swap.get_new_filename(tonights.parameters,
                                            'detected_catalog', source='ML')
                print "SWAP: saving catalog of Machine detected subjects..."
                Ndetected, Nsubjects = swap.write_catalog(MLsample, bureau,
                                                catalog, threshold, 
                                                kind='detected', source='ML')
                print "SWAP: From "+str(Nsubjects)+" subjects classified,"
                print "SWAP: %i detected (with P > MLthreshold) "\
                "written to %s"%(Ndetected, catalog)    


    

    # If is hasn't been done already, save the current directory
    # ---------------------------------------------------------------------
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

    pdb.set_trace()

if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", "--config", dest="configfile", 
                      help="Name of config file")
    parser.add_option("-o", "--offline", dest="offline", default=False,
                      action='store_true',
                      help="Run in offline mode; e.g. on existing SWAP output.")

    (options, args) = parser.parse_args()
    MachineClassifier(options, args)

