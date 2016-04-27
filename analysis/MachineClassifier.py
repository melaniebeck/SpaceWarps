from sklearn.cross_validation import KFold
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier as KNC

import swap
import machine_utils as ml
import metrics as mtrx

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

def MachineClassifier(options, args):

    try: config = options.configfile
    except: pdb.set_trace()

    tonights = swap.Configuration(config)

    #"""
    # Read the pickled random state file
    random_file = open(tonights.parameters['random_file'],"r");
    random_state = cPickle.load(random_file);
    random_file.close();
    np.random.set_state(random_state);
    #"""

    # Get the machine threshold (make retirement decisions)
    threshold = tonights.parameters['machine_threshold']

    # Get list of evaluation metrics and criteria   
    eval_metrics = tonights.parameters['evaluation_metrics']

    survey = tonights.parameters['survey']
    subdir = 'sup_run4'

    #----------------------------------------------------------------------
    # read in the metadata for all subjects (Test or Training sample?)
    subjects = swap.read_pickle(tonights.parameters['metadatafile'], 'metadata')

    #----------------------------------------------------------------------
    # read in the SWAP collection
    sample = swap.read_pickle(tonights.parameters['samplefile'],'collection')

    #----------------------------------------------------------------------
    # read in or create the ML collection
    MLsample = swap.read_pickle(tonights.parameters['MLsamplefile'],
                                'MLcollection')

    # read in or create the ML bureau for machine agents (history)
    MLbureau = swap.read_pickle(tonights.parameters['MLbureaufile'], 'MLbureau')

    #-----------------------------------------------------------------------    
    #        DETERMINE IF THERE IS A TRAINING SAMPLE TO WORK WITH 
    #-----------------------------------------------------------------------
    # TO DO: training sample should only select those which are NOT part of 
    # validation sample (Nair catalog objects) 2/22/16

    # IDENTIFY TRAINING SAMPLE
    train_sample = subjects[subjects['MLsample']=='train']
    train_meta, train_features = ml.extract_training(train_sample)
    train_labels = np.array([1 if p > 0.3 else 0 \
                             for p in train_meta['SWAP_prob']])

    # IDENTIFY VALIDATION SAMPLE (FINAL) 
    valid_sample = subjects[subjects['MLsample']=='valid']
    valid_meta, valid_features = ml.extract_training(valid_sample)
    valid_labels = valid_meta['Expert_label'].filled()

    #if len(train_sample) >= 100: 
    # TO DO: LOOP THROUGH DIFFERENT MACHINES? HOW MANY MACHINES?
    for metric in eval_metrics:
        
        # REGISTER Machine Classifier
        # Construct machine name --> Machine+Metric? For now: KNC
        machine = 'KNC'
        Name = machine+'_'+metric
        
        # register an Agent for this Machine
        try: 
            test = MLbureau.member[Name]
        except: 
            MLbureau.member[Name] = swap.Agent_ML(Name, metric)
            

        #---------------------------------------------------------------    
        #     TRAIN THE MACHINE; EVALUATE ON VALIDATION SAMPLE
        #---------------------------------------------------------------        

        # Now we run the machine -- need cross validation on whatever size 
        # training sample we have .. 
        
        # For now this will be fixed until we build in other machine options
        params = {'n_neighbors':np.arange(1, 2*(len(train_sample)-1) / 3, 2), 
                  'weights':('uniform','distance')}
        
        # Create the model 
        general_model = GridSearchCV(estimator=KNC(), param_grid=params,
                                     error_score=0, scoring=metric)        

        # Train the model -- k-fold cross validation is embedded
        trained_model = general_model.fit(train_features, train_labels)

        # Test "accuracy" (metric of choice) on validation sample
        score = trained_model.score(valid_features, valid_labels)

        """
        MLbureau.member[Name].record_training(\
                            model_described_by=trained_model.best_estimator_, 
                            with_params=trained_model.best_params_, 
                            trained_on=len(train_features), 
                            at_time=TIME, 
                            with_train_acc=traineed_model.best_score_,
                            and_valid_acc=trained_model.score(valid_features,
                                                              valid_labels))
        """
        # Store the trained machine
        MLbureau.member[Name].model = trained_model

        
        # Compute / store confusion matrix as a function of threshold
        # produced by this machine on the Expert Validation sample

        fps, tps, thresh = mtrx._binary_clf_curve(valid_labels,
                            trained_model.predict_proba(valid_features)[:,1])
        metric_list = mtrx.compute_binary_metrics(fps, tps)
        ACC, TPR, FPR, FNR, TNR, PPV, FDR, FOR, NPV = metric_list
        
        MLbureau.member[Name].record_evaluation(accuracy=ACC, 
                                                completeness_s=TPR,
                                                contamination_s=FDR,
                                                completeness_f=TNR,
                                                contamination_f=NPV)

        pdb.set_trace()

        

        
        # 3. compare the metric of choice with the evaluation criterion to
        # see if this machine has sufficiently learned? 
        # ... what if my criterion is simply "Maximize Accuracy"? 
        # ... or minimize feature contamination? these require that we 
        # compare tonight's machine with the previous night's machine 
        # But if my criterion is simply "have feature contam less than 20%"
        # then it's easy.... 
        
        # IF TRAINED MACHINE PREDICTS WELL ON VALIDATION .... 
        if MLbureau.member[Name].evaluate():
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

