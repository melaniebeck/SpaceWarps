# ======================================================================

import swap

import numpy as np
import pylab as plt
import pdb

actually_it_was_dictionary = {'SMOOTH': 1, 'NOT': 0, 'UNKNOWN': -1}

# ======================================================================

class Agent_ML(object):
    """
    NAME
        Agent

    PURPOSE
        A little robot who will interpret the training  of an
        individual Machine.

    COMMENTS
        An Agent is assigned to represent a MACHINE, whose Name is
        the Algorithm (KNC, RF, etc) + the Evaluation Metric of user's choice.
        Agents each have a History which 
        records the size of the training sample, N. 

        History also records various metrics for Machine evaluation: 
        -- list of metrics? should these be 

    INITIALISATION
        name

    METHODS
        Agent.update_contribution()  Calculate the expected
                                          information contributed
                                          per classification
        Agent.heard(it_was=X,actually_it_was=Y)     Read report.
        Agent.plot_history(axes)

    BUGS

    AUTHORS
      This file is part of the Space Warps project, and is distributed
      under the MIT license by the Space Warps Science Team.
      http://spacewarps.org/

    HISTORY
      2013-04-17:  Started Marshall (Oxford)
      2015-01-19:  Added 'kind'. (CPD)
    """

# ----------------------------------------------------------------------

    def __init__(self, name, metric):
        self.name = name
        self.model = None  # Can I actually store the trained machine??? :D

        if metric in ['accuracy', 'precision', 'recall']:
            self.eval_metric = metric
        else: 
            print "%s is not a valid scoring metric."%metric
            exit

        
        # Track the evolution of the model and the best parameters found
        # during a Grid Search over the parameter space with CV
        self.traininghistory = {'Model':np.array([]), 
                                'Parameters':np.array([]), 
                                'TrainingSize':np.array([]), 
                                'ClassRatio':np.array([]),
                                'At_Time':np.array([]),
                                'TrainingScore':np.array([]),
                                'ValidScore':np.array([])}


        # Record various metrics measured on the validation sample
        # as a function of a threshold (for ROC curve plotting, etc.)
        self.validationhistory = {'accuracy':[],
                                  'precision':[],
                                  'recall':[],
                                  'false_pos':[],
                                  'Completeness(F)':[],
                                  'Contamination(F)':[]}


        # FOR TESTING ONLY -- REMOVE THIS BEFORE APPLYING TO ANOTHER DATASET
        # Once the machine has learned, apply it to the test sample for 
        # overall method evaluation
        self.evaluationhistory = {'accuracy_score':[], 
                                  'precision_score':[],
                                  'recall_score':[], 
                                  'at_time':[]}


        return None


    def record_training(self, model_described_by=None, with_params=None, 
                        trained_on=None, with_ratio=None, at_time=None, 
                        with_train_score=None, and_valid_score=None):

        # Need to record the outcome of the cross-validation (which params 
        # were used each time) but this is dependent on the algorithm
        self.traininghistory['Model'] = \
                    np.append(self.traininghistory['Model'], model_described_by)
        self.traininghistory['Parameters'] = \
                    np.append(self.traininghistory['Parameters'], with_params)
        self.traininghistory['TrainingSize'] = \
                    np.append(self.traininghistory['TrainingSize'], trained_on)
        self.traininghistory['ClassRatio'] = \
                    np.append(self.traininghistory['ClassRatio'], with_ratio)
        self.traininghistory['At_Time'] = \
                    np.append(self.traininghistory['At_Time'], at_time)
        self.traininghistory['TrainingScore'] = \
            np.append(self.traininghistory['TrainingScore'], with_train_score)
        self.traininghistory['ValidScore'] = \
            np.append(self.traininghistory['ValidScore'], and_valid_score)

        return

    def record_validation(self, accuracy=None, recall=None, precision=None, 
                          false_pos=None, completeness_f=None, 
                          contamination_f=None):

        self.validationhistory['accuracy'].append(accuracy)
        self.validationhistory['recall'].append(recall)
        self.validationhistory['precision'].append(precision)
        self.validationhistory['false_pos'].append(false_pos)
        self.validationhistory['Completeness(F)'].append(completeness_f)
        self.validationhistory['Contamination(F)'].append(contamination_f)

        return

    def record_evaluation(self,accuracy_score=None,precision_score=None,
                          recall_score=None,at_time=None):

        self.evaluationhistory['accuracy_score'].append(accuracy_score)
        self.evaluationhistory['precision_score'].append(precision_score)
        self.evaluationhistory['recall_score'].append(recall_score)
        self.evaluationhistory['at_time'].append(at_time)
        
        return 

    def is_trained(self, metric):
        # LEARNING CURVE -- find where accuracy plateaus
        # Define "plateau" -- 3 nights in a row with less than XXX difference?

        # if the differences in the metric for the past three nights are all
        # less than a percent -- WE'VE REACHED A PLATEAU

        if len(self.traininghistory['ValidScore']) >= 4:

            differences = np.diff(self.traininghistory['ValidScore'])

            if np.all(differences[-3:]<.01):
                return True
        
        return False


    def plot_learning_curve(self):
        self

    def plot_ROC(self):
        plt.figure(figsize=(10,10))
        plt.plot(self.evaluationhistory['false_pos'][-1], 
                 self.evaluationhistory['recall'][-1], color='purple')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate (Recall)')
        plt.show()
        plt.close()

        
        
