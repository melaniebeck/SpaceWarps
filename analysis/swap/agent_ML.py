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

        # This might eventually be removed but for now I think it'll be useful
        # This will store the 'metric' calculated on the Expert Sample as a
        # function of the 'threshold', which in our case is the probability
        # of being "smooth" (having the characteristic of choice)
        
        # KEY: Astronomer's Term == Statistician's Term
        # Completeness(S) == Recall (True Positive Rate)
        # Contamination(S) == 1 - Precision (False Discovery Rate)
        # Completeness(F) == Specificity (True Negative Rate)
        # Contamination(F) == False Omission Rate 
        self.evaluationhistory = {'Accuracy':np.array([]),
                                  'Completeness(S)':np.array([]),
                                  'Contamination(S)':np.array([]),
                                  'Completeness(F)':np.array([]),
                                  'Contamination(F)':np.array([])}

        # This is SUPER necessary -- 
        # Track the evolution of the model and the best parameters found
        # during a Grid Search over the parameter space with CV
        self.traininghistory = {'Model':np.array([]), 
                                'Parameters':np.array([]), 
                                'TrainingSize':np.array([]), 
                                'At_Time':np.array([]),
                                'TrainACC':np.array([]),
                                'ValidACC':np.array([])}

        # Based on some combination of the above we can eventually determine
        # appropriate "scores" to evaluate the efficacy of our trained machine

        return None


    def record_training(self, model_described_by=None, with_params=None, 
                        trained_on=None, at_time=None, with_train_acc=None,
                        and_valid_acc=None):
        # Need to record the outcome of the cross-validation (which params 
        # were used each time) but this is dependent on the algorithm
        self.traininghistory['Model'] = \
                    np.append(self.traininghistory['Model'], model_described_by)
        self.traininghistory['Parameters'] = \
                    np.append(self.traininghistory['Parameters'], with_params)
        self.traininghistory['TrainingSize'] = \
                    np.append(self.traininghistory['TrainingSize'], trained_on)
        self.traininghistory['At_Time'] = \
                    np.append(self.traininghistory['At_Time'], at_time)
        self.traininghistory['TrainACC'] = \
                    np.append(self.traininghistory['TrainACC'], with_train_acc)
        self.traininghistory['ValidACC'] = \
                    np.append(self.traininghistory['ValidACC'], with_valid_acc)

        return

    def record_evaluation(self, accuracy=None, completeness_s=None, 
                          contamination_s=None, completeness_f=None, 
                          contamination_f=None):

        self.evaluationhistory['Accuracy'] = \
                    np.append(self.evaluationhistory['Accuracy'], accuracy)

        self.evaluationhistory['Completeness(S)'] = \
                    np.append(self.evaluationhistory['Completeness(S)'], 
                              completeness_s)

        self.evaluationhistory['Contamination(S)'] = \
                    np.append(self.evaluationhistory['Contamination(S)'], 
                              contamination_s)

        self.evaluationhistory['Completeness(F)'] = \
                    np.append(self.evaluationhistory['Completeness(F)'], 
                              completeness_f)

        self.evaluationhistory['Contamination(F)'] = \
                    np.append(self.evaluationhistory['Contamination(F)'], 
                              contamination_f)
        return

    def evaluate(self):
        # Can we use this to evaluate the metric? 
        # which metric is it? Metric is a string? 
        # For now let's make this simple -- Let's say we require that 
        # feature contamination be below 20%
        # To test for this, I need to know which metric(s), and criteria
        # For now, the only metrics you can choose are ACC, TPR, TNR, CONT_S, 
        # or CONT_F

        # This evaluation based on the fact that no one would want to 
        # minimize the true positive rate or maximize the contamination...
        if self.eval_metric in ['TPR', 'TNR', 'ACC']:
            if self.evaluationhistory[self.eval_metric] > self.eval_criterion:
                return True
            else: return False

        elif self.eval_metric in ['CONT_S', 'CONT_F']:
            if self.evaluationhistory[self.eval_metric] < self.eval_criterion:
                return True
            else: return False

        else:
            print "Not a valid metric."
            return False

    #def plot_metrics(self):
        
        
