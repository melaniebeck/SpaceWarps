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
        A little robot who will interpret the classifications of an
        individual volunteer.

    COMMENTS
        An Agent is assigned to represent a MACHINE, whose Name is
        the Algorithm (KNC, RF, etc). Agents each have a History which 
        records the size of the training sample, N. 
        classified, and is equal to N in the simple "SMOOTH or NOT"
        analysis. Each Agent carries a "confusion matrix"
        parameterised by two numbers, PD and PL, the meaning of which is
        as follows:

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

    def __init__(self,name,metric,criterion):
        self.name = name
        self.model = None  # Can I actually store the trained machine??? :D
        self.eval_metric = metric
        self.eval_criterion = criterion

        self.evaluationhistory = {'N':np.array([]),
                                  'ACC':np.array([]),
                                  'TPR':np.array([]),
                                  'TNR':np.array([]),
                                  'SCONT':np.array([]),
                                  'FCONT':np.array([]),
                                  'At_Time': np.array([])}
        return None

    def record(self, training_sample_size=None, with_accuracy=None, 
               smooth_completeness=None, feature_completeness=None, 
               smooth_contamination=None, feature_contamination=None,
               at_time=None):

        # Always log on what are we trained, even if not learning:
        self.evaluationhistory['N']=np.append(self.evaluationhistory['N'], 
                                              training_sample_size)
        self.evaluationhistory['ACC']=np.append(self.evaluationhistory['ACC'],
                                                with_accuracy)
        self.evaluationhistory['TPR']=np.append(self.evaluationhistory['TPR'],
                                                smooth_completeness)
        self.evaluationhistory['TNR']=np.append(self.evaluationhistory['TNR'], 
                                                feature_completeness)
        self.evaluationhistory['CONT_S'] = \
                            np.append(self.evaluationhistory['CONT_S'], 
                                      smooth_contamination)
        self.evaluationhistory['CONT_F'] = \
                            np.append(self.evaluationhistory['CONT_F'], 
                                      feature_contamination)
        self.evaluationhistory['At_Time'] = \
                        np.append(self.evaluationhistory['At_Time'], at_time)
        self.evaluationhistory['Trained'] = np.append(self.evaluate())

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

    def plot_metrics(self):
        
        
