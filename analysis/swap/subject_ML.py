# ======================================================================

import swap

import numpy as np
import pylab as plt
import pdb

# Every subject starts with the following probability of being a LENS:
#prior = 2e-4

# Every subject starts 50 trajectories. This will slow down the code,
# but its ok, we can always parallelize
Ntrajectory=50
# This should really be a user-supplied constant, in the configuration.

# ======================================================================

class Subject_ML(object):
    """
    NAME
        Subject_ML

    PURPOSE
        Model an individual Space Warps subject and track whether this 
        subject will TRAIN the Machine or be part of the TEST sample

    COMMENTS
        Each subject knows whether it is a test or training subject.
        A subject is converted from Training to Test if it is classified 
        by users and f it's probability (given by SWAP) crosses either the
        detection or rejection threshold. 

        If the Machine then classifies the object over a given Machine 
        Threshold, then the state of the subject will be converted switched
        to Inactive. Annotationhistory keeps track of how the Machine 
        classifies the subject each time. 
       
        Subject sample: 
          * train    Will be used to train the Machine (if MORPH exists)
          * test     Trained Machine will be applied to this subject

        Subject state:
          * active    Still being classified
          * inactive  No longer being classified
        Training subjects are always active. Retired = inactive.

        Subject status:
          * MLdetected   Machine P > MLthreshold
          * undecided otherwise

        CPD 23 June 2014:
        Each subject also has an annotationhistory, which keeps track of who
        clicked, what they said it was, where they clicked, and their ability
        to tell a lens (PL) and a dud (PD)

    INITIALISATION
        ID

    METHODS

    AUTHORS
      This file is part of the Space Warps project, and is distributed
      under the MIT license by the Space Warps Science Team.
      http://spacewarps.org/

    trajectory
      2013-04-17  Started Marshall (Oxford)
      2013-05-15  Surhud More (KIPMU)
    """

# ----------------------------------------------------------------------

    def __init__(self,ID,ZooID,category,kind,truth,threshold,location):

        self.ID = ID
        self.ZooID = ZooID
        self.category = category
        self.kind = kind
        self.truth = truth

        self.state = 'active'
        self.status = 'undecided'

        self.retirement_time = 'not yet'
        self.retirement_age = 0.0
        self.retiredby = 'undecided'

        self.probability = 0.0
        self.exposure = 0
        
        self.machine_threshold = threshold

        self.location = location

        self.annotationhistory = {'Name': np.array([]),
                                  'ItWas': np.array([]), 
                                  'PL': np.array([]), 
                                  'At_Time': []}

        return None

# ----------------------------------------------------------------------

    def was_described(self,by=None,as_being=None,withp=None,at_time=None,
                      haste=True, record=True):

        if by==None or as_being==None:
            pass

        # Optional: skip straight past inactive subjects.
        elif haste and (     self.state == 'inactive' \
                         or self.status == 'detected' \
                         or self.status == 'rejected' ):
                pass

        else:
            # update the annotation history
            if record:
                as_being_dict = {'SMOOTH': 1, 'NOT': 0}
                self.annotationhistory['Name'] = \
                            np.append(self.annotationhistory['Name'], by)
                self.annotationhistory['ItWas'] = \
                            np.append(self.annotationhistory['ItWas'],as_being)
                self.annotationhistory['PL'] = \
                            np.append(self.annotationhistory['PL'], withp)
                self.annotationhistory['At_Time'].append(at_time)
                
            self.exposure += 1
                
            self.update_state(at_time)
                
            '''
            # Update agent (Machine) - training history is taken care of in 
            # agent.heard(), which also keeps agent.skill up to date.
            if self.kind == 'test' and record:
                
                by.testhistory['ID'] = np.append(by.testhistory['ID'], self.ID)
                by.testhistory['I'] = np.append(by.testhistory['I'], 
                                    swap.informationGain(self.mean_probability,
                                    by.PL, by.PD, as_being))
                by.testhistory['Skill'] = \
                                    np.append(by.testhistory['Skill'], by.skill)
                by.testhistory['ItWas'] = \
                                    np.append(by.testhistory['ItWas'],
                                    as_being_number)
                by.testhistory['At_Time'] = \
                                    np.append(by.testhistory['At_Time'], at_time)
                by.contribution += by.skill
                
                #else:
                # offline
                #return likelihood

            else:
                # Still advance exposure, even if by.NT <= ignore:
                # it would be incorrect to calculate mean classns/retirement
                # different from strict and alt-strict:
                self.exposure += 1
            '''

        return

    def update_state(self,at_time=None):
        if self.probability >= self.machine_threshold:
            self.status = 'detected'
            self.retiredby = 'machine'

            if self.kind == 'test':
                self.state = 'inactive'
            self.retirement_time = at_time
            self.retirement_age = self.exposure

        elif (1-self.probability) >= self.machine_threshold:
            self.status = 'rejected'
            self.retiredby = 'machine'

            if self.kind == 'test':
                self.state = 'inactive'
                if at_time: 
                    self.retirement_time = at_time
                else:
                    self.retiremenet_time = 'end of time'
                self.retirement_age = self.exposure

        else:
            self.status = 'undecided'
            self.state = 'active'
            self.retirement_time = 'not yet'
            self.retirement_age = 0.0
            self.retiredby = 'undecided'
