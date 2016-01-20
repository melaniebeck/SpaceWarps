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

    def __init__(self,ID,ZooID,MLsample,morph,threshold):

        self.ID = ID
        self.ZooID = ZooID

        self.MLsample = MLsample
        self.state = 'active'
        self.status = 'undecided'

        self.MLthresh = threshold
     
        if morph:
            self.G = morph['G']
            self.M20 = morph['M20']
            self.C = morph['C']
            self.A = morph['A']
            self.E = morph['E']
        else:
            self.G = self.M20 = self.C = self.A = self.E = 'nan'


        self.annotationhistory = {'ItWas': np.array([]), 
                                  'Psmooth': np.array([]), 
                                  'At_Time': []}

        return None

# ----------------------------------------------------------------------
