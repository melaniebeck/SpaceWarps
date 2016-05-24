# ======================================================================

import swap

import os,cPickle,numpy as np
import pdb, heapq
from collections import Counter

# ======================================================================

"""
    NAME
        io

    PURPOSE
        Useful general functions to streamline file input and output.

    COMMENTS

    FUNCTIONS
        write_pickle(contents,filename):

        read_pickle(filename):

        write_list(sample, filename, item=None):
        
        read_list(filename):
     
        write_catalog(sample, filename, thresholds, kind=kind):

        rm(filename):

        write_config():

        get_new_filename(parameters, flavour, source=None):

    BUGS

    AUTHORS
      This file is part of the Space Warps project, and is distributed
      under the MIT license by the Space Warps Science Team.
      http://spacewarps.org/

      SWAP io is modelled on that written for the
      Pangloss project, by Tom Collett (IoA) and Phil Marshall (Oxford).
      https://github.com/drphilmarshall/Pangloss/blob/master/pangloss/io.py

    HISTORY
      2013-04-17  Started Marshall (Oxford)
"""

#=========================================================================
# Read in an instance of a class, of a given flavour. Create an instance
# if the file does not exist.

def read_pickle(filename,flavour):

    try:
        F = open(filename,"rb")
        contents = cPickle.load(F)
        F.close()

        print "SWAP: read an old contents from "+filename

    except:

        if filename is None:
            print "SWAP: no "+flavour+" filename supplied."
        else:
            print "SWAP: "+filename+" does not exist."

        if 'bureau' in flavour:
            contents = swap.Bureau()
            print "SWAP: made a new",contents

        elif 'collection' in flavour:
            contents = swap.Collection()
            print "SWAP: made a new",contents

        elif 'database' in flavour:
            contents = None

        elif 'metadata' in flavour:
            contents = swap.StorageLocker()
            print "SWAP: generated a new", contents

    return contents

# ----------------------------------------------------------------------------
# Write out an instance of a class to file.

def write_pickle(contents,filename):

    F = open(filename,"wb")
    cPickle.dump(contents,F,protocol=2)
    F.close()

    return

# ----------------------------------------------------------------------------
# Write out a simple list of subject IDs, of subjects to be retired.

def write_list(sample, filename, item=None, source=None):

    count = 0
    F = open(filename,'w')
    for ID in sample.list():
        subject = sample.member[ID]
        string = None

        if item == 'retired_subject':
            if subject.state == 'inactive':
                string = subject.ZooID

        elif item == 'detected':
            if subject.kind == 'test' and subject.status == 'detected':
                string = subject.location

        elif item == 'rejected':
            if subject.kind == 'test' and subject.status == 'rejected':
                string = subject.location

        elif item == 'true_positive':
            if subject.kind == 'sim' and subject.status == 'detected':
                string = subject.location

        elif item == 'false_positive':
            if subject.kind == 'dud' and subject.status == 'detected':
                string = subject.location

        elif item == 'true_negative':
            if subject.kind == 'dud' and subject.status == 'rejected':
                string = subject.location

        elif item == 'false_negative':
            if subject.kind == 'sim' and subject.status == 'rejected':
                string = subject.location

        # Write a new line:
        if item is not None and string is not None:
            F.write('%s\n' % string)
            count += 1

    F.close()

    return count

# ----------------------------------------------------------------------------
# Read in a simple list of string items.

def read_list(filename):
    return np.atleast_1d(np.loadtxt(filename,dtype='string'))

# ----------------------------------------------------------------------------
# Write out a multi-column catalog of high probability candidates.

def write_catalog(sample,filename,thresholds,kind='test',source=None):

    Nsubset, Nsubjects = 0, 0

    # Open a new catalog and write a header:
    F = open(filename,'w')
    F.write('%s\n' % "# zooid     P         Nclass  image")

    for ID in sample.list():
        subject = sample.member[ID]
        P = subject.mean_probability

        """
        The first two take care of when kind == retired, rejected, detected
        """
        # Perhaps you want a catalog of ALL rejected & detected subjects...
        if kind == 'retired':
            if subject.status != 'undecided':
                output = (subject.ZooID, P, subject.exposure, subject.location)
                F.write('%s  %9.7f  %s       %s\n'%output)
                Nsubset += 1
                
        # Perhaps you want ONLY the rejected or detected objects... 
        elif subject.status == kind:
            output = (subject.ZooID, P, subject.exposure, subject.location)
            F.write('%s  %9.7f  %s       %s\n'%output)           
            Nsubset += 1
            
        # Perhaps you want the possible candidates (P > rejected)
        else:
            if P > thresholds['rejection'] and subject.kind == kind:
                output = (subject.ZooID, P, subject.exposure, subject.location)
                F.write('%s  %9.7f  %s       %s\n'%output)
                Nsubset += 1

        Nsubjects += 1

    F.close()

    return Nsubset,Nsubjects

# ----------------------------------------------------------------------------

def get_new_filename(pars,flavour,source=None):

    # Usually, this is what we want filenames to look like:
    stem = pars['trunk']+'_'+flavour
    folder = pars['dir']
    ext = 'txt'

    # Pickle filenames
    if flavour in ['bureau', 'collection', 'MLcollection', 'metadata', 
                   'database', 'offline']:
        if source == 'ML':   stem = pars['survey']+'_ML'+flavour
        else: stem = pars['survey']+'_'+flavour
        ext = 'pickle'
        folder = '.'
    
    # image filenames
    elif flavour in ['histories', 'trajectories', 'sample', 'probabilities']:
        ext = 'png'

    # catalog filenames
    elif flavour in ['retire_these', 'detected', 'rejected', 
                     'training_true_positives', 'training_false_positives', 
                     'training_true_negatives', 'training_false_negatives', 
                     'sim_catalog', 'dud_catalog', 'retired_catalog', 
                     'detected_catalog', 'rejected_catalog', 
                     'candidate_catalog']:
        ext = 'txt'
        folder = pars['dir']
        
    else:
        raise Exception("SWAP: io: unknown flavour "+flavour)
      
        
    return folder+'/'+stem+'.'+ext

# ----------------------------------------------------------------------------
# Write configuration file given a dictionary of parameters:

def write_config(filename, pars):

    F = open(filename,'w')

    header = """
# ======================================================================
#
# Space Warps Analysis Pipeline configuration file.
#
# Lines starting with '#' are ignored; all other lines must contain a
# Name : Value pair to be read into the parameters dictionary.
#
# This file is part of the Space Warps project, and is distributed
# under the MIT license by the Space Warps Science Team.
# http://spacewarps.org/
#
# SWAP configuration is modelled on that written for the
# Pangloss project, by Tom Collett (IoA) and Phil Marshall (Oxford).
# https://github.com/drphilmarshall/Pangloss/blob/master/example/example.config
#
# ======================================================================
    """
    F.write(header)

    for k,v in pars.iteritems():
        F.write('\n')
        F.write('%s: %s\n'%(k,v))


    F.write('\n')
    footer = '# ======================================================================'
    F.write(footer)

    F.close()

    return

# ----------------------------------------------------------------------------
# Remove file, if it exists, stay quiet otherwise:

def rm(filename):
    try:
        os.remove(filename)
    except OSError:
        pass
    return

# ======================================================================
