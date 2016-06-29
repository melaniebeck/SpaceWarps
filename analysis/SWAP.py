#!/usr/bin/env python
# ======================================================================

import swap

import sys,getopt,datetime,os,subprocess
import numpy as np
import cPickle
import pdb
from astropy.table import Table

# ======================================================================

def SWAP(argv):
    """
    NAME
        SWAP.py

    PURPOSE
        Space Warps Analysis Pipeline

        Read in a Space Warps classification database from a MongoDB
        database, and analyse it.

    COMMENTS
        The SW analysis is "online" in the statistical sense: we step
        through the classifications one by one, updating each
        classifier's agent's confusion matrix, and each subject's lens
        probability. The main reason for taking this approach is that
        it is the most logical one; secondarily, it opens up the
        possibility of performing the analysis in real time (and maybe even
        with this piece of python).

        Currently, the agents' confusion matrices only depend on the
        classifications of training subjects. Upgrading this would be a
        nice piece of further work. Likewise, neither the Marker
        positions, the classification  durations, nor any other
        parameters are used in estimating lens probability - but they
        could be. In this version, it's LENS or NOT.

        Standard operation is to update the candidate list by making a
        new, timestamped catalog of candidates - and the classifications
        that led to them. This means we have to know when the last
        update was made - this is done by SWAP writing its own next
        config file, and by reading in a pickle of the last
        classification to be SWAPped. The bureau has to always be read
        in in its entirety, because a classifier can reappear any time
        to  have their agent update its confusion matrix.

    FLAGS
        -h            Print this message

    INPUTS
        configfile    Plain text file containing SW experiment configuration

    OUTPUTS
        stdout
        *_bureau.pickle
        *_collection.pickle

    EXAMPLE

        cd workspace
        SWAP.py startup.config > CFHTLS-beta-day01.log

    BUGS

    AUTHORS
      This file is part of the Space Warps project, and is distributed
      under the MIT license by the Space Warps Science Team.
      http://spacewarps.org/

    HISTORY
      2013-04-03  started. Marshall (Oxford)
      2013-04-17  implemented v1 "LENS or NOT" analysis. Marshall (Oxford)
      2013-05-..  "fuzzy" trajectories. S. More (IPMU)
    """

    # ------------------------------------------------------------------

    try:
       opts, args = getopt.getopt(argv,"h",["help"])
    except getopt.GetoptError, err:
       print str(err) # will print something like "option -a not recognized"
       print SWAP.__doc__  # will print the big comment above.
       return
    for o,a in opts:
       if o in ("-h", "--help"):
          print SWAP.__doc__
          return
       else:
          assert False, "unhandled option"

    # Check for setup file in array args:
    if len(args) == 1:
        configfile = args[0]
        print swap.doubledashedline
        print swap.SW_hello
        print swap.doubledashedline
        print "SWAP: taking instructions from",configfile

    else:
        print SWAP.__doc__
        return

    # ------------------------------------------------------------------
    # Read in run configuration:

    tonights = swap.Configuration(configfile)

    # Read the pickled random state file if it exists, else create it
    try:
        random_file = open(tonights.parameters['random_file'],"r");
        random_state = cPickle.load(random_file);

    except:
        random_file = open(tonights.parameters['random_file'],"w");
        random_state = np.random.get_state();

    random_file.close();
    np.random.set_state(random_state);

    practise = (tonights.parameters['dbspecies'] == 'Toy')
    if practise:
        print "SWAP: doing a dry run using a Toy database"
    else:
        print "SWAP: data will be read from the current live Mongo database"

    stage = str(tonights.parameters['stage'])
    survey = tonights.parameters['survey']
    print "SWAP: looks like we are on Stage "+stage+" of the ",\
        survey," survey project"

    try: supervised = tonights.parameters['supervised']
    except: supervised = False
    try: supervised_and_unsupervised = \
       tonights.parameters['supervised_and_unsupervised']
    except: 
        supervised_and_unsupervised = False
        tonights.parameters['supervised_and_unsupervised']=False

    # will agents be able to learn?
    try: agents_willing_to_learn =tonights.parameters['agents_willing_to_learn']
    except: agents_willing_to_learn = False
    if agents_willing_to_learn:

        if supervised_and_unsupervised:
            print "SWAP: agents will use both training AND test data to "\
                "update their confusion matrices"
        elif supervised:
            print "SWAP: agents will use training data to update their "\
                "confusion matrices"
        else:
            print "SWAP: agents will only use test data to update their "\
                "confusion matrices"

        a_few_at_the_start = tonights.parameters['a_few_at_the_start']
        if a_few_at_the_start > 0:
            print "SWAP: but at first they'll ignore their volunteer until "
            print "SWAP: they've done ",int(a_few_at_the_start)," images"

    else:
        a_few_at_the_start = 0
        print "SWAP: agents will use fixed confusion matrices without "\
            "updating them"


    waste = tonights.parameters['hasty']
    if waste:
        print "SWAP: agents will ignore the classifications of "\
            "rejected and detected subjects"
    else:
        print "SWAP: agents will use all classifications, even of "\
            "rejected subjects"


    vb = tonights.parameters['verbose']
    if not vb: print "SWAP: only reporting minimal stdout"

    one_by_one = tonights.parameters['one_by_one']

    report = tonights.parameters['report']
    plots = tonights.parameters['plot']
    if report:
        print "SWAP: will write a report at the end"
    else:
        print "SWAP: postponing reporting until the last minute"

    # From when shall we take classifications to analyze?
    if tonights.parameters['start'] == 'the_beginning':
        t1 = datetime.datetime(1978, 2, 28, 12, 0, 0, 0)
    elif (tonights.parameters['start'] == 'dont_bother') or \
         (tonights.parameters['start'] == tonights.parameters['end']):
        print "SWAP: looks like there is nothing more to do!"
        swap.set_cookie(False)
        print swap.doubledashedline
        return
    else:
        t1 = datetime.datetime.strptime(tonights.parameters['start'],
                                        '%Y-%m-%d_%H:%M:%S')
    print "SWAP: updating all subjects classified between "+\
        tonights.parameters['start']

    # When will we stop considering classifications?
    if tonights.parameters['end'] == 'the_end_of_time':
        tstop = datetime.datetime(2100, 1, 1, 12, 0, 0, 0)
    else:
        tstop = datetime.datetime.strptime(tonights.parameters['end'], 
                                        '%Y-%m-%d_%H:%M:%S')
    print "SWAP: and "+tonights.parameters['end']

    # MAKE IT POSSIBLE TO RUN "DAILY" OR "BATCHLY"
    # in what timestep shall we select classifications to analyze? 
    inc = tonights.parameters['increment']
    increment = datetime.timedelta(days=inc)
    print "SWAP: in increments of %.1f DAY(S)"%inc
    t2 = t1 + increment

    # How many classifications do we look at per batch?
    try: N_per_batch = tonights.parameters['N_per_batch']
    except: N_per_batch = 5000000
    print "SWAP: setting the number of classifications made in this "\
        "batch to ",N_per_batch

    try: 
        prior = tonights.parameters['prior']
    except: 
        tonights.parameters['prior'] = prior
    print "SWAP: set prior for analysis to ",prior

    # Will we do offline analysis?
    try: offline = tonights.parameters['offline']
    except: 
        offline = False
        tonights.parameters['offline']=False
    print "SWAP: should we do offline analysis? ",offline

    # How will we make decisions based on probability?
    thresholds = {}
    thresholds['detection'] = tonights.parameters['detection_threshold']
    thresholds['rejection'] = tonights.parameters['rejection_threshold']

    # will we perform machine learning after SWAP?
    try: machine = tonights.parameters['machine']
    except: machine = False
    print "SWAP: running MachineClassifier.py after this run?",machine
    
    # If we are doing ML, are we sending ALL subjects or just those which are
    # 'best' classified by our users (cross one of the thresholds)? 
    if machine:
        try: training_sample = tonights.parameters['training_sample'].lower()
        except: training_sample = None
        print "SWAP: will pass %s training subjects to machine"%training_sample


    # ------------------------------------------------------------------
    # Read in, or create, a bureau of agents who will represent the
    # volunteers:
    bureau = swap.read_pickle(tonights.parameters['bureaufile'],'bureau')

    # ------------------------------------------------------------------
    # Read in, or create, an object representing the candidate list:
    sample = swap.read_pickle(tonights.parameters['samplefile'],'collection')

    # ------------------------------------------------------------------
    # Read in metadata (all subjects for which we have morph params)
    storage = swap.read_pickle(tonights.parameters['metadatafile'], 'metadata')
    subjects = storage.subjects

    # ------------------------------------------------------------------
    # Open up database:
    db = swap.MySQLdb()

    # Read in a batch of classifications, made since the aforementioned
    # start time:
    batch = db.find('between',t1,t2)  
    print "SWAP: found %i classifications for this batch"%len(batch)

    # ------------------------------------------------------------------

    count_max = N_per_batch
    print "SWAP: interpreting up to",count_max," classifications..."
    if one_by_one: print "SWAP: ...one by one - hit return for the next one..."

    count = 0
    for classification in batch:

        if one_by_one: next = raw_input()

        # Get the vitals for this classification:
        items = db.digest(classification,survey,subjects)
        if vb: print "#"+str(count+1)+". items = ",items
        if items is None:
            continue 

        tstring,Name,ID,ZooID,category,kind,flavor,X,Y,location=items
        t = datetime.datetime.strptime(tstring, '%Y-%m-%d_%H:%M:%S')

        # Break out if we've reached the time limit:
        if t > t2:
            break

        #-------------------------------------------------------------------
        #                 REGISTER NEW VOLUNTEERS / SUBJECTS
        #-------------------------------------------------------------------
        # Register new volunteers, and create an agent for each one:
        try: test = bureau.member[Name]
        except: bureau.member[Name] = swap.Agent(Name,tonights.parameters)

        # Register newly-classified subjects:
        try: test = sample.member[ID]
        except: sample.member[ID] = swap.Subject(ID,ZooID,category,kind,
                                                 flavor,Y,thresholds,location,
                                                 prior=prior)

        # Update the subject's probability using input from the classifier. 
        # We send that classifier's agent to the subject to do this.
        sample.member[ID].was_described(by=bureau.member[Name],as_being=X,
                                        at_time=tstring, while_ignoring=
                                        a_few_at_the_start, haste=waste)

        P = sample.member[ID].mean_probability

        #----------------------------------------------------------------
        #                UPDATE THE METADATA FILE FOR ML
        #----------------------------------------------------------------
        if machine:            
            # --------------------------------------------------
            # Method1: EVERYTHING humans see -- the machine sees
            if training_sample == 'all':
                new_subject = subjects[int(ID)-1]
                new_subject['SWAP_prob'] = P
                
                if new_subject['MLsample'] == 'test':
                    new_subject['MLsample'] = 'train'
                
            # ----------------------------------------------------
            # Method2: Machine only sees subjects which have crossed 
            #          rejected/accepted thresholds; and expert sample
            #          are treated as validation only! 
            elif training_sample == 'best':
                if (sample.member[ID].status != 'undecided') or \
                   (sample.member[ID].state == 'inactive'):
                    new_subject = subjects[int(ID)-1]
                    new_subject['SWAP_prob'] = P

                    if new_subject['MLsample'] == 'test':
                        new_subject['MLsample'] = 'train'

            else:
                print "SWAP: not sure what type of training sample to send "\
                    "to the machine!"

        #-------------------------------------------------------------------
        #                 UPDATE AGENT'S CONFUSION MATRIX
        #-------------------------------------------------------------------
        if supervised_and_unsupervised:
            # use both training and test images
            if agents_willing_to_learn * ((category == 'test') + \
                                          (category == 'training')):
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,
                                          with_probability=P,ignore=False,
                                          ID=ID,at_time=tstring)
            elif ((category == 'test') + (category == 'training')):
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,
                                          with_probability=P,ignore=True,
                                          ID=ID,at_time=tstring)
        elif supervised:
            # Only use training images!
            if category == 'training' and agents_willing_to_learn:
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,
                                          with_probability=P,ignore=False,
                                          ID=ID,at_time=tstring)
            elif category == 'training':
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,
                                          with_probability=P,ignore=True,
                                          ID=ID,at_time=tstring)
        else:
            # Unsupervised: ignore all the training images...
            if category == 'test' and agents_willing_to_learn:
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,
                                          with_probability=P,ignore=False,
                                          ID=ID,at_time=tstring)
            elif category == 'test':
                bureau.member[Name].heard(it_was=X,actually_it_was=Y,
                                          with_probability=P,ignore=True,
                                          ID=ID,at_time=tstring)

        # Brag about it:
        count += 1
        if vb:
            print swap.dashedline
            print "SWAP: Subject "+ID+" was classified by "+Name+\
                " during Stage ",stage
            # GENERISIZE THIS
            print "SWAP: he/she said "+X+" when it was actually "+Y+\
                ", with Pr(LENS) = "+str(P)
            print "SWAP: their agent reckons their contribution (in bits) = ",\
                bureau.member[Name].contribution
            print "SWAP: while estimating their PL,PD as ",\
                bureau.member[Name].PL,bureau.member[Name].PD
            print "SWAP: and the subject's new probability as ",\
                sample.member[ID].probability
        else:
            # Count up to 74 in dots:
            if count == 1: sys.stdout.write('SWAP: ')
            elif np.mod(count,int(count_max/73.0)) == 0: sys.stdout.write('.')
            # elif count == db.size(): sys.stdout.write('\n')
            sys.stdout.flush()

        # When was the first classification made?
        if count == 1:
            t1 = t
        # Did we at least manage to do 1?
        elif count == 2:
            swap.set_cookie(True)
        # Have we done enough for this run?
        elif count == count_max:
            break

    sys.stdout.write('\n')
    if vb: print swap.dashedline
    print "SWAP: total no. of classifications processed: ",count

    #-------------------------------------------------------------------------
    # offline.py goes here
    #-------------------------------------------------------------------------

    # PUT THIS BACK THE WAY IT WAS KINDA? 
    # IT DOESN'T WORK HOW I WANT FOR GZ STUFF BUT IT APPARENTLY WORKED WELL
    # FOR SW STUFF. 

    # All good things come to an end:
    if count == 0:
        print "SWAP: 0 classifications found."

        if sample.size()>0 and bureau.size() > 0: 
            plots = True
            print "SWAP: plotting what has been processed so far..."
        else:
            print "SWAP: something might be wrong..."

        t = t1
        more_to_do = False

        # return
    #elif count < count_max: # ie we didn't make it through the whole \
    # batch  this time!
    #more_to_do = False
    else:
        more_to_do = True

    # ------------------------------------------------------------------
    #                       WRITE REPORTS/CATALOGS
    # ------------------------------------------------------------------
    if report:
        
        tonights.parameters['trunk']=survey+'_'+tonights.parameters['start']
            
        tonights.parameters['dir']=os.getcwd()+'/'+tonights.parameters['trunk']
        
        if not os.path.exists(tonights.parameters['dir']):
            os.makedirs(tonights.parameters['dir'])

        # Output list of subjects to retire, based on this batch of
        # classifications. Note that what is needed here is the ZooID,
        # not the subject ID:

        new_retirefile = swap.get_new_filename(tonights.parameters,
                                               'retire_these')
        print "SWAP: saving retiree subject Zooniverse IDs..."
        N = swap.write_list(sample,new_retirefile,item='retired_subject')
        print "SWAP: "+str(N)+" lines written to "+new_retirefile


        # ----------------------------------------------------------------
        # Also print out lists of detections etc! These are urls of images.

        new_samplefile = swap.get_new_filename(tonights.parameters,'rejected')
        print "SWAP: saving rejected subjects..."
        N = swap.write_list(sample, new_samplefile, item='rejected')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile

        new_samplefile = swap.get_new_filename(tonights.parameters,'detected')
        print "SWAP: saving detected subjects..."
        N = swap.write_list(sample,new_samplefile,item='detected')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile

        # Now save the training images, for inspection:
        new_samplefile = swap.get_new_filename(tonights.parameters,\
                                               'training_true_positives')
        print "SWAP: saving true positives..."
        N = swap.write_list(sample,new_samplefile,item='true_positive')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile

        new_samplefile = swap.get_new_filename(tonights.parameters,\
                                               'training_false_positives')
        print "SWAP: saving false positives..."
        N = swap.write_list(sample,new_samplefile,item='false_positive')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile

        new_samplefile = swap.get_new_filename(tonights.parameters,\
                                               'training_false_negatives')
        print "SWAP: saving false negatives..."
        N = swap.write_list(sample,new_samplefile,item='false_negative')
        print "SWAP: "+str(N)+" lines written to "+new_samplefile


        # -------------------------------------------------------------------
        #####   THESE ARE CATALOGS THAT SPACEWARPS WAS INTERESTED IN   #####

        catalog = swap.get_new_filename(tonights.parameters,'candidate_catalog')
        print "SWAP: saving catalog of high probability subjects..."
        N, Ntot = swap.write_catalog(sample,catalog,thresholds, kind='test')
        print "SWAP: From "+str(Ntot)+" subjects classified,"
        print "SWAP: "+str(N)+" candidates (with P > rejection) written to "\
            +catalog

        catalog = swap.get_new_filename(tonights.parameters,'sim_catalog')
        print "SWAP: saving catalog of high probability subjects..."
        N, Ntot = swap.write_catalog(sample,catalog,thresholds, kind='sim')
        print "SWAP: From "+str(Ntot)+" subjects classified,"
        print "SWAP: "+str(N)+" sim 'candidates' (with P > rejection) "\
            "written to "+catalog

        catalog = swap.get_new_filename(tonights.parameters,'dud_catalog')
        print "SWAP: saving catalog of high probability subjects..."
        N,Ntot = swap.write_catalog(sample,catalog,thresholds,kind='dud')
        print "SWAP: From "+str(Ntot)+" subjects classified,"
        print "SWAP: "+str(N)+" dud 'candidates' (with P > rejection) "\
            "written to "+catalog


        # ------------------------------------------------------------------
        ##########    THESE ARE CATALOGS I'M INTERESTED IN    ##############

        catalog = swap.get_new_filename(tonights.parameters,'detected_catalog')
        print "SWAP: saving catalog of detected subjects..."
        N1,Ntot = swap.write_catalog(sample, catalog,thresholds,kind='detected')
        print "SWAP: From "+str(Ntot)+" subjects classified"
        print "SWAP: "+str(N1)+" detected subjects written to "+catalog 

        catalog = swap.get_new_filename(tonights.parameters,'rejected_catalog')
        print "SWAP: saving catalog of rejected subjects..."
        N2,Ntot = swap.write_catalog(sample, catalog,thresholds,kind='rejected')
        print "SWAP: From "+str(Ntot)+" subjects classified"
        print "SWAP: "+str(N2)+" rejected subjects written to"+catalog

        catalog = swap.get_new_filename(tonights.parameters,'retired_catalog')
        print "SWAP: saving catalog of retired subjects..."
        N3,Ntot = swap.write_catalog(sample, catalog, thresholds,kind='retired')
        print "SWAP: From "+str(Ntot)+" subjects classified"
        print "SWAP: "+str(N3)+" retired subjects (with P < rejection OR "\
            "P > detection) written to "+catalog
      
        equal = N1+N2==N3
        print "SWAP: SANITY CHECK! %i + %i = %i? %s"%(N1,N2,N3, equal)


    # ------------------------------------------------------------------
    # Pickle the bureau, sample, and database, if required. If we do
    # this, its because we want to pick up from where we left off
    # (ie with SWAPSHOP) - so save the pickles in the $cwd. This is
    # taken care of in io.py. Note that we update the parameters as
    # we go - this will be useful later when we write update.config.
    
    if tonights.parameters['repickle'] and count > 0:

        new_bureaufile = swap.get_new_filename(tonights.parameters,'bureau')
        print "SWAP: saving agents to "+new_bureaufile
        swap.write_pickle(bureau,new_bureaufile)
        tonights.parameters['bureaufile'] = new_bureaufile

        new_samplefile = swap.get_new_filename(tonights.parameters,'collection')
        print "SWAP: saving subjects to "+new_samplefile
        swap.write_pickle(sample,new_samplefile)
        tonights.parameters['samplefile'] = new_samplefile
        
        metadatafile = swap.get_new_filename(tonights.parameters,'metadata')
        print "SWAP: saving metadata to "+metadatafile
        swap.write_pickle(storage,metadatafile)
        tonights.parameters['metadatafile'] = metadatafile

    # ------------------------------------------------------------------
    # If there is more to do we need to update the config file for the next day

    if more_to_do:
        # Turn off cookie and start the big plots
        if t2 == tstop: 
            swap.set_cookie(False)
            plots = True

        # otherwise, increment by the timestep 
        # Turn cookie on and update the config "start" 
        else: 
            swap.set_cookie(True)
            tonights.parameters['start'] = t2.strftime('%Y-%m-%d_%H:%M:%S')
                
    else:
        swap.set_cookie(False)
    # SWAPSHOP will read this cookie and act accordingly.

    
    # UPDATE CONFIG FILE with pickle filenames, dir/trunk, and (maybe) new day
    # ----------------------------------------------------------------------
    configfile = configfile.replace('startup','update')
    #configfile = 'update.config'

    # Random_file needs updating, else we always start from the same random
    # state when update.config is reread!
    random_file = open(tonights.parameters['random_file'],"w");
    random_state = np.random.get_state();
    cPickle.dump(random_state,random_file);
    random_file.close();
    swap.write_config(configfile, tonights.parameters)
    #------------------------------------------------------------------
    plots = True
    if plots:

        # Make plots! Can't plot everything - uniformly sample 200 of each
        # thing (agent or subject).

        Nc = np.min([200,bureau.size()])

        # ---------------------------------------------------------------
        # Agent histories 
        # (This can't be plotted if some of your agents
        # don't see training images -- requires Ntraining > 0):
        
        try:
            #"""
            fig1 = bureau.start_history_plot()
            pngfile = swap.get_new_filename(tonights.parameters,'histories')
            print "SWAP: plotting "+str(Nc)+" agent histories in "+pngfile
            
            for Name in bureau.shortlist(Nc):
                bureau.member[Name].plot_history(fig1)
                
                bureau.finish_history_plot(fig1,t,pngfile)
                tonights.parameters['historiesplot'] = pngfile
            #"""
        except: 
            print "SWAP: Couldn't print agent historys because not all "\
                "volunteers saw at least one training image"
            pass

        # ---------------------------------------------------------------
        # Agent probabilities:

        pngfile = swap.get_new_filename(tonights.parameters,'probabilities')
        print "SWAP: plotting "+str(Nc)+" agent probabilities in "+pngfile
        bureau.plot_probabilities(Nc,t,pngfile)
        tonights.parameters['probabilitiesplot'] = pngfile

        # ---------------------------------------------------------------
        # Subject trajectories:

        fig3 = sample.start_trajectory_plot()
        pngfile = swap.get_new_filename(tonights.parameters,'trajectories')

        # Random 500  for display purposes:
        Ns = np.min([500,sample.size()])
        print "SWAP: plotting "+str(Ns)+" subject trajectories in "+pngfile

        for ID in sample.shortlist(Ns):
            sample.member[ID].plot_trajectory(fig3)

        # To plot only false negatives, or only true positives:
        # for ID in sample.shortlist(Ns,kind='sim',status='rejected'):
        #     sample.member[ID].plot_trajectory(fig3)
        # for ID in sample.shortlist(Ns,kind='sim',status='detected'):
        #     sample.member[ID].plot_trajectory(fig3)

        sample.finish_trajectory_plot(fig3,pngfile,t=t)
        tonights.parameters['trajectoriesplot'] = pngfile

        # ---------------------------------------------------------------
        # Candidates! Plot all undecideds or detections:

        fig4 = sample.start_trajectory_plot(final=True)
        pngfile = swap.get_new_filename(tonights.parameters,'sample')

        # BigN = 100000 # Would get them all...
        BigN = 100      # Can't see them all!
        candidates = []
        candidates += sample.shortlist(BigN,kind='test',status='detected')
        candidates += sample.shortlist(BigN,kind='test',status='undecided')
        sims = []
        sims += sample.shortlist(BigN,kind='sim',status='detected')
        sims += sample.shortlist(BigN,kind='sim',status='undecided')
        duds = []
        duds += sample.shortlist(BigN,kind='dud',status='detected')
        duds += sample.shortlist(BigN,kind='dud',status='undecided')

        print "SWAP: plotting "+str(len(sims))+" sims in "+pngfile
        for ID in sims:
            sample.member[ID].plot_trajectory(fig4)
        print "SWAP: plotting "+str(len(duds))+" duds in "+pngfile
        for ID in duds:
            sample.member[ID].plot_trajectory(fig4)
        print "SWAP: plotting "+str(len(candidates))+" candidates in "+pngfile
        for ID in candidates:
            sample.member[ID].plot_trajectory(fig4)

        # They will all show up in the histogram though:
        sample.finish_trajectory_plot(fig4,pngfile,final=True)
        tonights.parameters['candidatesplot'] = pngfile

        # ------------------------------------------------------------------
        # Finally, write a PDF report:

        swap.write_report(tonights.parameters,bureau,sample)

    print swap.doubledashedline
    return

# ======================================================================

if __name__ == '__main__':
    SWAP(sys.argv[1:])

# ======================================================================
