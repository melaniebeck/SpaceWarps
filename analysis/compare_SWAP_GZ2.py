import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pdb, os, subprocess, sys
import MySQLdb as mdb
import datetime, cPickle
from astropy.table import Table, vstack, join
from argparse import ArgumentParser
import swap
import glob

from ground_truth_create_catalog import find_indices
from figure_styles import set_pub



def fetch_parameters(config):

    # Open Configuration File 
    p = swap.Configuration(config)
    params = p.parameters

    return params


def fetch_num_days(params):
    days = int(subprocess.check_output("find %s*/ -maxdepth 1 -type d -print "
                                       "| wc -l"%params['survey'], shell=True))

    return int(days)

       
        
def fetch_filelist(params, kind='detected'):
     
    # ------------------------------------------------------------------
    # Load up detected subjects... 
    print "Fetching list of %s files created by GZX..."%kind

    try:
        cmd = "ls %s*/*%s_catalog.txt"%(params['survey'], kind)
        cmdout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        filelist = cmdout.stdout.read().splitlines()
        
    except:
        print "No '%s' files found! Aborting..."%kind
        print ""
        sys.exit()
        
    return filelist


def fetch_number_of_subjects(filelist, kind='detected'):
    # Fetch the NUMBER detected/rejected 
    # Don't care about anything else
    
    lengthlist = []
    
    for filename in filelist:
        
        with open(filename,'rb') as f1:
            stuff = f1.read().splitlines()
        lengthlist.append(len(stuff))
    
    try:
        print "Fetched a grand total of %i subjects %s by GZX"\
            %(lengthlist[-1],kind)
        return lengthlist 

    except:
        print "Could not fetch any %s subjects by GZX. Aborting..."%kind
        sys.exit()
    

    
def fetch_classifications(filename):
   
    try:
        dat = Table.read(filename,format='ascii')
    except:
        print "Did not find %s"%filename
        sys.exit()
    else:
        return dat

   
def fetch_num_retired_GZ2(num_days, delta=1, condition=25, get_retired=True,
                          expert=False):
        
    print "Fetching subjects retired by Galaxy Zoo 2..."

    # ----------------------------------------------------------------------
    # Get number of "retired" subjects per delta t from original GZ2

    starttime = '2009-02-17 00:00:00'

    # Check for a previous version of this file 
    try: 
        F = open('GZ2_cumulative_retired_subjects.pickle','rb')
        cum_retired_per_day = cPickle.load(F)
        F.close()

    except:
        cum_retired_per_day = []

    # ----------------------------------------------------------------------
    # Check that the previous version (if any) is up to date

    if len(cum_retired_per_day)!=num_days: 
        get_retired=True

        time = datetime.datetime(2009,02,17,0,0,0)+\
               datetime.timedelta(days=len(cum_retired_per_day))
        time = time.strftime('%Y-%m-%d %H:%M:%S')

    else: time = starttime

    
    # ----------------------------------------------------------------------
    # Connect to GZ2 database

    connection = mdb.connect('localhost', 'root', '8croupier!', 'gz2')
    cursor = connection.cursor(mdb.cursors.DictCursor)


    # ----------------------------------------------------------------------
    # If one of those doesn't hold, fetch all (or additional) retired subjects

    if get_retired:
        if expert: 
            table = "task1_expert"
            outfile = 'GZ2_cumulative_retired_subjects_expert.pickle'
        else:
            table = "task1_full"
            outfile = 'GZ2_cumulative_retired_subjects.pickle'
            

        delta = datetime.timedelta(days=1)

        for d in range(len(cum_retired_per_day), num_days):

            # This will return a list of all subjects and the cumulative number
            # of classifications they've received thus far

            query = ("select t.name, count(t.name) as count "
                     "from %s as t "
                     "where t.created_at < '%s' "
                     "group by t.name "
                     "having count(t.name) > %i"%(table, time, condition))
            cursor.execute(query)
            batch = cursor.fetchall()
            
            cum_retired_per_day.append(len(batch))

            # convert stringtime to datetime object
            time = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S')

            # add one day
            time = time + delta

            # convert datetime object to string
            time = time.strftime('%Y-%m-%d %H:%M:%S')
            
        
        F = open(outfile,'w')
        cPickle.dump(cum_retired_per_day,F,protocol=2)
        F.close()

    print "Fetched a grand total of %i subject retired by GZ2"\
        %cum_retired_per_day[-1]

    return cum_retired_per_day



def plot_retired_GZ_vs_SWAP(GZX_retired, GZ2_retired, num_days,
                            outfilename=None, classifications_per_day=True, 
                            bar=False, expert=False):

    set_pub()

    #--------------------------------------------------------------
    # Make the Figure - Bar Chart

    dates = np.arange(num_days)
    width = 0.35
    

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    if classifications_per_day:
        if expert: table = 'task1_expert'
        else: table = 'task1_full'

        try: 
            clicks_per_day = Table.read('%s_classbyday.txt'%table,
                                        format='ascii')
        except:
            fetch_classifications_per_day(num_days, expert=expert)

        # select only up to whatever day we're currently on
        batch = clicks_per_day[:num_days]

        # plot that shit!
        ax.plot(batch['col4'], color='black', alpha=.25)
        gzclass = ax.fill_between(dates,0,batch['col4'],facecolor='black',
                                  alpha=0.20, label='GZ2 classifications')
                                        

    GZX_retired = np.array(GZX_retired)
    GZ2_retired = GZ2_retired[:len(GZX_retired[0,:])]

    if GZX_retired.ndim == 1:
        if bar: 
            tots = ax.bar(dates, GZX_retired, width, color='orange', alpha=0.5)

        else:
            ax.plot(dates, GZX_retired, color='orange')
            tots = ax.fill_between(dates, GZX_retired, color='orange',alpha=0.5)

    else:
        detected = GZX_retired[0,:]
        rejected = GZX_retired[1,:]
        
        if bar:
            dets = ax.bar(dates, detected, width, color='orange',bottom=retired)
            rejs = ax.bar(dates, retired, width, color='yellow')

        else:
            ax.plot(dates, detected+rejected, color='orange')
            dets = ax.fill_between(dates, rejected, detected+rejected, 
                                   color='orange',alpha=0.5)

            ax.plot(dates, rejected, color='yellow')
            rejs = ax.fill_between(dates, rejected, color='yellow', alpha=0.5)


    if bar:
        orig = ax.bar(dates+width, GZ2_retired, width, color='b')

    else:
        ax.plot(dates, GZ2_retired, color='b', alpha=.65)
        orig = ax.fill_between(dates, GZ2_retired, color='b', alpha=0.4)
    
    
    #ax.set_title("Cumulative Number of Classified Subjects", weight='bold')
    ax.set_xlabel("Days in GZ2 project", fontsize=16, weight='bold')
    ax.set_ylabel("Counts", fontsize=16, weight='bold')

    ax.set_xticks(dates[::4]+width)
    ax.set_xticklabels(dates[::4])
    ax.set_xlim(0,num_days-1)
    
    if GZX_retired.ndim > 1 and classifications_per_day:
        legend = ax.legend((dets, rejs, orig, gzclass), 
                           ("GZX: 'Smooth'","GZX: 'Not'", 'GZ2', 
                            'GZ2 user votes'), loc='best')

    elif GZX_retired.ndim > 1:
        legend = ax.legend((dets[0], rejs[0], orig[0]), 
                           ("GZX: 'Smooth'","GZX: 'Not'", 'GZ2'), loc='best')

    elif GZX_retired.ndim == 1 and classifications_per_day:
        legend = ax.legend((tots, orig, gzclass), 
                           ('GZX', 'GZ2', 'GZ2 user votes'), loc='best')

    elif GZX_retired.ndim == 1:
        legend = ax.legend((tots, orig), ('GZX', 'GZ2'), loc='best')

    frame = legend.get_frame()
    frame.set_linewidth(2)
    for label in legend.get_texts():
        label.set_fontsize('large')

    plt.tight_layout()
    plt.savefig('retired_per_day_%s_%idays.png'%(outfilename, num_days))
    plt.show()

    return


def plot_GZX_evaluation(num_days, accuracy, precision, recall, outfile):

    set_pub()

    days = np.arange(num_days)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)

    ax.plot(days, accuracy, c='r', label='Accuracy')
    ax.plot(days, precision, c='g', label='Precision')
    ax.plot(days, recall, c='b', label='Recall')
    ax.legend(loc='best')

    ax.set_xlabel('Days in GZ2 project',fontsize=16,fontweight='bold')
    ax.set_ylabel('Per cent',fontsize=16,fontweight='bold')

    plt.savefig("GZX_evaluation_%s.png"%outfile)
    plt.show()
    plt.close()


def plot_num_classifications_to_retire(filename, days, config):

    dat = Table.read(filename)
    
    SWAP = np.array(dat['Nclass'])

    ret = np.array(dat['Nclass'][np.where(dat['P']<0.02)])
    retired = np.concatenate((ret, np.full(len(SWAP)-len(ret), -1)))

    det = np.array(dat['Nclass'][np.where(dat['P']>0.95)])
    detected = np.concatenate((det, np.full(len(SWAP)-len(det), -1)))

    GZ2 = dat['total_classifications']
    
    #pdb.set_trace()

    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)

    binsize=1
    swapbins = np.arange(np.min(SWAP), np.max(SWAP), binsize)
    retbins = np.arange(0, np.max(SWAP), binsize)
    gz2bins = np.arange(np.min(GZ2), np.max(GZ2), binsize)

    weights=np.ones_like(SWAP)*1.0/len(SWAP),
    ax.hist(SWAP, bins=swapbins, weights=weights, color='purple', 
            histtype='stepfilled', alpha=1, label='SWAP')

    ax.hist(detected,weights=weights, bins=retbins, range=(0,50), color='blue',
            histtype='stepfilled', alpha=0.25, label="SWAP: 'Smooth'")

    ax.hist(retired, weights=weights, bins=retbins, range=(0,50), color='red', 
            histtype='stepfilled', alpha=0.25, label="SWAP: 'Not'")

    weights=np.ones_like(GZ2)*1.0/len(GZ2), 
    ax.hist(GZ2, bins=gz2bins, weights=weights, color='green', 
            histtype='stepfilled', alpha=0.85, label='Original GZ2')

    ax.set_ylim(0,.15)
    ax.set_xlabel('Number of Classifications')
    ax.set_ylabel('Frequency')
    ax.legend(loc='best')

    plt.savefig('counts_till_classification_SWAPvGZ2_%i_%s.png'%(days,config))
    plt.show()



def generate_SWAP_eval_report(detectedfilelist, rejectedfilelist, subjects):

    recall, precision, accuracy = [], [], []
    
    for dfile, rfile in zip(detectedfilelist, rejectedfilelist):
        
        predicted_smooth = fetch_classifications(dfile)
        predicted_not = fetch_classifications(rfile)
        
        gzx = vstack([predicted_smooth['zooid','P','Nclass'], 
                      predicted_not['zooid','P','Nclass']])
        
        # For each subject in detected/rejected, find label in subjects
        result = find_indices(subjects['SDSS_id'], gzx['zooid'])
        gz2 = subjects[result]
        
        actually_smooth = gz2[gz2['GZ2_label'] == 1]
        actually_not = gz2[gz2['GZ2_label'] != 1]
        
        
        # True Positives (predicted 'smooth' == actually smooth)
        tps_idx = find_indices(subjects['SDSS_id'],
                               predicted_smooth['zooid'])
        tps = sum(subjects['GZ2_label'][tps_idx]==1)
        
        # False Positives (predicted 'smooth but actually NOT)
        fps = sum(subjects['GZ2_label'][tps_idx]!=1)
        
        # True Negatives (predicted 'not' == actually not)
        tns_idx = find_indices(subjects['SDSS_id'], predicted_not['zooid'])
        tns = sum(subjects['GZ2_label'][tns_idx]!=1)
        
        # False Negatives (predicted 'not' but actually SMOOTH)
        fns = sum(subjects['GZ2_label'][tns_idx]==1)
        
        if predicted_smooth and predicted_not: 
            recall.append(float(tps)/float(len(actually_smooth)))
            precision.append(float(tps)/float(len(predicted_smooth)))
            accuracy.append(float(tps + tns)/float(tps+fps+tns+fns))
        else:
            recall.append(0.)
            precision.append(0.)
            accuracy.append(0.)          
            
    evalution = Table(data=(accuracy, precision, recall), 
                      names=('accuracy','precision','recall'))
    evaluation.write('GZXevalution_%s.txt'%outname, format='ascii')

    return accuracy, precision, recall



# ----------------------------------------------------------------------#
# ----------------------------------------------------------------------#
def main(args):   

    params = fetch_parameters(args.config)
    num_days = fetch_num_days(params)

    # ---------------------------------------------------------------------
    # Fetch lists of relevant filenames over the course of the run

    detectedfilelist = fetch_filelist(params, kind='detected')

    if args.old_run:
        rejectedfilelist = fetch_filelist(params, kind='retired')
    else:
        rejectedfilelist = fetch_filelist(params, kind='rejected')

    # ---------------------------------------------------------------------
    # Fetch the cumulative number of classified subjects from SWAP

    detected = fetch_number_of_subjects(detectedfilelist, kind='detected')
    rejected = fetch_number_of_subjects(rejectedfilelist, kind='rejected')
    GZX_retired_subjects = np.vstack([detected, rejected])

    # ---------------------------------------------------------------------
    # Fetch the cumulative number of classified subjects from GZ2

    GZ2_retired_subjects = fetch_num_retired_GZ2(num_days,expert=args.expert)


    # Generate appropriate output filename

    if args.combined_subjects:
        outname = args.config[len('update_'):-len('.config')]+'_combo'
        GZX_retired_subjects = np.sum(GZX_retired_subjects,axis=0)
    else:
        outname = args.config[len('update_'):-len('.config')]

    # ---------------------------------------------------------------------
    # Plot that shit

    plot_retired_GZ_vs_SWAP(GZX_retired_subjects, GZ2_retired_subjects, 
                            num_days, outfilename=outname)


    # ---------------------------------------------------------------------
    ### Generate evaluation report as a function of time 

    if args.eval_report:
        try:     
            eval_report = Table.read('GZXevaluation_%s.txt'%outname, 
                                     format='ascii')
            recall = eval_report['recall']
            accuracy = eval_report['accuracy']
            precision = eval_report['precision']
        except:
            meta = swap.read_pickle(params['metadatafile'], 'storage')
            subjects = meta.subjects

            accuracy, recall, precision = generate_SWAP_eval_report(
                detectedfilelist, rejectedfilelist, subjects)

        plot_GZX_evaluation(num_days, accuracy, precision, recall, outname)


# ----------------------------------------------------------------------#
# ----------------------------------------------------------------------#
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("-c", dest="config", default=None)
    parser.add_argument("-s", "--combo", dest="combined_subjects",
                      action='store_true', default=False)
    parser.add_argument("-o", "--old", dest='old_run', action='store_true',
                      default=False)
    parser.add_argument("-e", "--eval", dest='eval_report', action='store_true',
                      default=False)   
    parser.add_argument("-x", "--expert", dest='expert', action='store_true',
                      default=False)

    args = parser.parse_args()

    main(args)
