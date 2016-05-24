import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pdb, os, subprocess, sys
import MySQLdb as mdb
import datetime, cPickle
from astropy.table import Table, vstack, join
from optparse import OptionParser
import swap
import glob

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


def fetch_num_retired_SWAP(params):   

    print "Fetching subjects classified by GZX..."

    retired = []
    
    cmd = "ls %s*/*retire_these.txt"%params['survey']

    # Pull up all "retired" files from each day of this run
    cmdout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    retirefiles = cmdout.stdout.read().splitlines()
   
    for retirefile in retirefiles:
        # read in the retirefile
        with open(retirefile,'rb') as f:
            stuff = f.read().splitlines()
            
        retired.append(len(stuff))

    print "Fetched a grand total of %i subjects classififed by GZX"%retired[-1]
    return np.array(retired)


def fetch_num_detected_rejected_SWAP(params):
    
    # ------------------------------------------------------------------
    # Load up detected subjects... 
    print "Fetching subjects detected and rejected by GZX..."

    try:
        cmd = "ls %s*/*detected_catalog.txt"%params['survey']
        cmdout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        detectfiles = cmdout.stdout.read().splitlines()

    except:
        print "No 'detected' files found! Aborting..."
        print ""
        sys.exit()

    try:
        cmd = "ls %s*/*retired_catalog.txt"%params['survey']
        cmdout = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
        rejectfiles = cmdout.stdout.read().splitlines()

    except:
        print "No 'rejected' files found! Aborting..."
        print ""
        sys.exit()

    detected, rejected = [], []
    for dfile, rfile in zip(detectfiles, rejectfiles):
            
        with open(dfile,'rb') as f1:
            stuff = f1.read().splitlines()
        detected.append(len(stuff))

        with open(rfile,'rb') as f2:
            stuff = f2.read().splitlines()
        rejected.append(len(stuff))
        
        
    num_retired = np.column_stack((detected,rejected))

    print "Fetched a grand total of %i subjects detected by GZX"%detected[-1]
    print "Fetched a grand total of %i subjects rejected by GZX"%rejected[-1]
    
    return num_retired    


def fetch_num_retired_GZ(num_days, delta=1, condition=25, get_retired=True):

    print "Fetching subjects retired by Galaxy Zoo 2..."
    
    # ----------------------------------------------------------------------
    # Connect to GZ2 database

    connection = mdb.connect('localhost', 'root', '8croupier!', 'gz2')
    cursor = connection.cursor(mdb.cursors.DictCursor)

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
    # If one of those doesn't hold, fetch all (or additional) retired subjects

    if get_retired:
        
        delta = datetime.timedelta(days=1)

        for d in range(len(cum_retired_per_day), num_days):

            # This will return a list of all subjects and the cumulative number
            # of classifications they've received thus far
            query = ("select t.name, count(t.name) as count "
                     "from task1_full as t "
                     "where t.created_at < '%s' "
                     "group by t.name "
                     "having count(t.name) > %i"%(time, condition))
            cursor.execute(query)
            batch = cursor.fetchall()
            
            cum_retired_per_day.append(len(batch))

            # convert stringtime to datetime object
            time = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S')

            # add one day
            time = time + delta

            # convert datetime object to string
            time = time.strftime('%Y-%m-%d %H:%M:%S')
            
            
        F = open('GZ2_cumulative_retired_subjects.pickle','w')
        cPickle.dump(cum_retired_per_day,F,protocol=2)
        F.close()

    print "Fetched a grand total of %i subject retired by GZ2"\
        %cum_retired_per_day[-1]

    return cum_retired_per_day



def plot_retired_GZ_vs_SWAP(GZX_retired, GZ2_retired, num_days,outfilename=None, 
                            classifications_per_day=True, bar=False):

    set_pub()


    #--------------------------------------------------------------
    # Make the Figure - Bar Chart

    dates = np.array([i for i in range(num_days)])
    width = 0.35
    

    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    if classifications_per_day:
        
        try: 
            clicks_per_day = Table.read('task1_full_classbyday.txt',
                                        format='ascii')
        except:
            fetch_classifications_per_day(num_days)

        # select only up to whatever day we're currently on
        batch = clicks_per_day[:num_days]

        # plot that shit!
        gzclass = ax.plot(batch['col4'], color='black', alpha=.25)
        ax.fill_between(dates,0,batch['col4'],facecolor='black',
                        alpha=0.20, label='GZ2 classifications')
                                        

    GZX_retired = np.array(GZX_retired)

    if GZX_retired.ndim == 1:
        if bar: 
            tots = ax.bar(dates, GZX_retired, width, color='orange', alpha=0.5)

        else:
            ax.plot(dates, GZX_retired, color='orange')
            tots = ax.fill_between(dates, GZX_retired, color='orange',alpha=0.5)

    else:
        detected = GZX_retired[:,0]
        rejected = GZX_retired[:,1]
        
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
    ax.set_xlabel("GZ2 Time (days)", fontsize=16, weight='bold')
    ax.set_ylabel("Number of Subjects", fontsize=16, weight='bold')

    ax.set_xticks(dates[::4]+width)
    ax.set_xticklabels(dates[::4])
    ax.set_xlim(0,num_days-1)
    
    if GZX_retired.ndim > 1 and classifications_per_day:
        legend = ax.legend((dets, rejs, orig, gzclass[0]), 
                           ("GZX: 'Smooth'","GZX: 'Not'", 'GZ2', 
                            'GZ2 classifications'), loc='best')

    elif GZX_retired.ndim > 1:
        legend = ax.legend((dets[0], rejs[0], orig[0]), 
                           ("GZX: 'Smooth'","GZX: 'Not'", 'GZ2'), loc='best')

    elif GZX_retired.ndim == 1 and classifications_per_day:
        legend = ax.legend((tots, orig, gzclass[0]), 
                           ('GZX', 'GZ2', 'GZ2 classifications'), loc='best')

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



# ----------------------------------------------------------------------#
# ----------------------------------------------------------------------#
def main(options, args):   

    params = fetch_parameters(options.config)

    num_days = fetch_num_days(params)

    if options.combined_subjects:
        GZX_retired_subjects = fetch_num_retired_SWAP(params)   
    else:
        GZX_retired_subjects = fetch_num_detected_rejected_SWAP(params)


    GZ2_retired_subjects = fetch_num_retired_GZ(num_days)

    plot_retired_GZ_vs_SWAP(GZX_retired_subjects, GZ2_retired_subjects, 
                            num_days, outfilename="sup_0.75")



# ----------------------------------------------------------------------#
# ----------------------------------------------------------------------#
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", dest="config", default=None)
    parser.add_option("-s", "--combo", dest="combined_subjects",
                      action='store_false', default=True)
    
    (options, args) = parser.parse_args()

    main(options, args)
