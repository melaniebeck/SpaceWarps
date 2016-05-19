import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import pdb, os, subprocess
import MySQLdb as mdb
import datetime, cPickle
from astropy.table import Table, vstack, join
from optparse import OptionParser
import swap

from figure_styles import set_pub

connection = mdb.connect('localhost', 'root', '8croupier!', 'gz2')
cursor = connection.cursor(mdb.cursors.DictCursor)



def plot_retired(config):
    set_pub()

    

    # PLOT #RETIRED/DETECTED PER DAY SWAP VS GZ2
    # ======================================================================
    classifications_per_day = True

    # First, get # of retired/detected subjects per day from SWAP
    retired, detected, total = np.array([]), np.array([]), np.array([])
    
    days = int(subprocess.check_output("ls logfiles_%s/* | wc -l"%config, 
                                       shell=True))
    command = "find %s*/ -maxdepth 1 -type d -print | wc -l"%tonights.parameters['survey']
    days = int(subprocess.check_output())
    days=days-1

    days = 76

    logfiles = ["logfiles_%s/GZ2_%i.log"%(config,i) for i in range(days)]


    for log in logfiles:
        #pdb.set_trace()

        ret = subprocess.check_output("awk '/retired_catalog/ {print $2}' %s"
                                      %log, shell=True)
        det = subprocess.check_output("awk '/detected_catalog/ {print $2}' %s"
                                      %log, shell=True)
        tot = subprocess.check_output("awk '/SWAP: From/ {print $3}' %s"
                                        %log, shell=True).splitlines()[0]
        try: 
            retired = np.append(retired, int(ret))
            detected = np.append(detected, int(det))
            total = np.append(total, int(tot))
        except: 
            retired = np.append(retired,np.nan)
            detected = np.append(detected, np.nan)
            total = np.append(total, np.nan)

    undecided = total - (retired+detected)
    size = total[-1]
    
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111)
    dates = np.array([i for i in range(days)])
    ax.plot(dates, retired*1./size, 'r',label="SWAP: 'Not'")
    ax.plot(dates, detected*1./size, 'b', label="SWAP: 'Smooth'")
    ax.plot(dates, undecided*1./size, 'k', label="SWAP: Undecided")
    ax.axhline(y=.35, ls='--', color='r')
    ax.axhline(y=.5, ls='--', color='b')
    ax.legend(loc='best')
    ax.set_xlabel('Days in GZ2')
    ax.set_ylabel('Number of Subjects')
    #plt.show()
    #'''

    # Now get # of "retired" subjects per day from original GZ2
    # ----------------------------------------------------------------------
    delta = datetime.timedelta(days=1)
    starttime = '2009-02-17 00:00:00'
    get_subjects = False

    try: 
        F = open('GZ2_cumulative_subjects.pickle','rb')
        subjects_per_day = cPickle.load(F)
        subjects_per_day = subjects_per_day[:76]
        F.close()
    except:
        subjects_per_day = []

    if len(subjects_per_day)!=days: 
        get_subjects=True
        time = datetime.datetime(2009,02,17,0,0,0)+\
               datetime.timedelta(days=len(subjects_per_day))
        time = time.strftime('%Y-%m-%d %H:%M:%S')
    else: time = starttime

    if get_subjects:
        for d in range(len(subjects_per_day), days):
            pdb.set_trace()
            subjects = 0
            # This will return a list of all subjects and the cumulative number
            # of classifications they've received thus far
            query = ("select t.name, count(t.name) as count "
                     "from task1_full as t "
                     "where t.created_at < '%s' "
                     "group by t.name order by count(t.name) desc"%time)
            cursor.execute(query)
            batch = cursor.fetchall()
            
            # do whatever processing needs to be done
            # find the number of objects with cumulative classifications > 40
            for thing in batch:
                if int(thing['count']) > 25: subjects += 1
                
            # convert stringtime to datetime object
            time = datetime.datetime.strptime(time,'%Y-%m-%d %H:%M:%S')
            # add one day
            time = time + delta
            # convert datetime object to string
            time = time.strftime('%Y-%m-%d %H:%M:%S')
            
            subjects_per_day.append(subjects)
            
        F = open('GZ2_cum_subjects_%s.pickle'%config,'w')
        cPickle.dump(subjects_per_day,F,protocol=2)
        F.close()

    # Make the Figure - Bar Chart
    #-------------------------------------------------------------------------
    dates = [i for i in range(days)]
    width = 0.35
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)

    if classifications_per_day:
        #read in classifications per day:
        data = Table.read('task1_full_classbyday.txt',format='ascii')

        # select only up to whatever day we're currently on
        batch = data[:days]

        # plot that shit!
        #gzclass = ax.plot(batch['col4'], color='purple', alpha=.25)
        #ax.fill_between(dates,0,batch['col4'],facecolor='purple',
        #                alpha=0.20, label='GZ2 classifications')
    
    #rets = ax.bar(dates, retired,width, color='r')
    #dets = ax.bar(dates, detected, width, color='b', bottom=retired)
    #tots = ax.bar(dates, detected+retired, width, color='r')
    #orig = ax.bar(np.array(dates)+width, subjects_per_day, width, color='y')
    tots = ax.fill_between(dates, detected+retired, color='y', 
                           alpha=0.6, label='Filtering')
    orig = ax.fill_between(dates, subjects_per_day, color='b', 
                           alpha=0.5, label='GZ2')
    
    #ax.set_title("Cumulative Number of 'Retired' Subjects: SWAP vs. GZ2")
    ax.set_title("Cumulative Number of Classified Subjects", fontsize=30, 
                 weight='bold')
    ax.set_xlabel("Time (days)",fontsize=26,weight='bold')
    ax.set_ylabel("Number of Subjects",fontsize=26,weight='bold')
    #ax.set_yscale("log")

    ax.set_xticks(np.array(dates[::4])+width)
    ax.set_xticklabels(dates[::4])
    ax.set_xlim(0,days-1)
    
    #ax.legend((dets[0],rets[0], orig[0], gzclass[0]), 
    #          ("SWAP: 'Smooth'","SWAP: 'Not'", 'GZ2 > 25', 
    #           'GZ2 classifications'), loc='best')
    legend = ax.legend(loc='best')
    frame = legend.get_frame()
    frame.set_linewidth(2)
    for label in legend.get_texts():
        label.set_fontsize('large')

    #ax.legend((tots[0],orig[0]), ("Filtering", "GZ2"))
    #plt.savefig('classificationsperday_SWAPvGZ2_%i_%s.png'%(days,config))
    plt.tight_layout()
    plt.savefig('DDF_classificationsperday.png')
    plt.show()

    return days
    

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


def summarize_accuracy(data):
    GZ2 = data['gz2class']
    SWAP = data['P']

    result, false_positives, false_negatives = [], [], []
    match, mismatch = 0, 0
    swap_not, swap_smooth = 0, 0 
    smooth_match, not_match = 0, 0
    
    for idx, d in enumerate(data):
        if ('E' in d['gz2class']) and (d['P']>0.95):
            smooth_match+=1
            match+=1
            result.append("smooth_match")
        elif ('E' not in d['gz2class']) and (d['P']<0.02):
            not_match+=1
            match+=1
            result.append("not_match")
        elif ('E' in d['gz2class']) and (d['P']<0.02):
            mismatch +=1
            swap_not +=1
            result.append("false_positive")
        elif ('E' not in d['gz2class']) and (d['P']>0.95):
            mismatch+=1
            swap_smooth +=1
            result.append("false_negative")

    data['result'] = np.array(result)

    percent_match = match*1.0/len(data)
    percent_mismatch = 1-percent_match

    percent_swap_not = swap_not*1.0/len(data)
    percent_swap_smooth = swap_smooth*1.0/len(data)

    print "Percent matched:", percent_match
    print "Percent SWAP said NOT ('incorrectly'):", percent_swap_not
    print "Percent SWAP said SMOOTH ('incorrectly'):", percent_swap_smooth

    return data


def explore_incorrect(data, config):

    false_negatives = data[np.where(data['result'] == 'false_negative')]
    false_positives = data[np.where(data['result'] == 'false_positive')]

    matches = data[np.where((data['result']=='smooth_match') | 
                           (data['result']=='not_match'))]
    mismatch = data[np.where((data['result']=='false_negative') | 
                            (data['result']=='false_positive'))]


    matches_smooth = matches['t01_smooth_or_features_a01_smooth_debiased']
    matches_feature = matches['t01_smooth_or_features_a02_features_or_disk_debiased']
    mismatch_smooth = mismatch['t01_smooth_or_features_a01_smooth_debiased']
    mismatch_feature = mismatch['t01_smooth_or_features_a02_features_or_disk_debiased']

    #pdb.set_trace()

    '''
    fn_smooth = false_negatives['t01_smooth_or_features_a01_smooth_debiased']
    fn_feature = false_negatives['t01_smooth_or_features_a02_features_or_disk_debiased']
    
    fs_smooth =  false_positives['t01_smooth_or_features_a01_smooth_debiased']
    fs_feature = false_positives['t01_smooth_or_features_a02_features_or_disk_debiased']
    #ax.plot(fn_smooth_deb, fn_feature_deb, 'yo', label='false negatives', 
    #        alpha=.5)
    #ax.plot(fs_smooth_deb, fs_feature_deb, 'ro', label='false positives', 
    #        alpha=.5)

    '''
    
    fig = plt.figure(figsize=(10,10))

    most = np.max(data['Nclass'])
    colors1 = (matches['Nclass']*1.0/most).tolist()

    ax1 = fig.add_subplot(211)
    thing1 = ax1.scatter(matches_smooth, matches_feature, c=colors1,  marker='o')
    ax1.set_xlim(-.1,1.1)
    ax1.set_ylim(-.1,1.1)

    ax1.set_ylabel('p_feature debiased')
    ax1.set_title('Matches')

    ticks = [.9, .6, .2]
    cbax = fig.colorbar(thing1, ticks=ticks)
    labels = [str(t*most) for t in ticks]
    cbax.ax.set_yticklabels(labels)
    cbax.ax.set_ylabel('Number of Classifications')

    #---------------------------------------------------------------
    colors2 = (mismatch['Nclass']*1.0/most).tolist()

    ax2 = fig.add_subplot(212)
    thing2 = ax2.scatter(mismatch_smooth, mismatch_feature, c=colors2, marker='o')
    ax2.set_xlim(-0.1,1.1)
    ax2.set_ylim(-0.1,1.1)

    ax2.set_xlabel('p_smooth debiased')
    ax2.set_ylabel('p_feature debiased')
    ax2.set_title('Mismatches')

    cbax = fig.colorbar(thing2, ticks=ticks)
    cbax.ax.set_yticklabels(labels)
    cbax.ax.set_ylabel('Number of Classifications')

    plt.tight_layout()

    plt.savefig('explore_mismatches_Nclass.png')
    plt.show()


def main(options, args):   
    '''
    I automated this! Booyah.
    
    # I've matched zoo2MainAll.fits to a concatenated version of
    # GZ2_sup_0.75_2009-XX-XX_00:00:00_retired+detected_catalogs
    
    #filename = 'GZ2_MainAll_SWAP_run3_5-29-2009.fits'
    '''
    # access last time-stamped directory in the sim of interest
    #directory = subprocess.check_output("ls -rt %s | tail -1"%options.config, 
    #                                    shell=True).splitlines()[0]
    #basename ="GZ2_MainAll_SWAP_%s_%s"%(options.config,directory.split('_')[3])
    directory = 'sup_0.75_run'
    prefix = 'GZ2_sup_0.75_2009-05-03_00:00:00'
    combofile = 'GZ2_MainAll_SWAP_run1_05-03-2009'

    #days = plot_retired(options.config)
    days = plot_retired('sup_0.75')
    pdb.set_trace()

    pwd = os.getcwd()
    try:
        joined = Table.read('%s.fits'%basename)
    except:
        print "Making joined catalog"
        # Prepare "retired" and "detected" catalogs -------------------------
        detect = '%s/%s/%s_detected_catalog.txt'%(directory,prefix,prefix)
        retire = '%s/%s/%s_retired_catalog.txt'%(directory,prefix,prefix)
        
        # read in the files and concatenate along rows
        detected = Table.read(detect, format='ascii')
        retired = Table.read(retire, format='ascii')
        combined = vstack((detected,retired))
        
        # Now, match it against zoo2MainAll.fits
        #zoo2MainAll = Table.read('zoo2MainAll_urls.fits')
        #zoo2MainAll['zooid']=zoo2MainAll['dr7objid']

        #joined = join(zoo2MainAll, combined, keys='zooid')
        
        #Table.write(joined, '%s.fits'%basename)
    

    # Now go to work --------------------------------------------------------
    #plot_num_classifications_to_retire(joined, days, options.config)

    try: results = Table.read('%s_results.fits'%basename)
    except:
        print "This should already exist..."
        pdb.set_trace()
        results = summarize_accuracy(joined)
        Table.write(results, '%s_results.fits'%filename)

    explore_incorrect(results, options.config)


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-c", dest="config", default=None)
    
    (options, args) = parser.parse_args()

    main(options, args)
