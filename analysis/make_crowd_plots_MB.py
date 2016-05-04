import sys,getopt,numpy as np
import pdb

import matplotlib
matplotlib.use('Agg')

# Fonts, latex:
matplotlib.rc('font',**{'family':'serif', 'serif':['TimesNewRoman']})
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt

bfs,sfs = 20,16
params = { 'axes.labelsize': bfs,
                'font.size': bfs,
          'legend.fontsize': bfs,
          'xtick.labelsize': sfs,
          'ytick.labelsize': sfs}
plt.rcParams.update(params)

import swap
import corner




# ======================================================================

def make_crowd_plots(argv):
    """
    NAME
        make_crowd_plots

    PURPOSE
        Given stage1 and stage2 bureau pickles, this script produces the
        4 plots currently planned for the crowd section of the SW system
        paper.

    COMMENTS

    FLAGS
        -h                Print this message

    INPUTS
        bureau.pickle

    """

    # ------------------------------------------------------------------

    try:
       opts, args = getopt.getopt(argv,"hc",["help"])
    except getopt.GetoptError, err:
       print str(err) # will print something like "option -a not recognized"
       print make_crowd_plots.__doc__  # will print the big comment above.
       return

    for o,a in opts:
       if o in ("-h", "--help"):
          print make_crowd_plots.__doc__
          return
       else:
          assert False, "unhandled option"

    # Check for pickles in array args:
    if len(args) >= 1:
        bureau_paths = [args[i] for i in range(len(argv))]
        bureau_names = ['_'.join(bp.split('_')[:-1]) for bp in bureau_paths]
        for bureau_path in bureau_paths:
            print "make_crowd_plots: illustrating behaviour captured in",\
            "bureau file '%s'\n"%bureau_path
    else:
        print make_crowd_plots.__doc__
        return

    output_directory = './'

    # ------------------------------------------------------------------

    d={}

    # Read in bureau objects and summarize data:
    for idx, bureau_path in enumerate(bureau_paths):

        # Initialize the dictionary with arrays for this bureau
        d["final_skill{0}".format(idx+1)] = np.array([]),
        d["contribution{0}".format(idx+1)] = np.array([]),
        d["experience{0}".format(idx+1)] = np.array([]),
        d["effort{0}".format(idx+1)] = np.array([]),
        d["information{0}".format(idx+1)] = np.array([]),
        d["early_skill{0}".format(idx+1)] = np.array([]),
        d["final_skill_all{0}".format(idx+1)] = np.array([]),
        d["contribution_all{0}".format(idx+1)] = np.array([]),
        d["experience_all{0}".format(idx+1)] = np.array([]),
        d["effort_all{0}".format(idx+1)] = np.array([]),
        d["information_all{0}".format(idx+1)] = np.array([])

        # Open the bureau
        bureau = swap.read_pickle(bureau_path, 'bureau')
        print "make_crowd_plots: agent numbers for %s: %i"%(bureau_path, 
                                                            len(bureau.list()))
        
        # make lists by going through agents
        N_early = 10
        
        # Populate the bureau data dictionary with summary stats
        for ID in bureau.list():
            agent = bureau.member[ID]
            
            d['final_skill_all{0}'.format(idx+1)] = \
                            np.append(d['final_skill_all{0}'.format(idx+1)], 
                                      agent.traininghistory['Skill'][-1])
            d['information_all{0}'.format(idx+1)] = \
                            np.append(d['information_all{0}'.format(idx+1)], 
                                      agent.testhistory['I'].sum())
            d['effort_all{0}'.format(idx+1)] = \
                            np.append(d['effort_all{0}'.format(idx+1)], 
                                      agent.N-agent.NT)
            d['experience_all{0}'.format(idx+1)] = \
                            np.append(d['experience_all{0}'.format(idx+1)], 
                                      agent.NT)
            d['contribution_all{0}'.format(idx+1)] = \
                            np.append(d['contribution_all{0}'.format(idx+1)], 
                                      agent.testhistory['Skill'].sum())


            if agent.NT < N_early:
                continue

            d['final_skill{0}'.format(idx+1)] = \
                                np.append(d['final_skill{0}'.format(idx+1)], 
                                          agent.traininghistory['Skill'][-1])
            d['information{0}'.format(idx+1)] = \
                                np.append(d['information{0}'.format(idx+1)], 
                                          agent.testhistory['I'].sum())
            d['effort{0}'.format(idx+1)] = \
                                np.append(d['effort{0}'.format(idx+1)], 
                                          agent.N-agent.NT)
            d['experience{0}'.format(idx+1)] = \
                                np.append(d['experience{0}'.format(idx+1)], 
                                          agent.NT)
            try:
                d['early_skill{0}'.format(idx+1)] = \
                                np.append(d['early_skill{0}'.format(idx+1)], 
                                        agent.traininghistory['Skill'][N_early])
            except: pdb.set_trace()
            d['contribution{0}'.format(idx+1)] = \
                                np.append(d['contribution{0}'.format(idx+1)], 
                                          agent.testhistory['Skill'].sum())


        # Report
        print "make_crowd_plots: mean stage 1 volunteer effort =",\
            phr(np.mean(d['effort_all{0}'.format(idx+1)]))

        print "make_crowd_plots: mean stage 1 volunteer experience =",\
            phr(np.mean(d['experience_all{0}'.format(idx+1)]))

        print "make_crowd_plots: mean stage 1 volunteer contribution =",\
            phr(np.mean(d['contribution_all{0}'.format(idx+1)])),"bits"

        print "make_crowd_plots: mean stage 1 volunteer skill =",\
            phr(np.mean(d['final_skill_all{0}'.format(idx+1)]),ndp=2),"bits"


    ######################################################################


    ######################################################################
    # Plot 1.1 and 1.2: cumulative distributions of contribution and skill
    ######################################################################
    
    #-------------------------
    # 1.1 Contribution
    #-------------------------

    plt.figure(figsize=(10,8),dpi=100)

    # All Stage 1 volunteers:
    # --------------------------------------------------------------------
    cumulativecontribution1_all = \
                            np.cumsum(np.sort(d['contribution_all1'])[::-1])
    totalcontribution1_all = cumulativecontribution1_all[-1]
    Nv1_all = len(cumulativecontribution1_all)

    print "make_crowd_plots: %i stage 1 volunteers contributed %.2f bits"\
        %(Nv1_all, totalcontribution1_all)

    # Fraction of total contribution, fraction of volunteers:
    cfrac1_all = cumulativecontribution1_all / totalcontribution1_all
    vfrac1_all = np.arange(Nv1_all) / float(Nv1_all)

    index = np.where(cfrac1_all > 0.9)[0][0]

    print "make_crowd_plots: %.2f%% the volunteers - %i people - contributed"\
        " 90%% of the information at Stage 1"%(100*vfrac1_all[index], 
                                             int(Nv1_all*vfrac1_all[index]))

    print "make_crowd_plots: total amount of information generated at "\
        "stage 1: %.2f bits"%np.sum(d['information_all1'])


    # plot fractions...
    plt.plot(vfrac1_all, cfrac1_all, '-b', linewidth=4, 
             label='CFHTLS Stage 1: All Volunteers')


    # Experienced Stage 1 volunteers (normalize to all!):
    # --------------------------------------------------------------------
    cumulativecontribution1 = np.cumsum(np.sort(d['contribution1'])[::-1])
    totalcontribution1 = cumulativecontribution1[-1]
    Nv1 = len(cumulativecontribution1)

    print "make_crowd_plots: %i experienced stage 1 volunteers contributed "\
        "%.2f bits"%(Nv1, totalcontribution1)

    # Fraction of total contribution (from experienced volunteers), 
    # fraction of (experienced) volunteers:
    cfrac1 = cumulativecontribution1 / totalcontribution1_all
    vfrac1 = np.arange(Nv1) / float(Nv1)

    index = np.where(cfrac1 > 0.9)[0][0]

    print "make_crowd_plots: %.2f%% of the experienced volunteers - %i "\
        "people - contributed 90%% of the information at Stage 1"\
        %(100*vfrac1[index], int(Nv1*vfrac1[index]))

    # plot experienced fractions... 
    plt.plot(vfrac1, cfrac1, '--b', linewidth=4, 
             label='CFHTLS Stage 1: Experienced Volunteers')

    plt.xlabel('Fraction of Volunteers')
    plt.ylabel('Fraction of Total Contribution')
    plt.xlim(0.0, 0.21)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right')

    pngfile = output_directory+'%s_crowd_contrib_cumul.png'%bureau_names[idx]
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()
    print "make_crowd_plots: cumulative contribution plot saved to "+pngfile


    #-----------------------
    # 1.2 Skill
    #-----------------------

    plt.figure(figsize=(10,8),dpi=100)

    # All Stage 1 volunteers:
    #---------------------------------------------------------------------
    cumulativeskill1_all = np.cumsum(np.sort(d['final_skill_all1'])[::-1])
    totalskill1_all = cumulativeskill1_all[-1]
    Nv1_all = len(cumulativeskill1_all)

    print "make_crowd_plots: %i stage 1 volunteers possess %.2f bits worth "\
        "of skill"%(Nv1_all, totalskill1_all)

    # Fraction of total skill, fraction of volunteers:
    cfrac1_all = cumulativeskill1_all / totalskill1_all
    vfrac1_all = np.arange(Nv1_all) / float(Nv1_all)

    index = np.where(vfrac1_all > 0.2)[0][0]

    print "make_crowd_plots: %.2f%% of the skill possessed by the (20%%) most "\
        "skilled %i people"%(100*cfrac1_all[index],
                             int(Nv1_all*vfrac1_all[index]))

    plt.plot(vfrac1_all, cfrac1_all, '-b', linewidth=4, 
             label='CFHTLS Stage 1: All Volunteers')


    # Experienced Stage 1 volunteers (normalize to all!):
    #---------------------------------------------------------------------
    cumulativeskill1 = np.cumsum(np.sort(d['final_skill1'])[::-1])
    totalskill1 = cumulativeskill1[-1]
    Nv1 = len(cumulativeskill1)

    print "make_crowd_plots: %i experienced stage 1 volunteers possess "\
        "%.2f bits worth of skill"%(Nv1, totalskill1)

    # Fraction of total skill (from experienced volunteers), 
    # fraction of (experienced) volunteers:
    cfrac1 = cumulativeskill1 / totalskill1_all
    vfrac1 = np.arange(Nv1) / float(Nv1)

    index = np.where(vfrac1 > 0.2)[0][0]

    print "make_crowd_plots: %.2f%% of the skill possessed by the (20%%) "\
        "most skilled %i people"%(100*cfrac1[index],int(Nv1*vfrac1[index]))

    plt.plot(vfrac1, cfrac1, '--b', linewidth=4, 
             label='CFHTLS Stage 1: Experienced Volunteers')
        
    plt.xlabel('Fraction of Volunteers')
    plt.ylabel('Fraction of Total Skill')
    plt.xlim(0.0, 0.21)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='upper left')

    pngfile = output_directory+'%s_crowd_skill_cumul.png'%bureau_names[idx]
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()
    print "make_crowd_plots: cumulative skill plot saved to "+pngfile


    # ------------------------------------------------------------------

    # Plot #3: corner plot for 5 variables of interest; 

    # For some runs, Effort is the same as Experience so the 'effort_all' 
    # column is all zeros --> that can't go into corner plotter!
    if np.any(d['effort_all1']>0.):
        X = np.vstack((d['effort_all1'],
                       d['experience_all1'], 
                       d['final_skill_all1'], 
                       d['contribution_all1'], 
                       d['information_all1'])).T
        labels=['log(Effort)', 'log(Experience)', 'log(Skill)', 
                'log(Contribution)', 'log(Information)']
        ranges=[(0.,5.5),(0.,4.),(-5.,0.),(-4.,5.),(-9.,3.)]
    else:
        X = np.vstack((d['experience_all1'], 
                       d['final_skill_all1'], 
                       d['contribution_all1'], 
                       d['information_all1'])).T
        labels=['log(Experience)', 'log(Skill)', 
                'log(Contribution)', 'log(Information)']
        ranges=[(0.,4.),(-5.,0.),(-4.,5.),(-9.,3.)]


    pos_filter = True

    #"""
    for Xi in X.T:
        pos_filter *= Xi > 0

    filter1 = d['final_skill_all1'] > 1e-7
    print "make_crowd_plots: filtering out %i final_skill_all1 < 1e-7"\
        %(len(filter1)-np.sum(filter1))
    pos_filter *= filter1

    filter2 = d['contribution_all1'] > 1e-11
    print "make_crowd_plots: filtering out %i contribution_all1 < 1e-11"\
        %(len(filter2)-np.sum(filter2))
    pos_filter *= filter2

    X = np.log10(X[pos_filter])
    #"""

    pngfile = output_directory+'%s_all_crowd_properties.png'%bureau_names[idx]

    figure = corner.corner(X, labels=labels, range=ranges,
                           plot_datapoints=True, smooth=None,
                           fill_contours=True, levels=[0.68, 0.95, 0.99],
                           color='purple')
    figure.savefig(pngfile)

    print "make_crowd_plots: corner plot saved to "+pngfile

# ======================================================================

def phr(x,ndp=1):
    fmt = "%d" % ndp
    fmt = '%.'+fmt+'f'
    return fmt % x

# ======================================================================

if __name__ == '__main__':
    make_crowd_plots(sys.argv[1:])

# ======================================================================
