#!/usr/bin/env python
# ======================================================================

# from __future__ import division
# from skimage import io
from subprocess import call
# from colors import blues_r

import sys,getopt,numpy as np

import matplotlib
# Force matplotlib to not use any Xwindows backend:
matplotlib.use('Agg')

# Fonts, latex:
matplotlib.rc('font',**{'family':'serif', 'serif':['TimesNewRoman']})
matplotlib.rc('text', usetex=True)

from matplotlib import pyplot as plt

bfs,sfs = 20,16
params = { 'axes.labelsize': bfs,
            'text.fontsize': bfs,
          'legend.fontsize': bfs,
          'xtick.labelsize': sfs,
          'ytick.labelsize': sfs}
plt.rcParams.update(params)

import swap

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
        --cornerplotter   $CORNERPLOTTER_DIR
        stage1_bureau.pickle
        stage2_bureau.pickle

    OUTPUTS
        Various png plots.

    EXAMPLE

    BUGS
        - Code is not tested yet...

    AUTHORS
        This file is part of the Space Warps project, and is distributed
        under the MIT license by the Space Warps Science Team.
        http://spacewarps.org/

    HISTORY
      2013-05-17  started Baumer & Davis (KIPAC)
      2013-05-30  opts, docs Marshall (KIPAC)
    """

    # ------------------------------------------------------------------

    try:
       opts, args = getopt.getopt(argv,"hc",["help","cornerplotter"])
    except getopt.GetoptError, err:
       print str(err) # will print something like "option -a not recognized"
       print make_crowd_plots.__doc__  # will print the big comment above.
       return

    cornerplotter_path = ''
    resurrect = False

    for o,a in opts:
       if o in ("-h", "--help"):
          print make_crowd_plots.__doc__
          return
       elif o in ("-c", "--cornerplotter"):
          cornerplotter_path = a+'/'
       else:
          assert False, "unhandled option"

    # Check for pickles in array args:
    if len(args) >= 1:
        paths = args
        for path in paths:
            print "make_crowd_plots: illustrating behaviour captured in ",
            "bureau file %s"%path
    else:
        print make_crowd_plots.__doc__
        return

    cornerplotter_path = cornerplotter_path+'CornerPlotter.py'
    output_directory = './'

    # ------------------------------------------------------------------

    d={}

    # Read in bureau objects and summarize data:
    for idx, bureau_path in enumerate(paths):

        # [SPACEWARPS ONLY] -- Stage 2 stuff dependent on Stage 1 stuff 
        if idx == 1:
            # stage 1 skill of stage 2 classifiers:
            d['final_skill_oldagent'] = np.array([])
            d['new_s2_contribution'] = np.array([])
            d['new_s2_skill'] = np.array([])
            d['new_s2_effort'] = np.array([])
            d['new_s2_information'] = np.array([])

            bureau2 = swap.read_pickle(bureau_path,'bureau')

            stage2_veteran_members = []
            for ID in bureau2.list():
                if ID in bureau.list():
                    stage2_veteran_members.append(ID)
                print "make_crowd_plots: ",len(stage2_veteran_members), \
                    " volunteers stayed on for Stage 2 from Stage 1"

                oldagent = bureau.member[ID]
                d['final_skill_oldagent'] = np.append(\
                                        d['final_skill_oldagent'], 
                                        oldagent.traininghistory['Skill'][-1])

                agent = bureau2.member[ID]
                if agent.name not in stage2_veteran_members:
                    d['new_s2_contribution']=np.append(d['new_s2_contribution'],
                                            agent.testhistory['Skill'].sum())
                    d['new_s2_skill'] = np.append(d['new_s2_skill'],
                                            agent.traininghistory['Skill'][-1])
                    d['new_s2_effort'] = np.append(d['new_s2_effort'], 
                                            agent.N-agent.NT)
                    d['new_s2_information'] = np.append(d['new_s2_information'],
                                            agent.testhistory['I'].sum())



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
                                        append(agent.N-agent.NT))
            d['experience_all{0}'.format(idx+1)] = \
                            np.append(d['experience_all{0}'.format(idx+1)], 
                                      agent.NT)
            d['contribution_all{0}'.format(idx+1)] = \
                            np.append(d['contribution_all{0}'.format(idx+1)], 
                                        agent.testhistory['Skill'].sum())


            if agent.NT > N_early:
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
            d['early_skill{0}'.format(idx+1)] = \
                                np.append(d['early_skill{0}'.format(idx+1)], 
                                        agent.traininghistory['Skill'][N_early])
            d['contribution{0}'.format(idx+1)] = \
                                np.append(d['contribution{0}'.format(idx+1)], 
                                          agent.testhistory['Skill'].sum())


        # Report
        print "make_crowd_plots: mean stage 1 volunteer effort = ",\
            phr(np.mean(d['effort_all{0}'.format(idx+1)]))
        print "make_crowd_plots: mean stage 1 volunteer experience = ",\
            phr(np.mean(d['experience_all{0}'.format(idx+1)]))
        print "make_crowd_plots: mean stage 1 volunteer contribution = ",\
            phr(np.mean(d['contribution_all{0}'.format(idx+1)])),"bits"
        print "make_crowd_plots: mean stage 1 volunteer skill = ",\
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
        %(Nv1_all, phr(totalcontribution1_all))

    # Fraction of total contribution, fraction of volunteers:
    cfrac1_all = cumulativecontribution1_all / totalcontribution1_all
    vfrac1_all = np.arange(Nv1_all) / float(Nv1_all)

    index = np.where(cfrac1_all > 0.9)[0][0]

    print "make_crowd_plots: %.2f% of the volunteers -%i people - ",\
        "contributed 90% of the information at Stage 1"\
        %(phr(100*vfrac1_all[index]), int(Nv1_all*vfrac1_all[index]))

    print "make_crowd_plots: total amount of information generated at ",\
        "stage 1: %.2f bits"%phr(np.sum(information_all))


    # plot fractions...
    plt.plot(vfrac1_all, cfrac1_all, '-b', linewidth=4, 
             label='CFHTLS Stage 1: All Volunteers')


    # Experienced Stage 1 volunteers (normalize to all!):
    # --------------------------------------------------------------------
    cumulativecontribution1 = np.cumsum(np.sort(d['contribution1'])[::-1])
    totalcontribution1 = cumulativecontribution1[-1]
    Nv1 = len(cumulativecontribution1)

    print "make_crowd_plots: %i experienced stage 1 volunteers contributed ",\
        "%.2f bits"%(Nv1, phr(totalcontribution1))

    # Fraction of total contribution (from experienced volunteers), 
    # fraction of (experienced) volunteers:
    cfrac1 = cumulativecontribution1 / totalcontribution1_all
    vfrac1 = np.arange(Nv1) / float(Nv1)

    index = np.where(cfrac1 > 0.9)[0][0]

    print "make_crowd_plots: %.2f% of the experienced volunteers - %i ",\
        "people - contributed 90% of the information at Stage 1"\
        %(phr(100*vfrac1[index]), int(Nv1*vfrac1[index]))

    # plot experienced fractions... 
    plt.plot(vfrac1, cfrac1, '--b', linewidth=4, 
             label='CFHTLS Stage 1: Experienced Volunteers')


    # [SPACEWARPS ONLY] All Stage 2 volunteers:
    # --------------------------------------------------------------------
    if idx >= 1:
        cumulativecontribution2_all = \
                            np.cumsum(np.sort(d['contribution_all2'])[::-1])
        totalcontribution2_all = cumulativecontribution2_all[-1]
        Nv2_all = len(cumulativecontribution2_all)

        print "make_crowd_plots: %i stage 2 volunteers contributed %.2f bits"\
            %(Nv2_all, phr(totalcontribution2_all))

        # Fraction of total contribution, fraction of volunteers:
        cfrac2_all = cumulativecontribution2_all / totalcontribution2_all
        vfrac2_all = np.arange(Nv2_all) / float(Nv2_all)

        index = np.where(cfrac2_all > 0.9)[0][0]

        print "make_crowd_plots: %.2f% of the volunteers - %i people - ",\
            "contributed 90% of the information at Stage 2"\
            %(phr(100*vfrac2_all[index]), int(Nv2_all*vfrac2_all[index]))
        
        print "make_crowd_plots: total amount of information generated at ",\
            "stage 2 = %.2f bits"%phr(np.sum(information_all2)),

        # plot stage 2 fractions...
        plt.plot(vfrac2_all, cfrac2_all, '#FF8000', linewidth=4, 
                 label='CFHTLS Stage 2: All Volunteers')


    plt.xlabel('Fraction of Volunteers')
    plt.ylabel('Fraction of Total Contribution')
    plt.xlim(0.0, 0.21)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='lower right')

    pngfile = output_directory+'crowd_contrib_cumul.png'
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

    print "make_crowd_plots: %i stage 1 volunteers possess %.2f bits worth ",\
        "of skill"%(Nv1_all,phr(totalskill1_all))

    # Fraction of total skill, fraction of volunteers:
    cfrac1_all = cumulativeskill1_all / totalskill1_all
    vfrac1_all = np.arange(Nv1_all) / float(Nv1_all)

    index = np.where(vfrac1_all > 0.2)[0][0]

    print "make_crowd_plots: %.2f% of the skill possessed by the (20%) most ",\
        "skilled %i people"%(phr(100*cfrac1_all[index]),
                             int(Nv1_all*vfrac1_all[index]))

    plt.plot(vfrac1_all, cfrac1_all, '-b', linewidth=4, 
             label='CFHTLS Stage 1: All Volunteers')


    # Experienced Stage 1 volunteers (normalize to all!):
    #---------------------------------------------------------------------
    cumulativeskill1 = np.cumsum(np.sort(d['final_skill1'])[::-1])
    totalskill1 = cumulativeskill1[-1]
    Nv1 = len(cumulativeskill1)

    print "make_crowd_plots: %i experienced stage 1 volunteers possess ",\
        "%.2f bits worth of skill"%(Nv1,phr(totalskill1))

    # Fraction of total skill (from experienced volunteers), 
    # fraction of (experienced) volunteers:
    cfrac1 = cumulativeskill1 / totalskill1_all
    vfrac1 = np.arange(Nv1) / float(Nv1)

    index = np.where(vfrac1 > 0.2)[0][0]

    print "make_crowd_plots: %.2f% of the skill possessed by the (20%) ",\
        "most skilled %i people"%(phr(100*cfrac1[index]),int(Nv1*vfrac1[index]))

    plt.plot(vfrac1, cfrac1, '--b', linewidth=4, 
             label='CFHTLS Stage 1: Experienced Volunteers')


    # [SPACEWARPS ONLY] All Stage 2 volunteers:
    #---------------------------------------------------------------------
    if idx >= 1:
        cumulativeskill2_all = np.cumsum(np.sort(d['final_skill_all2'])[::-1])
        totalskill2_all = cumulativeskill2_all[-1]
        Nv2_all = len(cumulativeskill2_all)
        
        print "make_crowd_plots: %i stage 2 volunteers possess %.2f bits ",\
            "worth of skill"%(Nv2_all,phr(totalskill2_all))
        
        # Fraction of total skill, fraction of volunteers:
        cfrac2_all = cumulativeskill2_all / totalskill2_all
        vfrac2_all = np.arange(Nv2_all) / float(Nv2_all)
        
        index = np.where(vfrac2_all > 0.2)[0][0]
        
        print "make_crowd_plots: %.2f% of the skill possessed by the (20%) ",\
            "most skilled %i people"%(phr(100*cfrac2_all[index]),
                                      int(Nv2_all*vfrac2_all[index]))
        
        
        plt.plot(vfrac2_all, cfrac2_all, '#FF8000', linewidth=4, 
                 label='CFHTLS Stage 2: All Volunteers')
        
    plt.xlabel('Fraction of Volunteers')
    plt.ylabel('Fraction of Total Skill')
    plt.xlim(0.0, 0.21)
    plt.ylim(0.0, 1.0)
    plt.legend(loc='upper left')

    pngfile = output_directory+'crowd_skill_cumul.png'
    plt.savefig(pngfile, bbox_inches='tight')
    plt.close()
    print "make_crowd_plots: cumulative skill plot saved to "+pngfile


    # ------------------------------------------------------------------

    # Plot #2: is final skill predicted by early skill?

    """ Commented out as we left this out of the paper.
    N = len(final_skill)
    prodigies_final_skill = final_skill[np.where(early_skill > 0.1)]
    Nprodigies = len(prodigies_final_skill)
    mean_prodigies_skill = np.mean(prodigies_final_skill)
    Ngood_prodigies = len(np.where(prodigies_final_skill > 0.05)[0])
    print "make_crowd_plots: the",Nprodigies,"-",phr(100*Nprodigies/N),"% - of experienced stage 1 volunteers who have early skill > 0.1 go on to attain a mean final skill of",phr(mean_prodigies_skill,ndp=2)
    print "make_crowd_plots: with",phr(100*Ngood_prodigies/Nprodigies),"% of them remaining at skill 0.05 or higher"

    plt.figure(figsize=(10,8),dpi=100)
    plt.xlim(-0.02,0.25)
    plt.ylim(-0.02,0.8)
    plt.xlabel('Early Skill, $\langle I \\rangle_{j<10}$ / bits')
    plt.ylabel('Final Skill, $\langle I \\rangle_{j=N_{\\rm T}}$ / bits')
    # Point size prop to contribution!
    # size = 400.0
    size = 20 + 0.01*contribution
    plt.scatter(early_skill,final_skill,s=size,color='blue',alpha=0.4)
    plt.plot((0.1, 0.1), (0.05, 0.8),color='black',ls='--')
    plt.plot((0.1, 0.25), (0.05, 0.05),color='black',ls='--')
    # pngfile = output_directory+'early_vs_final_skill.png'
    pngfile = output_directory+'early_vs_final_skill.pdf'
    plt.savefig(pngfile, bbox_inches='tight')
    print "make_crowd_plots: skill-skill plot saved to "+pngfile
    """

    # ------------------------------------------------------------------

    # Plot #3: corner plot for 5 variables of interest; 
    # stage1 = blue shaded 
    # stage2 = orange outlines.

    X = np.vstack((d['effort_all1'], 
                   d['experience_all1'], 
                   d['final_skill_all1'], 
                   d['contribution_all1'], 
                   d['information_all1'])).T

    pos_filter = True

    for Xi in X.T:
        pos_filter *= Xi > 0
    pos_filter *= final_skill_all > 1e-7
    pos_filter *= contribution_all > 1e-11
    X = np.log10(X[pos_filter])

    comment = 'log(Effort), log(Experience), log(Skill), log(Contribution), ',\
              'log(Information)\n{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8},',\
              '{9}'.format(X[:, 0].min(), X[:, 0].max(),
                           X[:, 1].min(), X[:, 1].max(),
                           X[:, 2].min(), X[:, 2].max(),
                           X[:, 3].min(), X[:, 3].max(),
                           X[:, 4].min(), X[:, 4].max(),)

    np.savetxt(output_directory+'volunteer_analysis1.cpt', X, header=comment)

    input1 = output_directory+'volunteer_analysis1.cpt,blue,shaded'


    # [SPACEWARPS ONLY]
    #-------------------------------------------------------------------------
    if idx >= 1:
        X = np.vstack((d['effort_all2'],
                       d['experience_all2'], 
                       d['final_skill_all2'], 
                       d['contribution_all2'], 
                       d['information_all2'])).T

        pos_filter = True
        for Xi in X.T:
            pos_filter *= Xi > 0
        pos_filter *= final_skill_all2 > 1e-7
        pos_filter *= contribution_all2 > 1e-11
        X = np.log10(X[pos_filter])

        np.savetxt(output_directory+'volunteer_analysis2.cpt', X, 
                   header=comment)
        input2 = output_directory+'volunteer_analysis2.cpt,orange,shaded'


    pngfile = output_directory+'all_skill_contribution_experience_education.png'

    # call([cornerplotter_path,'-o',pngfile,input1,input2])
    call([cornerplotter_path,'-o',pngfile,input1])

    print "make_crowd_plots: corner plot saved to "+pngfile

    """
    #-------------------------------------------------------------------
    # [SPACEWARPS ONLY]
    # ------------------------------------------------------------------
    # Plot #4: stage 2 -- new volunteers vs. veterans: contribution.

    # PJM: updated 2014-09-03 to show stage 1 vs 2 skill, point size shows effort.

    # plt.figure(figsize=(10,8))
    plt.figure(figsize=(8,8),dpi=100)
    # plt.xlim(-10.0,895.0)
    plt.xlim(-0.02,0.85)
    plt.ylim(-0.02,0.85)
    # plt.xlabel('Stage 2 Contribution $\sum_k \langle I \\rangle_k$ / bits')
    plt.xlabel('Stage 1 Skill $\langle I \\rangle_{j=N_{\\rm T}}$ / bits')
    plt.ylabel('Stage 2 Skill $\langle I \\rangle_{j=N_{\\rm T}}$ / bits')

    # size = 0.5*effort2
    # size = 20 + 10*information2
    size = 10 + 5*contribution2
    # plt.scatter(contribution2, final_skill2, s=size, color='blue', alpha=0.4)
    # plt.scatter(contribution2, final_skill2,         color='blue', alpha=0.4, label='Veteran volunteers from Stage 1')
    plt.scatter(final_skill1, final_skill2, s=size, color='blue', alpha=0.4, label='Veteran volunteers from Stage 1')
    # plt.scatter(final_skill1, final_skill2,         color='blue', alpha=0.4, label='Veteran volunteers from Stage 1')

    # size = 0.5*new_s2_effort
    # size = 20 + 10*new_s2_information
    size = 10 + 5*new_s2_contribution
    # plt.scatter(new_s2_contribution, new_s2_skill,s = size, color='#FFA500', alpha=0.4)
    # plt.scatter(new_s2_contribution, new_s2_skill,          color='#FFA500', alpha=0.4, label='New Stage 2 volunteers')
    new_s1_skill = new_s2_skill.copy()*0.0 # All had zero skill at stage 1, because they didn't show up!
    plt.scatter(new_s1_skill, new_s2_skill,s = size, color='#FFA500', alpha=0.4, label='New Stage 2 volunteers')
    # plt.scatter(new_s1_skill, new_s2_skill,          color='#FFA500', alpha=0.4, label='New Stage 2 volunteers')

    Nvets = len(contribution2)
    Nnewb = len(new_s2_contribution)
    N = Nvets + Nnewb
    totalvets = np.sum(contribution2)
    totalnewb = np.sum(new_s2_contribution)
    total = totalvets + totalnewb
    print "make_crowd_plots: total contribution in Stage 2 was",phr(total),"bits by",N,"volunteers"

    x0,y0,w0,z0 = np.mean(final_skill1),np.mean(final_skill2),np.mean(contribution2),np.mean(effort2)
    l = plt.axvline(x=x0,color='blue',ls='--')
    l = plt.axhline(y=y0,color='blue',ls='--')
    print "make_crowd_plots: ",Nvets,"stage 1 veteran users (",phr(100*Nvets/N),"% of the total) made",phr(100*totalvets/total),"% of the contribution"
    print "make_crowd_plots: the average stage 1 veteran had skill1, skill2, contribution, effort = ",phr(x0,ndp=2),phr(y0,ndp=2),phr(w0),int(z0)

    x0,y0,w0,z0 = np.mean(new_s1_skill),np.mean(new_s2_skill),np.mean(new_s2_contribution),np.mean(new_s2_effort)
    l = plt.axvline(x=x0,color='#FFA500',ls='--')
    l = plt.axhline(y=y0,color='#FFA500',ls='--')
    print "make_crowd_plots: ",Nnewb,"new users (",phr(100*Nnewb/N),"% of the total) made",phr(100*totalnewb/total),"% of the contribution"
    print "make_crowd_plots: the average stage 2 newbie had skill1, skill2, contribution, effort = ",phr(x0,ndp=2),phr(y0,ndp=2),phr(w0),int(z0)

    lgnd = plt.legend(loc='upper right')
    lgnd.legendHandles[0]._sizes = [30]
    lgnd.legendHandles[1]._sizes = [30]

    # pngfile = output_directory+'stage2_veteran_contribution.png'
    pngfile = output_directory+'stage2_veteran_contribution.pdf'
    plt.savefig(pngfile, bbox_inches='tight')
    print "make_crowd_plots: newbies vs veterans plot saved to "+pngfile

    # ------------------------------------------------------------------

    print "make_crowd_plots: all done!"
    """

    return

# ======================================================================

def phr(x,ndp=1):
    fmt = "%d" % ndp
    fmt = '%.'+fmt+'f'
    return fmt % x

# ======================================================================

if __name__ == '__main__':
    make_crowd_plots(sys.argv[1:])

# ======================================================================
