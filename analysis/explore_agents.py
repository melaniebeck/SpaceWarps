import cPickle
import numpy as np
import matplotlib.pyplot as plt
import pdb

F = open('GZ2_sup_0.75_bureau.pickle','rb')
bureau = cPickle.load(F)
F.close()
agent_ids = bureau.list()


total_number_of_users = bureau.size()

number_of_training_seen, number_of_images_seen = [], []
smooths, nots = [], []
number_correct_smooths, number_correct_nots = [], []
total_number_correct, total_number_incorrect = [], []  
more_than_none, right_away = [], [] 

for ID in agent_ids:
    agent = bureau.member[ID]
    
    traininghistory = agent.traininghistory['ActuallyItWas']
    classifyhistory = agent.traininghistory['ItWas']

    # look for any training images in the training history (1 or 0)
    training = np.where(traininghistory!=-1)[0]

    # how many training images did they see? 
    # want to determine the average/fraction so keep track for each 
    # take the number in this list as the number of people who have seen
    # at least ONE training image
    number_of_images_seen.append(len(traininghistory))
    number_of_training_seen.append(len(training))

    if len(training) > 0:
        more_than_none.append(len(training))

        # how many saw one "right away" (within 5 images)?
        right_away.append(len(np.where(traininghistory[:5] != 1)[0]))

        # how many did they get correct? 
        smooth_subjects = np.where(traininghistory==1)[0]
        not_subjects = np.where(traininghistory==0)[0]

        smooths.append(len(smooth_subjects))
        nots.append(len(training) - len(smooth_subjects))

        smooth_correct = (traininghistory[smooth_subjects] == 
                        classifyhistory[smooth_subjects]).tolist().count(True)
        not_correct = (traininghistory[not_subjects] == 
                        classifyhistory[not_subjects]).tolist().count(True)
        total = smooth_correct + not_correct

        number_correct_smooths.append(smooth_correct)
        number_correct_nots.append(not_correct)
        total_number_correct.append(total)
        total_number_incorrect.append(len(training) - total)

    else:
        pdb.set_trace()


print "%5.3f users saw at least one training image:"%(len(more_than_none)*1.0/total_number_of_users)
print "%i users saw at least 1 training image within the first 5 images"%len(right_away)

binsize = 1

fig = plt.figure(figsize=(8,8))
ax1 = fig.add_subplot(311)

bins = np.arange(np.min(number_of_training_seen), 
                 np.max(number_of_training_seen)+binsize, binsize)
# How many training images do users see? 
ax1.hist(number_of_training_seen, bins=bins, histtype='stepfilled',
         color='green', alpha=0.9, label='Seen')

# Of those, what's the distribution of "right answers"?
ax1.hist(total_number_correct, bins=bins, histtype='stepfilled', 
         color='yellow', alpha=0.6, label='Correct')

ax1.legend(loc='upper right')

ax1.set_xlim(0,20)
ax1.set_xlabel("Number of Training Subjects")
ax1.set_ylabel("Number of Users")

#------------------------------------------------------------------------
ax3 = fig.add_subplot(312)

training_fraction = np.array(number_of_training_seen)*1.0/\
                    np.array(number_of_images_seen)
bin_frac = np.arange(np.nanmin(training_fraction),
                     np.nanmax(training_fraction)+0.02,0.02)
ax3.hist(training_fraction, bins=bin_frac, histtype='stepfilled')
ax3.set_xlabel('Fraction of Training Images')
ax3.set_ylabel('Number of Users')

#------------------------------------------------------------------------
ax2 = fig.add_subplot(313)

binsize = 0.05
smooth_fraction = np.array(number_correct_smooths)*1.0/np.array(smooths)
bins_smooth = np.arange(np.nanmin(smooth_fraction), 
                        np.nanmax(smooth_fraction)+binsize, binsize)

not_fraction = np.array(number_correct_nots)*1.0/np.array(nots)
bins_not =  np.arange(np.nanmin(not_fraction), 
                      np.nanmax(not_fraction)+binsize, binsize)

correct_fraction = np.array(total_number_correct)*1.0/np.array(more_than_none)
bins_correct = np.arange(np.nanmin(correct_fraction), 
                      np.nanmax(correct_fraction)+binsize, binsize)

# What's the dist of the fraction of users who got smooths right? 
#ax2.hist(correct_fraction, bins=bins_correct, histtype='stepfilled', 
#         color='yellow', label='Total')
ax2.hist(smooth_fraction, bins=bins_smooth, histtype='stepfilled', color='blue',
         alpha=0.5, label='Smooths')
ax2.hist(not_fraction, bins=bins_not, histtype='stepfilled', color='red', 
         alpha=0.5, label='Nots')

ax2.axvline(x=np.nanmean(smooth_fraction),ls='--',color='b')
ax2.axvline(x=np.nanmean(not_fraction),ls='--',color='r')

ax2.set_xlabel("Fraction Correct for Population")
ax2.set_ylabel("Number of Users")
ax2.legend(loc='upper left')

plt.tight_layout()

plt.savefig('user_analysis.png')

plt.show()

pdb.set_trace()
