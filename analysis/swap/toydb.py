# ======================================================================

import numpy as np

import datetime,sys

# ======================================================================

class ToyDB(object):
    """
    NAME
        ToyDB

    PURPOSE
        Make a toy database, and serve up data just like Mongo does
        (except using standard dictionaries).

    COMMENTS

    INITIALISATION
        From scratch.

    METHODS AND VARIABLES
        ToyDB.get_classification()

    BUGS

    AUTHORS
      This file is part of the Space Warps project, and is distributed
      under the MIT license by the Space Warps Science Team.
      http://spacewarps.org/

    HISTORY
      2013-04-18  Started Marshall (Oxford)
    """

# ----------------------------------------------------------------------------

    def __init__(self,pars=None):

        self.client = "Highest bidder"
        self.db = "Hah! This is all fake"

        try: self.surveysize = int(pars['surveysize']) # No. of subjects
        except: self.surveysize = 400

        try: self.trainingsize = int(pars['trainingsize']) # No. of subjects
        except: self.trainingsize = 400

        try: self.population = int(pars['population']) # No. of classifiers
        except: self.population = 100

        try: self.enthusiasm = int(pars['enthusiasm']) # Mean no. of classifications per person
        except: self.enthusiasm = 40

        try: self.prior = int(pars['lensrate']) # Mean no. of classifications per person
        except: self.prior = 1e-3 # Probability of an image containing a lens

        try: self.difficulty = int(pars['difficulty']) # Mean no. of classifications per person
        except: self.difficulty = 0.5

        self.classifiers = self.populate('classifiers')

        self.trainingset = self.populate('subjects',category='training')
        self.testset = self.populate('subjects',category='test')
        self.subjects = self.trainingset + self.testset

        self.classifications = self.populate('classifications')

        return None

# ----------------------------------------------------------------------------

    def __str__(self):
        return 'database of %d Toy classifications' % (self.size())

# ----------------------------------------------------------------------------
# Generate various tables in the database:

    def populate(self,things,category=None):

        array = []

        if things == 'classifiers':
            # Store their name, and other information:
            for k in range(self.population):
                classifier = {}
                classifier['Name'] = 'MRbeck'+str(k)
                classifier['count'] = 0
                classifier['truePL'],classifier['truePD'] = self.draw_from_Beta2D()
                array.append(classifier)


        elif things == 'subjects':

            if category not in ['training','test']:
                print "ToyDB: confused by category "+category
                sys.exit()

            if category == 'training': Nj = self.trainingsize
            if category == 'test': Nj = self.surveysize

            for j in range(Nj):
                subject = {}
                ID = category+'Image'+str(j)
                ZooID = 'ASWXXXX'+str(j)
                subject['ID'] = ID
                subject['ZooID'] = ID

                subject['category'] = category

                if subject['category'] == 'training':
                    if np.random.rand() > 0.5:
                        subject['kind'] = 'sim'
                        subject['truth'] = 'SMOOTH'
                    else:
                        subject['kind'] = 'dud'
                        subject['truth'] = 'NOT'

                elif subject['category'] == 'test':
                    subject['kind'] = 'test'
                    subject['truth'] = 'UNKNOWN'
                    # But we do actually need to know what this is!
                    if np.random.rand() < self.prior:
                        subject['strewth'] = 'SMOOTH'
                    else:
                        subject['strewth'] = 'NOT'

                png = ID+'_gri.png'
                subject['location'] = 'http://toy.org/standard/'+png

                array.append(subject)


        elif things == 'classifications':

            count = 0
            N = self.population*self.enthusiasm
            for i in range(N):
                classification = {}
                t = self.pick_one('epochs')

                classifier = self.pick_one('classifiers')
                classification['Name'] = classifier['Name']

                subject = self.pick_one('subjects',classifier=classifier)

                classification['updated_at'] = t
                classification['ID'] = subject['ID']
                classification['category'] = subject['category']
                classification['kind'] = subject['kind']
                classification['truth'] = subject['truth']
                classification['result'] = \
                  self.make_classification(subject=subject,classifier=classifier)

                array.append(classification)
                count += 1

                # Count up to 74 in dots:
                if count == 1: sys.stdout.write('SWAP: ')
                elif np.mod(count,int(N/73.0)) == 0: sys.stdout.write('.')
                elif count == N: sys.stdout.write('\n')
                sys.stdout.flush()

        return array

# ----------------------------------------------------------------------------
# Random selection of something from its list:

    def pick_one(self,things,classifier=None):

        if things == 'classifiers':

            # Distribution of number of classifications peaks at low N.
            # Suppose mean number is 40; exponential distribution with this
            # mean?

            # Original uniform distribution:
            # k = int(self.population*np.random.rand())

            j = 0
            while (j == 0):

                # Exponential distribution for Nk with mean = enthusiasm:
                Nk = int(np.random.exponential(scale=self.enthusiasm)) + 1
                # This is the number of classifications made by the kth classifier
                # Nk = 1 is the most likely number, it's the bin with the most
                # classifiers in it. Now find where in the ordered sequence of
                # classifiers we are, by drawing randomly from this bin of the
                # histogram.

                mu = self.enthusiasm
                KK = self.population

                # First count the classifiers who will do less than Nk
                # classifications:
                i = int((KK/mu)*(mu - (Nk-1.0+mu)*np.exp(-1.0*(Nk-1.0)/mu)))

                # Now draw a classifier from the Nk column:
                j = int((KK*Nk/(mu*mu))*np.exp(-1.0*Nk/mu))
                k = i + int(np.random.rand() * j)

                # BUG: this should really be a draw without replacement.
                # No matter - its good enough for a sim.

                # print "Classifier: Nk,i,j,k = ",Nk,i,j,k
                if j == 0 or k >= KK:
                    # print "Rejecting Classifier: Nk,i,j,k = ",Nk,i,j,k
                    j = 0

            something = self.classifiers[k]


        elif things == 'subjects':

            # Here, we have to emulate the stream. What the classifier
            # is shown depends on what they have already seen!

            j = classifier['count'] + 1
            level = int(j/20.0) + 1
            alpha = self.difficulty

            training_rate = 2.0 / (5.0*2.0**(alpha*(level - 1)))

            if np.random.rand() < training_rate:
                j = int(len(self.trainingset)*np.random.rand())
                something = self.trainingset[j]
            else:
                j = int(len(self.testset)*np.random.rand())
                something = self.testset[j]

        elif things == 'epochs':

            day = int(14*np.random.rand()) + 1
            hour = int(24*np.random.rand())
            minute = int(60*np.random.rand())
            second = int(60*np.random.rand())
            something = datetime.datetime(2013, 4, day, hour, minute, second, 0)

        return something

# ----------------------------------------------------------------------------
# Use the hidden confusion matrix of each toy classifier to classify
# the subject provided:

    def make_classification(self,subject=None,classifier=None):

        # If all toy classifiers were equally skilled, we could ignore them,
        # and just use constant P values:
        # PL = 0.9
        # PD = 0.8
        # Instead, we use the classifier's own PD and PL:

        if subject['category'] == 'training':
            truth = subject['truth']
        elif subject['category'] == 'test':
            truth = subject['strewth']

        if truth == 'SMOOTH':
            if np.random.rand() < classifier['truePL']: word = 'SMOOTH'
            else: word = 'NOT'

        elif truth == 'NOT':
            if np.random.rand() < classifier['truePD']: word = 'NOT'
            else: word = 'SMOOTH'

        return word

# ----------------------------------------------------------------------------
# Return a tuple of the key quantities:

    def digest(self,C):

        return str(C['updated_at']),C['Name'],C['ID'],C['ZooID'],C['category'],C['kind'],C['result'],C['truth'],C['location']

# ----------------------------------------------------------------------------
# Return a batch of classifications, defined by a time range - either
# claasifications made 'since' t, or classifications made 'before' t:

    def find(self,word,t):

       batch = []

       if word == 'since':

            for classification in self.classifications:
                if classification['updated_at'] > t:
                    batch.append(classification)

       elif word == 'before':

            for classification in self.classifications:
                if classification['updated_at'] < t:
                    batch.append(classification)

       else:
           print "ToyDB: error, cannot find classifications '"+word+"' "+str(t)

       return batch

# ----------------------------------------------------------------------------
# Return the size of the classification table:

    def size(self):

        return len(self.classifications)

# ----------------------------------------------------------------------------
# Draw a PL,PD pair from circular beta PDF:

    def draw_from_Beta2D(self):

        # First draw a radius:
        alpha = 1.0/0.25
        beta = 1.0/0.8
        R = 0.48*np.random.beta(alpha,beta,size=1)

        # Now draw an azimuthal angle:
        alpha = 1.0/0.25
        beta = 1.0/0.3
        phi = -0.5*np.pi + 1.5*np.pi*np.random.beta(alpha,beta,size=1)

        # Convert to PL and PD, fuzzy up, and truncate:
        Pmax = 0.99
        PL = 0.5 + R*np.cos(phi) + 0.03*np.random.randn()
        PL[np.where(PL > Pmax)] = Pmax
        PD = 0.5 + R*np.sin(phi) + 0.03*np.random.randn()
        PD[np.where(PD > Pmax)] = Pmax

        return PL[0],PD[0]

# ======================================================================

if __name__ == '__main__':

    db = ToyDB()

    # Select all classifications that were made before t1.
    # Note the greater than operator ">".
    # Make sure we catch them all!
    t1 = datetime.datetime(1978, 2, 28, 12,0, 0, 0)

    batch = db.find('since',t1)

    # Now, loop over classifications, digesting them.

    # Batch has a next() method, which returns the subsequent
    # record, or we can execute a for loop as follows:
    count = 0
    for classification in batch:

        items = db.digest(classification)

        # Check we got all 7 items:
        if items is not None:
            if len(items) != 7:
                print "oops! ",items[:]
            else:
                # Count classifications
                count += 1

    # Good! Whole database dealt with.
    print "Counted ",count," classifications, that each look like:"
    print items[:]

