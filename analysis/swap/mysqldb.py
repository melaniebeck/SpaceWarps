# ======================================================================
### THIS SCRIPT MUST BE GENERALIZED FOR ANY SUBJECT LABEL
import swap
import numpy as np
import os,sys,datetime,pdb

try: import MySQLdb as mdb
except:
    print "MySQLdb: MySQLdb is not installed."
    # sys.exit()

# ======================================================================

class MySQLdb(object):
    """
    NAME
        MySQLDB

    PURPOSE
        Interrogate an actual mysql database, and serve up data for a 
        a simple python analysis.

    COMMENTS
       I dunno right now.


    """

# ----------------------------------------------------------------------------

    def __init__(self):
        
        try: 
            # GENERISIZE THIS
            con1 = mdb.connect('localhost', 'root', '8croupier!', 'gz2')
            con2 = mdb.connect('localhost', 'root', '8croupier!', 'gz2')
        except:
            print "MySQLdb: couldn't connect to a MySQL DB with that name"
            sys.exit()

        with con1:
            self.cur1 = con1.cursor(mdb.cursors.DictCursor)
        with con2:
            self.cur2 = con2.cursor(mdb.cursors.DictCursor)

        return None


    def find(self,word,t,t2=None):

        try: print "SWAP: start time = %s and end time = %s"%(str(t),str(t2))
        except: pass

        if word == 'since':
            # GENERISIZE THIS
            query = ("select * from task1_full as t "
                     "where t.created_at > '%s'"%str(t))
            self.cur1.execute(query)

        elif word == 'before':
            query = ("select * from task1_full as t "
                     "and t.created_at < '%s'"%str(t))
            self.cur1.execute(query)

        elif word == 'between':
            query = ("select * from task1_full as t "
                "where t.created_at between '%s' and '%s'"%(str(t), str(t2)))
            self.cur1.execute(query)

        else:
            print "MySQLdb: error, cannot find classifications %s %s"\
                %(word, str(t))

        batch = self.cur1.fetchall()
        return batch



# ----------------------------------------------------------------------------
# Return a tuple of the key quantities, given a cursor pointing to a 
# record in the classifications table:

    def digest(self,classification,survey,subjects,method=False):
        # GENERISIZE THIS -- set up a different way to label "flavor", "kind",
        # etc? Also, standardize the names of the classification column names?

        # When was this classification made?
        t = classification['created_at'].strftime('%Y-%m-%d_%H:%M:%S')

        # Who made the classification?
        try: Name = classification['user_id']
        except: Name = '00001'

        # Which subject was classified? 
        try: ID  = classification['asset_id']
        except: ID = '00001'

        try: ZooID = classification['name']
        except: ZooID = ID

        # What did user say about this subject? 
        # --- if answer_id == 1, smooth
        # ---    answer_id == 2, features/disk
        # ---    answer_id == 3, star/artifact
        if classification['answer_id'] == 2: result = 'FEAT'
        else: result = 'NOT'

        idx = np.where(subjects['SDSS_id']==long(ZooID))
        try: subject = subjects[idx][0]
        except: pdb.set_trace()

        location = subject['external_ref']

        # No longer need to have the breakdown of Nair classification! 
        # These have been taken care of when building the metadata file
        # Use the Expert_Label if available; otherwise use Nair_label?
        # 5-4-26 Only use Epert labels now (using task1_expert table)

        if subject['Nair_label']!=-1:
            category = 'training'

            if subject['Nair_label']==0:
                flavor='lensing cluster'
                kind='sim'
                truth='NOT'

            elif subject['Nair_label']==1:
                flavor='dud'
                kind='dud'
                truth='SMOOTH'
        else:                 
            category = 'test'
            kind = 'test'
            flavor = 'test'
            truth = 'UNKNOWN'

        items = t, str(Name), str(ID), str(ZooID), category, kind, flavor,\
                result, truth, location

        return items[:]


# ----------------------------------------------------------------------------
# Return the size of the classification table:

    def size(self):
        return self.cur1.rowcount

# ----------------------------------------------------------------------------

    def cleanup(self):
    
        try: self.cur1.close()
        except: pass
        try: self.con1.close()
        except: pass
        del self.cur1

        return

# ======================================================================

if __name__ == '__main__':
   
    db = MySQLdb()
    
    #t1 = datetime.datetime(1978, 2, 28, 12,0, 0, 0)
    t1 = datetime.datetime(2009, 2, 16, 23, 59, 59)

    batch = db.find('before',t1)


    pdb.set_trace()
    print db.size

    db.cleanup()
