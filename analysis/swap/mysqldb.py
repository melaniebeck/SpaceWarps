# ======================================================================
import swap
import numpy as np
import os,sys,datetime,pdb

try: import MySQLdb as mdb
except:
    print "MySQLdb: MySQLdb is not installed. You can still practise though --really?"
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
            query = ("select * from task1_full as t "
                     "where t.created_at > '%s'"%str(t))
            self.cur1.execute(query)

        elif word == 'before':
            query = ("select * from task1_full as t "
                     "where t.user_id = '142530' "
                     "and t.created_at < '%s'"%str(t))
            self.cur1.execute(query)
            #"where t.asset_id = '5507' "
            #"or t.name = '587722981742084144' "

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
        if classification['answer_id'] == 1: result = 'SMOOTH'
        else: result = 'NOT'

        idx = np.where(subjects['name']==long(ZooID))
        subject = subjects[idx][0]

        location = subject['external_ref']

        if subject['JID']: 
            category = 'training'
            if subject['TType'] <= -2 and subject['dist'] <= 2: 
                flavor = 'lensing cluster'  # so that I don't have to
                kind = 'sim'                # change stuff elsewhere...
                truth = 'SMOOTH'
            elif subject['TType'] >= 1 and subject['flag'] != 2: 
                kind = 'dud' 
                flavor = 'dud'
                truth = 'NOT'
            else:
                kind = 'test'
                flavor = 'test'
                truth = 'UNKNOWN'
        else: 
            category = 'test'
            kind = 'test'
            flavor = 'test'
            truth = 'UNKNOWN'

        """
        # If we separate SWAP and MACHINE pickles, this isn't needed!
        morphdata = {}
        # deal with morphology data
        if not np.isnan(subject['M20']):
            morphdata = {'M20':subject['M20'], 'C':subject['C'], 
                    'G':subject['G'], 'A':subject['A'], 'E':subject['elipt']}
        """

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
