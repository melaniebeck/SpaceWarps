# ======================================================================
import swap
import numpy as np
import os,sys,datetime,pdb

try: import MySQLdb as mdb
except:
    print "MySQLdb: MySQLdb is not installed. You can still practise though --really?"
    # sys.exit()

# ======================================================================

class MySQLdb_ML(object):
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
            con = mdb.connect('localhost', 'root', '8croupier!', 'gz2')
        except:
            print "MySQLdb: couldn't connect to a MySQL DB with that name"
            sys.exit()

        with con:
            self.cur = con.cursor(mdb.cursors.DictCursor)

        return None

    def find(self):
        query = ("select * from asset_morph")
        self.cur.execute(query)

        subjects = self.cur.fetchall()
        return subjects


# ----------------------------------------------------------------------------
# Return a tuple of the key quantities, given a cursor pointing to a 
# record in the classifications table:

    def digest(self,metadata):
        
        # Which subject are we looking at? 
        try: ID = metadata['id']
        except: ID = '0000'

        try: ZooID = metadata['name']
        except: ZooID = '0000'

        # Does this subject have morphological metadata?
        if metadata['Rp'] > 0:
            morph = {'C':metadata['C'], 'A':metadata['A'], 
                     'G':metadata['G'], 'M20':metadata['M20'],
                     'E':metadata['elipt']}
        else:
            morph = {}
            
        sample = 'test'
        
        items = str(ID), str(ZooID), sample, morph
        return items[:]


# ----------------------------------------------------------------------------
# Return the size of the classification table:

    def size(self):
        return self.cur.rowcount

# ----------------------------------------------------------------------------

    def cleanup(self):
    
        try: self.cur.close()
        except: pass
        try: self.con.close()
        except: pass
        del self.cur

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
