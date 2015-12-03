import cPickle
import numpy as np
import matplotlib.pyplot as plt
import pdb
import datetime
from astropy.table import Table



### Take 2.

ancillary = Table.read('GZ2assets_Nair_Morph2_urls.fits')


startdate = datetime.datetime(2009,2,26,0,0,0)
delta = datetime.timedelta(days=10)

prefix = 'GZ2_sup_0.75'
rejected_gals, detected_gals = {}, {}

for i in range(7):
    datestring = datetime.datetime.strftime(startdate, '%Y-%m-%d_%H:%M:%S')
    trunk = prefix+'_'+datestring
    detected_file = '%s/%s_detected_catalog.txt'%(trunk, trunk)
    retired_file = '%s/%s_retired_catalog.txt'%(trunk, trunk)

    dets = Table.read(detected_file, format='ascii')
    rets = Table.read(retired_file, format='ascii')
    
    random_dets = dets[np.random.random_integers(0,len(dets),10)]
    random_rets = rets[np.random.random_integers(0,len(rets),10)]

    detected_gals[datestring+'_zooids']=random_dets['zooid']
    rejected_gals[datestring+'_zooids']=random_rets['zooid']

    det_urls, ret_urls = [], []
    for d, r in zip(random_dets['zooid'], random_rets['zooid']):
        d_loc = np.where(d == ancillary['name'])
        r_loc = np.where(r == ancillary['name'])
        det_urls.append(ancillary['urls'][d_loc][0])
        ret_urls.append(ancillary['urls'][r_loc][0])

    detected_gals[datestring+'_urls']=np.array(det_urls)
    rejected_gals[datestring+'_urls']=np.array(ret_urls)
    
    '''
    print "urls for detected objects, %s:"%datestring
    for url in det_urls:
        print url
    print "urls for rejected objects, %s:"%datestring
    for url in ret_urls:
        print url
    '''
    startdate = startdate+delta


