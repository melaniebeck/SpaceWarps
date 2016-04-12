import numpy as np
import pdb

truth = {'SMOOTH':1, 'NOT':0, 'UNKNOWN':-1}

def Nair_label_SMOOTH_NOT(Nair_subjects):
    """ 
    Nair_subject should have stucture like a dictionary and should
       include field names from NairAbrahamMorphology.cat

    "truth" label is determined as follows:

    - TType = -1, 0 correspond to S0+ and S0a galaxies which are too difficult
          to discern. These are thus not labelled one way or the other
    - Nair_subjects with TType >= 1 (and flag != 2) correspond to galaxies 
          which are >= Sa (I can't remember why the flag is there... )
    - Nair_subjects with TType <= -2 (and dist <=2) correspond to galaxies
          which are E0 to S0-, very early-type. The dist <=2 excludes objects
          which could have tidal tails or merger features which users might 
          instead classify as "Featured/Disk"
    """
    label = []
    #for Nair_subject in Nair_subjects:
    for Nair_subject in Nair_subjects:

        if Nair_subject['JID'][0] == 'J':
            if Nair_subject['TType'] <= -2 and Nair_subject['dist'] <= 2: 
                #label = truth['SMOOTH']
                label.append(truth['SMOOTH'])
            elif Nair_subject['TType'] >= 1 and Nair_subject['flag'] != 2:
                #label = truth['NOT']
                label.append(truth['NOT'])
            else:
                #label = truth['UNKNOWN']
                label.append(truth['UNKNOWN'])
        else: 
            #label = truth['UNKNOWN']
            label.append(truth['UNKNOWN'])

    print "Nair labels complete."
    return np.array(label)


def GZ2_label_SMOOTH_NOT(GZ2_subjects):
    """ 
    GZ2_subject should have stucture like a dictionary and should
       include field names GZ2assets_Nair_Morph_zoo2Main.fits or similar

    "truth" label is determined as follows:

    - Only the top level task is considered and a majority vote on the 
         debiased answer is used to determine class label
    """
    label = []

    smooth = GZ2_subjects['t01_smooth_or_features_a01_smooth_debiased']
    disk = GZ2_subjects['t01_smooth_or_features_a02_features_or_disk_debiased']
    star = GZ2_subjects['t01_smooth_or_features_a03_star_or_artifact_debiased']
    
    #for GZ2_subject in GZ2_subjects:
    for sm,d,st in zip(smooth,disk,star):
        
        majority = np.max([sm, d, st])
        
        if sm == majority:
            #label = truth['SMOOTH']
            label.append(truth['SMOOTH'])
            
        elif d == majority:
            #label = truth['NOT']
            label.append(truth['NOT'])
            
        else: 
            #label = truth['UNKNOWN']
            label.append(truth['UNKNOWN'])

    print "GZ2 labels complete."
    return np.array(label)


def Expert_label_SMOOTH_NOT(expert_subjects):
    # When aggregating the results I labeled SMOOTH as 0 and NOT as 1;
    # in COMPLETE contrast with everything else I was doing. 
    # Should probably fix this at some point...
    label = []

    for expert_subject in expert_subjects:
        #if 'sum_of_votes' in expert_subject.colnames:
        if expert_subject['sum_of_votes'] >= 3:
            #label = truth['NOT']
            label.append(truth['NOT'])
            
        elif expert_subject['sum_of_votes'] < 3:
            #label = truth['SMOOTH']
            label.append(truth['SMOOTH'])
            
        else: 
            #label = truth['UNKNOWN']
            label.append(truth['UNKNOWN'])
            
        #else: 
        #    #label = truth['UNKNOWN']
        #    label.append(truth['UNKNOWN'])

    print "Expert labels complete."
    return np.array(label)

def select_Nair_subjects(candidates):
    """
    Candidates must be dictionary or json structured and have a field
        called "JID" (from NairAbrahamMorphology.cat) which denotes the 
        object ID for the Nair-Abraham catalog
    """
    indices = []
    for idx, candidate in enumerate(candidates):
        if candidate['JID'][0] == 'J':
            indices.append(idx)

    return candidates[indices]

def find_indices(x, y):
    """
    Find the index of every element of y in x 
    """
    # indices that would sort x (length x)
    index = np.argsort(x)

    sorted_x = x[index]

    # index of every element of y in sorted x (length y)
    # can't plug this into either x or y: 
    # it's too short for x (being length y) and it's x's indices
    # which are too large for y
    sorted_index = np.searchsorted(sorted_x, y)

    # this returns the value of "index" at each index, "sorted_index"
    # so if sorted_index = [273404, 273405, ...]
    # you'll get the value in "index" at THOSE indices
    # since "index" is itself a list of indices, you're actually getting
    # "unsorted" indices in x that correspond to matching values in y
    # (length y) ; x[yindex] = matching values in y
    yindex = np.take(index, sorted_index)
    mask = x[yindex] != y

    result = np.ma.array(yindex, mask=mask)

    return result

def main():
    
    from astropy.table import Table, join

    # use this to test overlap of assets with expert sample
    #assets = Table.read('GZ2assets_Nair_Morph.fits')
    #assets = Table.read('gz2_assetstable_better.txt',format='ascii')

    # This one has the "category" columns like gz2_..._better.fits with the 
    # regular columns of GZ2assets_Nair... but WITHOUT stripe82_coadd objects
    # RIGHT JOINED assets+gzmain (all the obj in assets; w or w/o match in gz)
    assets = Table.read('GZ2ASSETS_NAIR_MORPH_MAIN.fits')
    #expert = Table.read('expert_sample.fits')

    #test1 = join(expert,assets, keys='JID')
    #test2 = join(expert,gzmain,keys='JID')

    # Construct a metadata table for use in SWAP/MACHINE 
    # Should include at LEAST these columns:
    # [SDSS objid, Nair JID, RA, DEC, Rp, C, A, G, M20, elipt, Nair Label, 
    # GZ2 user label, Expert label, MLsample]
    #pdb.set_trace()
    metadata = Table(names=('SDSS_id', 'JID', 'asset_id',
                            'stripe82','extra_original', 
                            'ra', 'dec', 'Rp', 
                            'C', 'A', 'G', 'M20', 'E',
                            'Rpflag', 'bflag', 
                            'total_classifications', 'TType'),
                     data= (assets['name'], assets['JID'], assets['id'],
                            assets['stripe82'], assets['extra_original'], 
                            assets['ra'], assets['dec'], assets['Rp'], 
                            assets['C'], assets['A'], assets['G'],
                            assets['M20'], assets['elipt'],
                            assets['Rpflag'], assets['bflag'], 
                            assets['total_classifications'], assets['TType']))

                    # still need to add: jpeg name, truth labels, urls12
                    #'int16','int16','int16','S1184')
    """
    Length of metadata should match that of ASSETS!! 
       - Some won't have measured morphology parameters
       - Some won't have Nair_labels (SMOOTH/NOT labels based on Nair TType)
       - Most won't have Expert_labels
       - 
    """
    # Feed these functions the ASSETS table (not gzmain)
    # If the necessary info for that subject doesn't exist, XXX_label = -1
    GZ2_label = GZ2_label_SMOOTH_NOT(assets)
    Nair_label = Nair_label_SMOOTH_NOT(assets)
    Expert_label = Expert_label_SMOOTH_NOT(assets)
    print "Finished labelling"
    
    # find which indices Expert_label has in gzmain
    #result = find_indices(assets['JID'], expert['JID'])

    # Create array with length assets, all -1 (UNKNOWN)
    #Expert_label = np.full_like(assets['stripe82'], truth['UNKNOWN'])
    #Expert_label[result] = Expert_label_short
    #print "Extended 'Expert' label array"

    # Create array to hold jpeg names 
    #jpegs = np.full_like(assets['urls_dr12'], None)
    #jpegs[result] = expert['image_name']
    #print "Extended 'image_name' array"

    # add these columns to metadata
    metadata['MLsample'] = 'valid'  # Change to test afterwards (initialize S5)
    metadata['SWAP_prob'] = 0.3     # Current prior 
    metadata['GZ2_label'] = GZ2_label
    metadata['Nair_label'] = Nair_label
    metadata['Expert_label'] = Expert_label
    #metadata['image_name'] = jpegs
    #metadata['urls12'] = assets['urls_dr12']

    print "Added additional columns to metadata"

    metadata['MLsample'] = 'test'
    pdb.set_trace()

    try:
        metadata.write('metadata_ground_truth_labels.fits')
    except:
        "Filename already exists. Overwrite?"
        pdb.set_trace()
    
    # Make a metadata pickle for GZ:Express
    import cPickle
    F = open('GZ2_testML2_metadata.pickle','wb')
    cPickle.dump(metadata,F,protocol=2)
    F.close()
    pdb.set_trace()



if __name__ == '__main__':
    main()
