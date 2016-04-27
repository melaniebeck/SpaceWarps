## SWAP has been modified to perform retrospective simulations on Galaxy Zoo 2

SWAP.py or SWAPSHOP.py run the SWAP code on GZ2 in single or batch modes respectively.

### Getting Started
GZ2 classifications are stored in an MySQL database. `swap/mysqldb.py` was created to pull classifications from this database instead of the Space Warps `swap/mongodb.py`.
In order to reduce the time of each SQL query, tables in the GZ2 db were joined before running SWAP. The script for the particular arrangement can be found in `prepare_gz2sql.py`. 

### SWAP Workflow
SWAP has been modified to run on a "daily" basis. A parameter called "increment" was added to the config file. This controls the size of the timestep in units of days. The start and end days are still provided but the code will now only collect from the database those classifications made in the increment timestep. At the end of the SWAP run, the date is automatically updated so that the update.config will start classifications on the next "day". This updating will continue until the config file has updated to the point where the start and end days are the same.  

For each night, SWAP creates a slew of output files (report is always True): 

    SURVEY_Y-D-H_00:00:00_candidate_catalog.txt     # any subject which has been classified
    SURVEY_Y-D-H_00:00:00_candidates.txt        
    SURVEY_Y-D-H_00:00:00_detected_catalog.txt      # any subject which has crossed the acceptance threshold
    SURVEY_Y-D-H_00:00:00_dud_catalog.txt       
    SURVEY_Y-D-H_00:00:00_retired_catalog.txt       # any subject which has crossed the rejectance threshold
    SURVEY_Y-D-H_00:00:00_sim_catalog.txt
    SURVEY_Y-D-H_00:00:00_training_false_negatives.txt
    SURVEY_Y-D-H_00:00:00_training_false_positives.txt
    SURVEY_Y-D-H_00:00:00_training_true_positives.txt

Additionally, SWAP requires a metadata pickle, the name of which can be predefined in the config (but is currently called `GZ2_testML2_metadata.pickle`.) This file must already exist and is not created by the code! It is  essentially the link between `SWAP.py` and `MachineClassifier.py` and contains metadata for all GZ2 subjects necessary for the machine classifiers to train. 

Specifically, the file contains the features which the machines train on (morphology indicators measured from the pixel values of the original FITS files for the galaxy images), the original labels from the GZ2 published data, and a tag that specifies that galaxy as part of the *train*, *valid*, or *test* sample. All tags start as *test* (except for a predefined validation sample). When an image crosses either the rejection or acceptance thresholds in SWAP, its tag flips from *test* to *train*.  [ISSUE: In order to make this more compatible with original SWAP, need to tuck this all away in one of the `swap/*.py` supporting modules.]

---

### Machine Classifications
`MachineClassifier.py` always reads the `update.config` file. There are several parameters which now serve only to inform the machine on what to do. As said above, MC also requires the metadata pickle. This module is by NO MEANS complete yet and will not run smoothly. 

Here's what it currently does: 
 * reads `update.config`
 * reads in the metadata pickle
 * selects out the validation sample and the training sample based on the tags in the metadata pickle
 * performs cross validation with the training sample to determine appropriate paramters for the machine classifier
 * creates an **agent** for the machine which tracks the training history and the confusion matrix produced by the trained machine on the validation set. 

LOTS more stuff still has to be done on this... 


### Exploring the output
There are a series of `explore_*.py` scripts that look at the output of the SWAP/Machine simulations. These require some of the output files described above (those with comments next to them) in addition to the `SURVEY_*.pickle` files. 


### What needs to be done [really should create issues for these...]:
* Determine how the Machine retires (threshold); flag retired galaxies so they are no longer processed through SWAP (This is being developed/explored in `explore_machine.py`)
* Once the above is implemented, run the first LIVE test of SWAP + MachineClassifier (tests so far have had MC in 'offline' mode.
* Try: Instead of using candidate and rejected catalogs - try detected and rejected catalogs? Within a few days there are so many classifications being processed by SWAP that the training sample becomes huge and the test sample miserably small.
* Try: Setting aside a strictly fixed test sample of XXX subjects (if they end up being classified, don't use them in the machine classifier)
* Try:  Break training and test samples up by redshift? magnitude? color? (test this separately?)


### Experiments to run
A slew of simulations (with and without MC) need to be run to explore the parameter space of SWAP. Parameters of interest include the acceptance and rejectance thresholds, the prior, and the initialPL and initialPD of user-agents
* Are # classifications relatively immune to initialPL and initialPD values? Vary PL/PD between .55 - .8
* Does it matter if PL and PD are the same or different?
* How drastically do classifications change when rejectance/acceptance thresholds change?
* What affect does the initial prior have? (currently set at 0.3)

Work should also be done on other questions in the GZ2 decision tree. 
* Bar or Not?  (task 3)
* Spiral arms or Not? (task 4)
* Bulge or Not? (task 5)
* Edge on or Not? (task 2)



