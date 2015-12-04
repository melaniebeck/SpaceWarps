## SWAP has been modified to perform retrospective simulations on Galaxy Zoo 2

SWAP.py or SWAPSHOP.py run the SWAP code on GZ2 in single or batch modes respectively.

### Getting Started
GZ2 classifications are stored in an MySQL database. `swap/mysqldb.py` was created to pull classifications from this database instead of the Space Warps `swap/mongodb.py`.
In order to reduce the time of each SQL query, tables in the GZ2 db were joined before running SWAP. The script for the particular arrangement can be found in `prepare_gz2sql.py`. 

### SWAP Workflow
SWAP has been modified to run on a "daily" basis whereby all classifications on a given day in the original GZ2 database are selected and processed. When running with SWAPSHOP, SWAP processes each day's data after which the day is automatically updated until hitting a point defined by the user in the `last` variable on line 108 in SWAP.py. 

For each night, SWAP creates a slew of output files: 

    SURVEY_Y-D-H_00:00:00_candidate_catalog.txt     # any subject which has been classified
    SURVEY_Y-D-H_00:00:00_candidates.txt        
    SURVEY_Y-D-H_00:00:00_detected_catalog.txt      # any subject which has crossed the acceptance threshold
    SURVEY_Y-D-H_00:00:00_dud_catalog.txt       
    SURVEY_Y-D-H_00:00:00_retired_catalog.txt       # any subject which has crossed the rejectance threshold
    SURVEY_Y-D-H_00:00:00_sim_catalog.txt
    SURVEY_Y-D-H_00:00:00_training_false_negatives.txt
    SURVEY_Y-D-H_00:00:00_training_false_positives.txt
    SURVEY_Y-D-H_00:00:00_training_true_positives.txt

Additionally, SWAP requires a pre-made catalog called `GZ2assets_Nair_Morph.fits` which contains further metadata for all GZ2 subjects. Many of the above output files contain some of this data for further processing (see below).  

### Machine Classifications
`MachineClassifier.py` accepts output from SWAP which it uses as a training sample. Once the machine has learned it then runs on a test sample which consists of the remaining galaxies in GZ2, i.e. those which have not yet been classified by SWAP. This is to avoid overfitting. When running SWAPSHOP, SWAP is called first and then MachineClassifier is called. MachineClassifier can also be run "offline" whereby it cycles through each night's SWAP output and processes them in rapid succession. This is not ideal as then subjects "retired" by the Machine are not fed back into SWAP. This is still under active modification. 

For each night, MachineClassifier creats the following output files:

    SURVEY_Y-D-H_00:00:00_machine_testsample.fits   # remaining test sample for that night
    SURVEY_Y-D-H_00:00:00_machine.txt               # machine output: predictions & probabilities for the test sample

Additionally, MachineClassifier requires a pre-made catalog called `GZ2assets_Nair_Morph_zoo2Main.fits` which contains further metadata for GZ2 subjects which also have morphological parameters measured for them. For this file and the one mentioned above, please ask me. 

### Exploring the output
There are a series of `explore_*.py` scripts that look at the output of the SWAP/Machine simulations. These require some of the output files described above (those with comments next to them) in addition to the `SURVEY_*.pickle` files. 


### What needs to be done:
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



