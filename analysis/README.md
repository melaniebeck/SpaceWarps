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

Additionally, MachineClassifier requires a pre-made catalog called `GZ2assets_Nair_Morph_zoo2Main.fits` which contains further metadata for GZ2 subjects which also have morphological parameters measured for them. 

### Exploring the output
There are a series of `explore_*.py` scripts that look at the output of the SWAP/Machine simulations. These require some of the output files described above (those with comments next to them) in addition to the `SURVEY_*.pickle` files. 





# Getting Started

Zooniverse classifications are stored in a Mongo database. To access raw classifications you need dumps from Mongo.  The two relevant collections are `spacewarps_subjects` and `spacewarps_classifications`.

  * `spacewarps_subjects` contains subjects and associated metadata, including the CFHTLS id.  Each subject has an id corresponding to an entry in the `spacewarps_classifications` collection
  * `spacewarps_classifications` contain all classifications that have been contributed by Zooniverse volunteers.

## Installation of Mongo

Visit the [Mongo website](http://www.mongodb.org/) for the binary installation.

## Restoring the Space Warps Database

Mongo comes with a nice command line utility to restore a database from a dump.  First request data from the Zooniverse development team.

    spacewarps-Y-m-d_H-M-S.tar.gz

Decompress it and run:
    
    # Spin up the mongo server, and provide a place for it to put stuff:
    mongod --dbpath . &
    
    # Load the database:
    mongorestore spacewarps-Y-m-d_H-M-S

Note: the server has to be running in the background for the mongo
python commands to run, but it still spits out stdout. So, it's best to
have the server running in a different terminal.

