/*
 * To run SWAP these quantities are needed:
 *    created_at
 *    user_id
 *    asset_id
 *    name (dr7objid)
 *    answer_id
 *    external_ref
 *
 * These values are found in 4 different tables in the GZ2 DB:
 *    annotations
 *    classifications
 *    asset_classifications
 *    assets
 * 
 * Join all of these into one new table on a given task_id 
 * to reduce time on each query
 */

USE gz2

CREATE TABLE task1_expert AS (
SELECT an.*, ex.user_id, ac.asset_id, a.name, a.external_ref
FROM annotations as an
     JOIN classifications as cl
     	  ON cl.id = an.classification_id
     JOIN expert_users as ex
          ON ex.user_id = cl.user_id
     JOIN asset_classifications as ac
     	  ON ac.classification_id = cl.id
     JOIN assets as a 
     	  ON a.id = ac.asset_id
WHERE an.task_id = 1 );

/* Here's how I created task1_expert
   ALL classifications from EVERY user who has classified at least ONE 
   asset from the Expert sample 

LOAD DATA INFILE 'expert_sample_for_mysql.csv' INTO TABLE expert_sample
FIELDS TERMINATED BY ','
LINES TERMMINATED BY '\n'
IGNORE 1 LINES;

GO

SELECT DISTINCT cl.user_id INTO TABLE expert_users
FROM asset_classications as ac
JOIN expert_sample as ex
     ON ex.asset_id = ac.asset_id
JOIN classifications as cl
     ON clid = ac.classification_id

GO

ALTER TABLE expert_users ADD PRIMARY KEY (id);

GO

/* Then run the final command above. [These previous commands were done 
    on the command line.] */

/**************************************************************************\

/* This command creates the table required by SWAP.py by joining the relevant
   tables in the gz2 database to provide SWAP with the necessary information. 
   Change the WHERE statement to select the task (filter) of interest. 

CREATE TABLE task3 AS (
SELECT an.*, cl.user_id, ac.asset_id, a.name, a.external_ref
FROM annotations as an
     JOIN classifications as cl
     	  ON cl.id = an.classification_id
     JOIN asset_classifications as ac
     	  ON ac.classification_id = cl.id
     JOIN assets as a 
     	  ON a.id = ac.asset_id
WHERE an.task_id = 3 );

/* This command takes forever to execute -- probably didn't index morphology
   4/26/16 -- This table is no longer needed seeing as I restructured how and 
   where the machine accesses morphology information

CREATE TABLE asset_morph AS (
SELECT a.id, a.location, a.stripe82, a.stripe82_coadd, a.extra_original, 
       a.classification_count, m.*
FROM assets as a
     JOIN morphology as m
     	  ON m.name = a.name );
*/

/* Use GZ2assets_morph.csv 
CREATE TABLE morphology (
       name VARCHAR(20), Rp FLOAT, elipt FLOAT, C FLOAT,
       A FLOAT, G FLOAT, M20 FLOAT, Rpflag INT(1), 
       bflag INT(1), outdir INT(1));
*/

/*
				NOTES on MYSQL
-------------------------------------------------------------------------
1. LOADING DATA FROM FILE
   ** infile.csv seems to work well but you have to tell mysql about ","
   ** if there is a header, use IGNORE # LINES; 
   ** the table into which the data is to be loaded must already exist
   ** infile.csv needs to reside in /var/lib/mysql/db_name/
 
  LOAD DATA INFILE 'infile.csv' into table_name FIELDS TERMINATED BY ',';
 
2. 
