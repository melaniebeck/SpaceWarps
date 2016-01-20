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

/* This command takes forever to execute -- probably didn't index morphology */
CREATE TABLE asset_morph AS (
SELECT a.id, a.location, a.stripe82, a.stripe82_coadd, a.extra_original, 
       a.classification_count, m.*
FROM assets as a
     JOIN morphology as m
     	  ON m.name = a.name );

/* Use GZ2assets_morph.csv 
CREATE TABLE morphology (
       name VARCHAR(20), Rp FLOAT, elipt FLOAT, C FLOAT,
       A FLOAT, G FLOAT, M20 FLOAT, Rpflag INT(1), 
       bflag INT(1), outdir INT(1));
*/

/*
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
*/


/*
				NOTES on MYSQL
-------------------------------------------------------------------------
1. LOADING DATA FROM FILE
   ** infile.csv seems to work well but you have to tell mysql about ","
   ** there should NOT be a header row in the infile!
   ** the table into which the data is to be loaded must already exist
   ** infile.csv needs to reside in /var/lib/mysql/db_name/
 
  LOAD DATA INFILE 'infile.csv' into table_name FIELDS TERMINATED BY ',';
 
2. 
