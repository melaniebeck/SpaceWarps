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
