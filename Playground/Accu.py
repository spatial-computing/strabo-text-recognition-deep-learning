import sys
import os
import psycopg2
import itertools


def dataPreprocessing(ground_truth_file, output_file):
    conn = psycopg2.connect(host="localhost", user="sandeep", password="sandeep", database="sandeep")
    cur = conn.cursor()
    cur.execute("drop table if exists ground_truth")
    cur.execute("drop table if exists output_result")
    cur.execute("drop table if exists non_overlap")
    cur.execute("drop table if exists intermediate_result")
    conn.commit()

    # put groud_truth and output into Postgres
    os.system('ogr2ogr -f PostgreSQL PG:"dbname=sandeep user=sandeep" ' + ground_truth_file + ' -nln ground_truth')
    print('done1')
    

    os.system('ogr2ogr -f PostgreSQL PG:"dbname=sandeep user=sandeep" ' + output_file + ' -nln output_result')
    print('done2')

    # check the name of "text" column
    # text = ""
    # sql_attribute_name = "Select column_name from information_schema.columns where table_name = \'ground_truth\'"
    # cur.execute(sql_attribute_name)
    # attributes = cur.fetchall()
    # for attribute in attributes:
    #     if attribute[0] == 'txt':
    #         text = "txt"
    #     if attribute[0] == 'text':
    #         text = "text"
    #
    # if text == "":
    #     print "Something wrong with this map"
    #     print
    # else:
        # create table of overlapping boxes



    sql = "create table intermediate_result as " \
               "select a.gt_ogc_fid, a.res_ogc_fid, st_intersection(a.res_geometry, a.gt_geometry) as overlap_geometry " \
               "from (" \
               "select gt.ogc_fid as gt_ogc_fid, res.ogc_fid as res_ogc_fid, " \
               "res.wkb_geometry as res_geometry, gt.wkb_geometry as gt_geometry " \
               "from (select * from ground_truth where ST_IsValid(wkb_geometry) = true) gt " \
               "join (select * from output_result where ST_IsValid(wkb_geometry) = true) res " \
               "on st_intersects(gt.wkb_geometry, res.wkb_geometry) = true) a"
    cur.execute(sql)
    conn.commit()

    sql1 = "select * from ground_truth where ST_IsValid(wkb_geometry)  = true" 
    sql2 = "select * from output_result where ST_IsValid(wkb_geometry) = true" 
    sql3 = "select * from intermediate_result"
    cur.execute(sql1)
    rows = cur.fetchall()
    groundTruthMatches = len(rows)

    cur.execute(sql2)
    rows = cur.fetchall()
    outPutResultMatches = len(rows)

    cur.execute(sql3)
    rows = cur.fetchall()
    overlaps = len(rows)

    Precision = float(overlaps)/float(outPutResultMatches)
    recall =  float(overlaps)/float(groundTruthMatches)

    print ("Ground truth Matchs= {0}".format(groundTruthMatches))
    print ("Output results = {0}".format(outPutResultMatches))
    print ("Overlaps = {0}".format(overlaps))
    print ("Precision = {0}%".format(Precision))
    print ("Recall = {0}%".format(recall))


if __name__ == '__main__':


    ground_truth_file = sys.argv[1]
    output_file = sys.argv[2]

    dataPreprocessing(ground_truth_file, output_file)



