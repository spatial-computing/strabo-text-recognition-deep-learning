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
               "left join (select * from output_result where ST_IsValid(wkb_geometry) = true) res " \
               "on st_intersects(gt.wkb_geometry, res.wkb_geometry) = true) a"
    cur.execute(sql)
    conn.commit()

    sql1 = "select sum(a.area) from (select st_area(wkb_geometry) as area from " \
           "(select * from ground_truth where ST_IsValid(wkb_geometry)  = true) gt) a"
    sql2 = "select sum(a.area) from (select st_area(wkb_geometry) as area from " \
           "(select * from output_result where ST_IsValid(wkb_geometry) = true) gt) a"
    sql3 = "select sum(st_area(a.overlap_geometry)) from intermediate_result a where res_ogc_fid is not null"
    cur.execute(sql1)
    rows = cur.fetchall()
    for row in rows:
        area_ground_truth = row[0]

    cur.execute(sql2)
    rows = cur.fetchall()
    for row in rows:
        area_output_result = row[0]

    cur.execute(sql3)
    rows = cur.fetchall()
    for row in rows:
        area_overlapping = row[0]

    print ("Area of ground truth = {0}".format(area_ground_truth))
    print ("Area of output = {0}".format(area_output_result))
    print ("Area of over lapping = {0}".format(area_overlapping))
    print ("Accuracy (overlapping/groundtruth) = {0}%".format(area_overlapping / area_ground_truth * 100))


if __name__ == '__main__':


    ground_truth_file = sys.argv[1]
    output_file = sys.argv[2]

    dataPreprocessing(ground_truth_file, output_file)





