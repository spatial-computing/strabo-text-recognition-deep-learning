from collections import namedtuple
import json
from pprint import pprint
import sys
import numpy as np
import math


def getCoOrdinates(myInstance):
    MainList = []
    sublist = []
    slope1 = calculateSlope(
        (myInstance["x0"], myInstance["y0"]), (myInstance["x1"], myInstance["y1"]))
    slope2 = calculateSlope(
        (myInstance["x0"], myInstance["y0"]), (myInstance["x2"], myInstance["y2"]))
    slope3 = calculateSlope(
        (myInstance["x0"], myInstance["y0"]), (myInstance["x3"], myInstance["y3"]))
    slope = max(slope1, slope2, slope3)

    slopefinal = getfinalSlope(myInstance)

    x1 = []
    x1.append(int(float(myInstance["x0"])))
    x1.append(0 - int(float(myInstance["y0"])))
    sublist.append(x1)
    x2 = []
    x2.append(int(float(myInstance["x1"])))
    x2.append(0-int(float(myInstance["y1"])))
    sublist.append(x2)
    x3 = []
    x3.append(int(float(myInstance["x2"])))
    x3.append(0-int(float(myInstance["y2"])))
    sublist.append(x3)
    x4 = []
    x4.append(int(float(myInstance["x3"])))
    x4.append(0-int(float(myInstance["y3"])))
    sublist.append(x4)
    x5 = []
    x5.append(int(float(myInstance["x0"])))
    x5.append(0 - int(float(myInstance["y0"])))
    sublist.append(x5)
    MainList.append(sublist)
    return slopefinal, MainList

def getfinalSlope(myInstance1):
    (x0,y0) = myInstance1["x0"], myInstance1["y0"]
    (x1,y1) = myInstance1["x1"], myInstance1["y1"]
    (x2,y2) = myInstance1["x2"], myInstance1["y2"]
    (x3,y3) = myInstance1["x3"], myInstance1["y3"]

    x0_x1 = lengthOfline((x0,y0),(x1,y1))
    #print(x0_x1)
    x1_x2 = lengthOfline((x1,y1),(x2,y2))
    #print(x1_x2)
    x2_x3 = lengthOfline((x2,y2),(x3,y3))
    #print(x2_x3)

    if(x0_x1>=x1_x2 and x0_x1>=x2_x3):
        #print("x0_x1")
        return calculateSlope((x0,y0),(x1,y1))
    if(x1_x2>=x2_x3 and x1_x2>=x0_x1):
        #print("x1_x2")
        return calculateSlope((x1,y1),(x2,y2))
    if(x2_x3>=x0_x1 and x2_x3>=x1_x2):
        #print("x2_x3")
        return calculateSlope((x2,y2),(x3,y3))
    

def lengthOfline(a1,a2):
    (x0,y0) = a1
    (x1,y1) = a2
    dist = math.hypot(x1 - x0, y1 - y0)
    return dist

def calculateSlope(p1, p2):
    
    #slope = (p2[1] - p1[1])/(p2[0] - p1[0])
    angle = np.rad2deg(np.arctan2((p2[1] - p1[1]),(p2[0] - p1[0])))

    return angle


def getFeature():
    global count
    feature = {}
    feature['type'] = "Feature"
    feature['NameAfterDictionary'] = " "
    feature['NameBeforeDictionary'] = " "
    feature['ImageId'] = count
    feature['DictionaryWordSimilarity'] = " "
    feature['TesseractCost'] = " "
    feature['SameMatches'] = " "
    count = count + 1
    return feature


count = 0


def convert(filename, pathname):
    import os
    text_file = filename
    print(text_file)
    data = json.load(open(text_file))
    text_data = data["text_lines"]
    featureList = []
    for myInstance in text_data:
        slope, MainList = getCoOrdinates(myInstance)
        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = MainList
        feature = getFeature()
        feature['geometry'] = geometry
        feature['inclination'] = slope
        # print(json.dumps(feature))
        featureList.append(feature)
    geoJson = {}
    geoJson['type'] = 'FeatureCollection'
    geoJson['features'] = featureList
    output_path = os.path.join(pathname, 'geoJson1.json')
    with open(output_path, 'w') as outfile:
        json.dump(geoJson, outfile)
    return json.dumps(geoJson)


'''
def main():
    convert('result.json','/home/sandeep/Desktop/TextDetection/EAST/jsonreader_check')

if __name__ == '__main__':
    main()
'''
