from collections import namedtuple        
import json
from pprint import pprint
import sys
import numpy as np
import math

def getCoOrdinates(myInstance):
    MainList=[]
    sublist = []
    slope1 = calculateSlope((myInstance["x0"],myInstance["y0"]),(myInstance["x1"],myInstance["y1"]))
    slope2 = calculateSlope((myInstance["x0"],myInstance["y0"]),(myInstance["x2"],myInstance["y2"]))
    slope3 = calculateSlope((myInstance["x0"],myInstance["y0"]),(myInstance["x3"],myInstance["y3"]))
    slope = min(slope1,slope2,slope3)
    x1 = []
    x1.append(int(float(myInstance["x0"])))
    x1.append( int(float(myInstance["y0"])))
    sublist.append(x1)
    x2 = []
    x2.append(int(float(myInstance["x1"])))
    x2.append(int(float(myInstance["y1"])))
    sublist.append(x2)
    x3 = []
    x3.append(int(float(myInstance["x2"])))
    x3.append(int(float(myInstance["y2"])))
    sublist.append(x3)
    x4 = []
    x4.append(int(float(myInstance["x3"])))
    x4.append(int(float(myInstance["y3"])))
    sublist.append(x4)
    MainList.append(sublist)
    return slope,MainList

def calculateSlope(p1,p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return math.degrees((ang1 - ang2) % (2 * np.pi))

def getFeature():
    feature = {}
    feature['type'] = "Feature"
    feature['NameAfterDictionary'] = " "
    feature['NameBeforeDictionary'] = " "
    feature['ImageId'] = " "
    feature['DictionaryWordSimilarity'] = " "
    feature['TesseractCost'] = " "
    feature['SameMatches']= " "
    return feature
    
    
def convert(filename,pathname):
    import os
    text_file = filename
    print(text_file)
    data = json.load(open(text_file))
    text_data = data["text_lines"]
    featureList = []
    for myInstance in text_data:
        slope,MainList = getCoOrdinates(myInstance)
        geometry = {}
        geometry['type'] = 'Polygon'
        geometry['coordinates'] = MainList
        feature = getFeature()
        feature['geometry'] = geometry
        feature['inclination'] = slope
        #print(json.dumps(feature))
        featureList.append(feature)
    geoJson = {}
    geoJson['type'] = 'type'
    geoJson['features'] = featureList
    output_path = os.path.join(pathname, 'geoJson1.json')
    with open(output_path, 'w') as outfile:
        json.dump(geoJson,outfile)
    return  json.dumps(geoJson)

def main():
    convert('result.json','/home/sandeep/Desktop/TextDetection/EAST/jsonreader_check')

if __name__ == '__main__':
    main()