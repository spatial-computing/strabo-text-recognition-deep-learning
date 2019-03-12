#!/usr/bin/python3
# coding=utf8

import sys, getopt
import json
from tesserocr import PyTessBaseAPI, PSM, OEM
import numpy
from PIL import Image

def text_recognition(input_image, input_json):
    
    data = json.load(open(input_json))
    
    features = data["features"]
    
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(input_image)
    
    img = numpy.asarray(image)
    count = 0
    
    for feat in features:
        geometry = feat["geometry"]
        coordinates = geometry["coordinates"]
        box = coordinates[0]
    
        x1 = min(box, key=lambda x:x[0])
        x1 = x1[0]
        x2 = max(box, key=lambda x:x[0])
        x2 = x2[0]
        y1 = max(box, key=lambda x:x[1])
        y1 = -1*y1[1]
        y2 = min(box, key=lambda x:x[1])
        y2 = -1*y2[1]
        if x1<0 or x2<0 or y1<0 or y2<0: 
            continue
        part_image = img[y1:y2, x1:x2, :]
        new = Image.fromarray(numpy.uint8(part_image))
       
        text = ""
        score = 0
    
        with PyTessBaseAPI(path='/usr/share/tesseract-ocr/tessdata/',psm=PSM.AUTO_OSD, oem=OEM.TESSERACT_ONLY) as api:
            api.SetImage(new)
            api.Recognize()
            text = api.GetUTF8Text()
            score = api.AllWordConfidences()
        
        feat["NameBeforeDictionary"] = text
        feat["TesseractCost"] = score
        feat["NameAfterDictionary"] = text
        
        features[count] = feat
        
        
        count = count+1
    
    data["features"] = features
    return data
    

def main(argv):
   inputfile = ''
   jsonfile = ''
   outputfile = ''

   opts, args = getopt.getopt(argv,"hi:j:o:",["ifile=","jfile=","ofile="])

   for opt, arg in opts:
      if opt == '-h':
         print ('test_recognition.py -i <Input image path> -j <Json file>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-j", "--jfile"):
         jsonfile = arg
      elif opt in ("-o", "--ofile"):
         outputfile = arg
    
   data = text_recognition(inputfile, jsonfile)
   with open(outputfile, 'w') as outfile:
        json.dump(data, outfile)
       

if __name__ == "__main__":
   main(sys.argv[1:])

