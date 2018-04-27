# -*- coding: utf-8 -*-
"""
Created on Sun Mar 18 13:37:55 2018

@author: lakshya
"""

#!/usr/bin/python3
# coding=utf8

import sys, getopt
import json
import pickle
from tesserocr import PyTessBaseAPI, PSM, OEM
import numpy
from PIL import Image, ImageDraw
import numpy as np
import codecs
from scipy.spatial import distance
import enhance
import scipy.ndimage, scipy.misc
import time

enhancer = enhance.NeuralEnhancer(loader=False)

def union(a, b):
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  angle = 0  
  
  if a[4]==0 and b[4]==0:
      angle = 0
  elif a[4]==90 and b[4]==90:
      angle = 90

  return (x, y, w, h, angle)

def intersection(a,b):
  x = max(a[0], b[0])
  y = max(a[1], b[1])
  
  w = min(a[0]+a[2], b[0]+b[2]) - x
  h = min(a[1]+a[3], b[1]+b[3]) - y
  if w<0 or h<0: return () # or (0,0,0,0) ?
  return (x, y, w, h, 0)

def combine_boxes(boxes):
     noIntersectLoop = False
     noIntersectMain = False
     posIndex = 0
     angled = True
     # keep looping until we have completed a full pass over each rectangle
     # and checked it does not overlap with any other rectangle
     while noIntersectMain == False:
         noIntersectMain = True
         posIndex = 0
         # start with the first rectangle in the list, once the first 
         # rectangle has been unioned with every other rectangle,
         # repeat for the second until done
         while posIndex < len(boxes)-1:
             noIntersectLoop = False
             while noIntersectLoop == False and len(boxes) > 1:
                a = boxes[posIndex]
                listBoxes = np.delete(boxes, posIndex, 0)
                index = 0
                for b in listBoxes:
                    #if there is an intersection, the boxes overlap
                    if intersection(a, b): 
                        newBox = union(a, b)
                        listBoxes[index] = newBox
                        boxes = listBoxes
                        noIntersectLoop = False
                        noIntersectMain = False
                        index = index + 1
                        break
                    noIntersectLoop = True
                    index = index + 1
             posIndex = posIndex + 1

     return np.array(boxes, dtype=np.int)

def merge_rectangles(input_image, input_json):
    
    data = json.load(open(input_json))
    
    features = data["features"]
    boxes = []
    
    image = Image.open(input_image)
    img = numpy.asarray(image)
    
    draw = ImageDraw.Draw(image)
    
    
    for feat in features:
        geometry = feat["geometry"]
        coordinates = geometry["coordinates"]
        box = coordinates[0]
        angle = feat["inclination"]
        inclination = 0
        
        x1 = min(box, key=lambda x:x[0])
        x1 = x1[0]
        x2 = max(box, key=lambda x:x[0])
        x2 = x2[0]
        y1 = max(box, key=lambda x:x[1])
        y1 = -1*y1[1]
        y2 = min(box, key=lambda x:x[1])
        y2 = -1*y2[1]
        
        if(abs(angle) <= 5.0 or abs(angle) >= 176.0):
            inclination = 0
        elif(abs(angle) >= 86.0 and abs(angle) <= 93.0):
            inclination = 90
        else:
            if angle > 0:
                inclination = angle
            else:
                inclination = -(180-abs(angle))
        
        box = [x1, y1, x2-x1, y2-y1, inclination]
        
        boxes.append(box)
             
    #boxes = [[3,2,3,2], [4,4,3,3]]
    
    boxes = combine_boxes(boxes)
    
    print(len(boxes))
    features = []
    count = 0
    
    for box in boxes:
        
        draw.rectangle([box[0], box[1], box[0]+box[2], box[1] + box[3]], outline="blue")
        feature_data = {}
        feature_data["inclination"] = box[4]
        feature_data["type"] = "Feature"
        feature_data["NameBeforeDictionary"] = ""
        feature_data["DictionaryWordSimilarity"] = ""
        feature_data["ImageId"] = count
        feature_data["TesseractCost"] = ""
        feature_data["NameAfterDictionary"] = ""
        feature_data["SameMatches"] = ""
        
        geometry = {}
        geometry["type"] = "Polygon"
        coordinates = []
        
        rectangle_coordinates = [[box[0],-box[1]], [box[0]+box[2], -box[1]], [box[0]+box[2], -(box[1]+box[3])], [box[0], -(box[1]+box[3])]]
        coordinates.append(rectangle_coordinates)
        
        geometry["coordinates"] = coordinates
        feature_data["geometry"] = geometry
        
        count+=1
        
        features.append(feature_data)
    
    data["features"] = features
    
#    img_name = input_image.split('/')[-1]
#    img_name = img_name.split('.')[0]
#    image.save(img_name + ".png", "PNG")    
    
    return data


def run_tesseract(image, inclinations):

    text = ""
    score = -1
    
    for i in range(0,len(inclinations)):
        
        angle = inclinations[i]
        rot_img = image.rotate(angle, expand=True)
        
#        rot_img.save(str(count)+"_"+str(angle)+".png")

        with PyTessBaseAPI(psm=PSM.AUTO_OSD, oem=OEM.LSTM_ONLY) as api:
            api.SetImage(rot_img)
            api.Recognize()
            temp_text = api.GetUTF8Text()
            temp_score = api.AllWordConfidences()
        
            if not temp_text:
                temp_score = 0
            else:
                temp_score = np.mean(temp_score)
            
            if temp_score > score:
                text = temp_text
                score = temp_score
    
        
#        with codecs.open(str(count)+"_"+str(angle)+".txt", 'w',encoding='utf8') as outfile:
#            outfile.write(text)
#            outfile.write(str(score))
#            outfile.write(str(feat["inclination"]))
    
    return text, score

def text_recognition_merge(input_image, jsonfile, super_resolution_flag):
    
    data = merge_rectangles(input_image, jsonfile)
    features = data["features"]

    image_name = input_image.split('/')[-1]    
    print(image_name[:4])

    if (image_name[:4] != 'USGS'):
        super_resolution_flag = 'no'
    
    Image.MAX_IMAGE_PIXELS = None
    image = Image.open(input_image)
    scipy_image = scipy.ndimage.imread(input_image)
    
    img = numpy.asarray(image)
    count = 0
    
    for feat in features:
        geometry = feat["geometry"]
        coordinates = geometry["coordinates"]
        box = coordinates[0]
        
        angle = feat["inclination"]
        inclinations = []    
        
        if(abs(angle) <= 5.0 or abs(angle) >= 176.0):
            inclinations = [0]
        elif(abs(angle) >= 86.0 and abs(angle) <= 93.0):
            inclinations = [90, -90]
        else:
            inclinations.append(0)
            inclinations.append(angle)
            if angle > 0:
                inclinations.append(-(180-angle))
            else:
                inclinations.append(-(180-abs(angle)))
    
        x1 = min(box, key=lambda x:x[0])
        x1 = x1[0]
        x2 = max(box, key=lambda x:x[0])
        x2 = x2[0]
        y1 = max(box, key=lambda x:x[1])
        y1 = -1*y1[1]
        y2 = min(box, key=lambda x:x[1])
        y2 = -1*y2[1]    
        
        part_image = img[y1:y2, x1:x2, :]
        new = Image.fromarray(numpy.uint8(part_image))
#        new.save(str(count)+ ".png")
        
        text = ""
        score = -1
        
        new_text = ""
        new_score = -1
        
        text, score = run_tesseract(new, inclinations)
        
        if super_resolution_flag.lower() == 'yes' or super_resolution_flag.lower() == 'combine':
            width, height = new.size
            if width < 125 and height < 125:
                if not score or (not any(c.isalpha() for c in text)) or (np.mean(score) < 70.0):
                    new = scipy_image[y1:y2, x1:x2, :]
                    new = enhancer.process(new)
#                    new.save(str(count)+ "_enhanced.png")
            
                    if super_resolution_flag.lower() == 'combine':
                        new_text, new_score = run_tesseract(new, inclinations)
        
        if new_score > score:
            if (any(c.isalpha() for c in new_text)):
                text = new_text
                score = new_score
        elif (not any(c.isalpha() for c in text)) and (any(c.isalpha() for c in new_text)):
            text = new_text
            score = new_score

        text = text.replace("'","")
        text = text.replace('"','')
        text = text.replace('\n','')
        feat["NameBeforeDictionary"] = text
        feat["TesseractCost"] = score
        
        features[count] = feat
        
        count = count+1
    
    data["features"] = features
    return data
    
    
def main(argv):
   inputfile = '/home/lakshya/Downloads/text_recognition/archive/USGS-15-CA-paloalto-e1899-s1895-rp1911.jpg'
   jsonfile = '/home/lakshya/Downloads/text_recognition/Answers/USGS-15-CA-paloalto-e1899-s1895-rp1911.jpg_d14a9c3e-2900-11e8-b1f5-2816adeaecff/geoJson1.json'
   flag = 'combine'

   opts, args = getopt.getopt(argv,"hi:j:f:",["ifile=","jfile=","flag="])

   for opt, arg in opts:
      if opt == '-h':
         print ('test_recognition.py -i <Input image path> -j <Json file>')
         sys.exit()
      elif opt in ("-i", "--ifile"):
         inputfile = arg
      elif opt in ("-j", "--jfile"):
         jsonfile = arg
      elif opt in ("-f", "--flag"):
          flag = arg
    
#   data = merge_rectangles(inputfile, jsonfile)
   start_time = time.time()
   data = text_recognition_merge(inputfile, jsonfile, flag)
   print("--- %s seconds ---" % (time.time() - start_time))
   
   data = str(data)
   data = data.replace('\'','"')   
   
   with open('final.txt', 'w') as outfile:
        outfile.write(data)
       

if __name__ == "__main__":
   main(sys.argv[1:])
