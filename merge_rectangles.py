# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 13:40:32 2018

@author: lakshya
"""
import numpy as np
import json
from PIL import Image, ImageDraw

def union(a, b):
  #print(len(a))
  #print(len(b))
  x = min(a[0], b[0])
  y = min(a[1], b[1])
  w = max(a[0]+a[2], b[0]+b[2]) - x
  h = max(a[1]+a[3], b[1]+b[3]) - y
  angle = 0  
  
  #if a[4]==0 and b[4]==0:
    #  angle = 0
 # elif a[4]==90 and b[4]==90:
   #   angle = 90

  return (x, y, w, h)

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
     
     
def merge_rectangles(input_image, input_json,apth):
    
    data = json.load(open(input_json))
    
    features = data["features"]
    boxes = []
    
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
        
        box = [x1, y1, x2-x1, y2-y1]
        
        boxes.append(box)
    
    #boxes = [[3,2,3,2], [4,4,3,3]]
    
    boxes = combine_boxes(boxes)
    image = Image.open(input_image)
    img = np.asarray(image)
    
    draw = ImageDraw.Draw(image)
    count = 0;    
    for box in boxes:
        draw.rectangle([box[0], box[1], box[0]+box[2], box[1] + box[3]], outline="blue")
        x1 = box[0]
        y1 = box[1]
        x2 = box[0]+box[2]
        y2 = box[1]+box[3]
        if x1<0 or x2<0 or y1<0 or y2<0:
          continue
        part_image = img[y1:y2, x1:x2, :]
        new = Image.fromarray(np.uint8(part_image))
        new.save(str(count)+ ".png")

        count += 1
    
    image.save(apth+"apth+output_merged.png", "PNG")
    print(len(boxes))
