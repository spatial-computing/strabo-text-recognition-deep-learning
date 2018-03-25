#!/usr/bin/env python3

import os

import time
import datetime
import cv2
import numpy as np
import uuid
import json
import math
import random

import functools
import logging
import collections
from pprint import pprint
from collections import defaultdict
from scipy.misc import toimage
from scipy import ndimage
from math import sin, cos, radians

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@functools.lru_cache(maxsize=1)
def get_host_info():
    ret = {}
    with open('/proc/cpuinfo') as f:
        ret['cpuinfo'] = f.read()

    with open('/proc/meminfo') as f:
        ret['meminfo'] = f.read()

    with open('/proc/loadavg') as f:
        ret['loadavg'] = f.read()

    return ret


@functools.lru_cache(maxsize=100)
def get_predictor(checkpoint_path):
    logger.info('loading model')
    import tensorflow as tf
    import model
    from icdar import restore_rectangle
    import lanms
    from eval import resize_image, sort_poly, detect

    input_images = tf.placeholder(
        tf.float32, shape=[None, None, None, 3], name='input_images')
    global_step = tf.get_variable(
        'global_step', [], initializer=tf.constant_initializer(0), trainable=False)

    f_score, f_geometry = model.model(input_images, is_training=False)

    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    print("here")
    ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
    model_path = os.path.join(checkpoint_path, os.path.basename(
        ckpt_state.model_checkpoint_path))
    logger.info('Restore from {}'.format(model_path))
    saver.restore(sess, model_path)

    def predictor(img):
        """
        :return: {
            'text_lines': [
                {
                    'score': ,
                    'x0': ,
                    'y0': ,
                    'x1': ,
                    ...
                    'y3': ,
                }
            ],
            'rtparams': {  # runtime parameters
                'image_size': ,
                'working_size': ,
            },
            'timing': {
                'net': ,
                'restore': ,
                'nms': ,
                'cpuinfo': ,
                'meminfo': ,
                'uptime': ,
            }
        }
        """
        start_time = time.time()
        rtparams = collections.OrderedDict()
        rtparams['start_time'] = datetime.datetime.now().isoformat()
        rtparams['image_size'] = '{}x{}'.format(img.shape[1], img.shape[0])
        timer = collections.OrderedDict([
            ('net', 0),
            ('restore', 0),
            ('nms', 0)
        ])

        im_resized, (ratio_h, ratio_w) = resize_image(img)
        rtparams['working_size'] = '{}x{}'.format(
            im_resized.shape[1], im_resized.shape[0])
        start = time.time()
        score, geometry = sess.run(
            [f_score, f_geometry],
            feed_dict={input_images: [im_resized[:, :, ::-1]]})
        timer['net'] = time.time() - start

        boxes, timer = detect(score_map=score, geo_map=geometry, timer=timer)
        logger.info('net {:.0f}ms, restore {:.0f}ms, nms {:.0f}ms'.format(
            timer['net']*1000, timer['restore']*1000, timer['nms']*1000))

        if boxes is not None:
            scores = boxes[:, 8].reshape(-1)
            boxes = boxes[:, :8].reshape((-1, 4, 2))
            boxes[:, :, 0] /= ratio_w
            boxes[:, :, 1] /= ratio_h

        duration = time.time() - start_time
        timer['overall'] = duration
        logger.info('[timing] {}'.format(duration))

        text_lines = []
        if boxes is not None:
            text_lines = []
            for box, score in zip(boxes, scores):
                box = sort_poly(box.astype(np.int32))
                if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
                    continue
                tl = collections.OrderedDict(zip(
                    ['x0', 'y0', 'x1', 'y1', 'x2', 'y2', 'x3', 'y3'],
                    map(float, box.flatten())))
                tl['score'] = float(score)
                text_lines.append(tl)
        ret = {
            'text_lines': text_lines,
            'rtparams': rtparams,
            'timing': timer,
        }
        # ret.update(get_host_info())
        return ret

    return predictor


# the webserver
from flask import Flask, request, render_template
import argparse


class Config:
    SAVE_DIR = 'static/results'


config = Config()


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html', session_id='dummy_session_id')


def draw_illu(illu, rst):
    for t in rst['text_lines']:
        d = np.array([t['x0'], t['y0'], t['x1'], t['y1'], t['x2'],
                      t['y2'], t['x3'], t['y3']], dtype='int32')
        d = d.reshape(-1, 2)
        cv2.polylines(illu, [d], isClosed=True, color=(255, 0, 0))
    return illu


def save_result(image_name, img, rst):
    import jsonReader
    import merge_rectangles
    session_id = str(uuid.uuid1())
    if(config_params['result_path'] != 0):
        dirpath = os.path.join(
            config_params['result_path'], image_name+'_'+session_id)
    else:
        dirpath = os.path.join(config.SAVE_DIR, image_name+'_'+session_id)

    os.makedirs(dirpath)

    # save input image
    output_path = os.path.join(
        dirpath, 'input.png')
    cv2.imwrite(output_path, img)

    # save illustration
    output_path = os.path.join(
        dirpath, 'output.png')
    #toimage(draw_illu(img.copy(), rst)).show()
    cv2.imwrite(output_path, draw_illu(img.copy(), rst))

    # save json data
    output_path = os.path.join(dirpath, 'result.json')
    with open(output_path, 'w') as f:
        json.dump(rst, f)

    jsonReader.convert(os.path.join(dirpath, 'result.json'), dirpath)
    merge_rectangles.merge_rectangles( os.path.join(
        dirpath, 'input.png'),os.path.join(dirpath, 'geoJson1.json'),dirpath)

    rst['session_id'] = session_id
    
    print(os.path.join(dirpath, 'input.png'))
    print(os.path.join(dirpath, 'output.png'))
    print(os.path.join(dirpath, 'result.png'))
    print(os.path.join(dirpath, 'geoJson1.json'))
    
    print("python3 text_recognition.py -i " + os.path.join(dirpath, 'input.png') +" -j " + os.path.join(dirpath, 'geoJson1.json') + " -o " + os.path.join(dirpath, 'final.txt'))
    #os.system("python3 text_recognition.py -i " + os.path.join(dirpath, 'input.png') +" -j " + os.path.join(dirpath, 'geoJson1.json') + " -o " + os.path.join(dirpath, 'final.txt'))

    
    
    return rst


checkpoint_path = './east_icdar2015_resnet_v1_50_rbox'


def detectText(image_name):
    global predictor
    import io
    import base64
    bio = io.BytesIO()
    print('image name is {}'.format(image_name))

    with open(image_name, 'rb') as infile:
        buf = infile.read()
    x = np.fromstring(buf, dtype='uint8')
    img = cv2.imdecode(x, 1)
    rst = get_predictor(checkpoint_path)(img)
    
    rotatedplus90 = rotate_bound(img, 90)
    rstplus90 = get_predictor(checkpoint_path)(rotatedplus90)   
    resplus90 = rotateBox(img,rotatedplus90,rstplus90,-90)

    rotatedminus90 = rotate_bound(img, -90)
    rstminus90 = get_predictor(checkpoint_path)(rotatedminus90)   
    resminus90 = rotateBox(img,rotatedminus90,rstminus90,90)

    iterList = rst['text_lines']
    iterList.extend(resplus90)
    iterList.extend(resminus90)
    rst['text_lines'] = iterList

    save_result(image_name, img, rst)
    print(rst['session_id'])


save_path = 'static/results'
config_params = defaultdict(int)


def rotate_bound(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    return cv2.warpAffine(image, M, (nW, nH))


def rotateBox(img_orig, rotated_img, rst,angleDeg):

    (heigth, width) = img_orig.shape[:2]
    (cx, cy) = (width // 2, heigth // 2)
    (new_height, new_width) = rotated_img.shape[:2]
    (new_cx, new_cy) = (new_width // 2, new_height // 2)

    points = rst['text_lines']
    # pprint(points)
    results = []
    for t in points:
        x = {}
        d = []
        d.append((t['x0'], t['y0']))
        d.append((t['x1'], t['y1']))
        d.append((t['x2'], t['y2']))
        d.append((t['x3'], t['y3']))
        m = rotatePolygon(d, new_cx,new_cy,new_height,new_width,angleDeg)
        x['x0'], x['y0'] = m[0]
        x['x1'], x['y1'] = m[1]
        x['x2'], x['y2'] = m[2]
        x['x3'], x['y3'] = m[3]
        results.append(x)
    return results


def rotatePolygon(bb, cx, cy, h, w, theta):
    new_bb = list(bb)
    for i,coord in enumerate(bb):
        # opencv calculates standard transformation matrix
        M = cv2.getRotationMatrix2D((cx, cy), theta, 1.0)
        # Grab  the rotation components of the matrix)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        v = [coord[0],coord[1],1]
        # Perform the actual rotation and return the image
        calculated = np.dot(M,v)
        new_bb[i] = (calculated[0],calculated[1])
    return new_bb


def readDocument(config_file):
    with open(config_file) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    for x in content:
        tags = x.split('=')
        config_params[tags[0].strip()] = tags[1]
    pprint(config_params)


def main():
    global checkpoint_path
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', default=8769, type=int)
    parser.add_argument('--checkpoint-path', default=checkpoint_path)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--image')
    parser.add_argument('--config')
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path
    print(args.image)

    if not os.path.exists(args.checkpoint_path):
        raise RuntimeError(
            'Checkpoint `{}` not found'.format(args.checkpoint_path))

    if not os.path.exists(args.image):
        raise RuntimeError(
            'Image`{}` not found'.format(args.image))

    if not os.path.exists(args.config):
        raise RuntimeError(
            'Configuration file`{}` not found'.format(args.config))

    readDocument(args.config)
    detectText(args.image)
    #app.debug = args.debug
    #app.run('0.0.0.0', args.port)


if __name__ == '__main__':
    main()
