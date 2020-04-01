import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from tqdm.autonotebook import tqdm
import cv2
import matplotlib.pyplot as plt
import asyncio, io, glob, os, sys, time, uuid, requests
from os import path
from urllib.parse import urlparse
from io import BytesIO
from azure.cognitiveservices.vision.face import FaceClient
from msrest.authentication import CognitiveServicesCredentials
from azure.cognitiveservices.vision.face.models import TrainingStatusType, Person, SnapshotObjectType, OperationStatusType
from enum import Enum

from cleanir.cleanir import Cleanir
from cleanir.tools.crop_face import *

face_client = None


class Emotion(Enum):
    ANGRY = 0
    DISGUST = 1
    FEAR = 2
    HAPPY = 3
    NEUTRAL = 4
    SAD = 5
    SURPRISE = 6


def analyze_one_face(face_img):
    faces = face_client.face.detect_with_stream(BytesIO(cv2.imencode('.png', face_img)[1]),
                                                return_face_attributes=["emotion"])

    time.sleep(5)  # to use the free plan (20 transactions per a minute)

    if len(faces) > 0:
        predicted = [faces[0].face_attributes.emotion.anger,
                     faces[0].face_attributes.emotion.contempt,
                     faces[0].face_attributes.emotion.disgust,
                     faces[0].face_attributes.emotion.fear,
                     faces[0].face_attributes.emotion.happiness,
                     faces[0].face_attributes.emotion.neutral,
                     faces[0].face_attributes.emotion.sadness,
                     faces[0].face_attributes.emotion.surprise]

        max_idx = np.argmax(np.array(predicted))
        if max_idx >= 2:
            max_idx -= 1

        return max_idx
    else:
        return None


def get_label(file_path):
    if 'AN' in file_path:
        return Emotion.ANGRY
    elif 'DI' in file_path:
        return Emotion.DISGUST
    elif 'FE' in file_path:
        return Emotion.FEAR
    elif 'HA' in file_path:
        return Emotion.HAPPY
    elif 'NE' in file_path:
        return Emotion.NEUTRAL
    elif 'SA' in file_path:
        return Emotion.SAD
    elif 'SU' in file_path:
        return Emotion.SURPRISE
    else:
        return None


def evaluate_emotion_jaffe_azure(dataset_path, cleanir, endpoint, key,
                                 dsize=(64, 64)):
    global face_client
    face_client = FaceClient(endpoint, CognitiveServicesCredentials(key))
    emotions = [str.lower(emo[0]) for emo in Emotion.__members__.items()]

    emo_results = {'original_pred': [], 'original_true': [],
                   '0_pred': [], '0_true': [],
                   '90_pred': [], '90_true': [],
                   '180_pred': [], '180_true': []}

    for file_path in tqdm(glob.glob(path.join(dataset_path, 'jaffedbase', '*.tiff'))):
        label = get_label(file_path)
        if label is None:
            print("{0} is an invalid file..".format(file_path))
            continue

        face_img = crop_face_from_file(file_path, dsize)
        deid = cleanir.get_deid_single_axis_func(face_img)

        pred = analyze_one_face(face_img)
        if pred is not None:
            emo_results['original_pred'] += [pred]
            emo_results['original_true'] += [label.value]

        for degree in [0, 90, 180]:
            pred = analyze_one_face(deid(degree))
            if pred is not None:
                emo_results['{0}_pred'.format(degree)] += [pred]
                emo_results['{0}_true'.format(degree)] += [label.value]

    cm_dict = {'original': None, '0': None, '90': None, '180': None}

    for k in cm_dict.keys():
        cm = confusion_matrix(emo_results[k + '_true'],
                              emo_results[k + '_pred'])
        df = pd.DataFrame(cm, index=emotions, columns=emotions)
        df.index.name = 'Actual'
        df.columns.name = 'Predicted'
        cm_dict[k] = df

    return cm_dict
