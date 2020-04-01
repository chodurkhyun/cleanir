from keras_facenet import FaceNet
import pandas as pd
import numpy as np
import os
from tqdm.autonotebook import tqdm
from cleanir.cleanir import Cleanir
from cleanir.tools.crop_face import *


def read_lfw_pair_info(dataset_path):
    """Reads pairs.txt of the LFW dataset
    and return a list of filepaths of image pairs

    Arguments:
        dataset_path {str} -- full path of the LFW

    Returns:
        tuple -- matched list and unmatched list of filepaths of image pairs
    """

    pairdata_path = os.path.join(dataset_path, 'pairs.txt')
    imgdata_path = os.path.join(dataset_path, 'lfw-deepfunneled')
    matched_list = []
    unmatched_list = []

    pair_info = pd.read_csv(pairdata_path, sep='\t', nrows=1, header=None)
    n_fold, n_matched_pair = pair_info.iloc[0][0], pair_info.iloc[0][1]
    read_rows = 1

    for i in tqdm(range(n_fold)):
        matched_data = pd.read_csv(pairdata_path, sep='\t', header=None,
                                   skiprows=read_rows, nrows=n_matched_pair)

        for j in range(n_matched_pair):
            name, idx1, idx2 = matched_data.iloc[j]
            seed_path = '{0}/{1}/{2}'.format(imgdata_path, name, name)
            matched_list += [('{0}_{1:04d}.jpg'.format(seed_path, idx1),
                              '{0}_{1:04d}.jpg'.format(seed_path, idx2))]

        read_rows += n_matched_pair

        unmatched_data = pd.read_csv(pairdata_path, sep='\t', header=None,
                                     skiprows=read_rows, nrows=n_matched_pair)

        for j in range(n_matched_pair):
            name1, idx1, name2, idx2 = unmatched_data.iloc[j]
            seed_path1 = '{0}/{1}/{2}'.format(imgdata_path, name1, name1)
            seed_path2 = '{0}/{1}/{2}'.format(imgdata_path, name2, name2)
            unmatched_list += [('{0}_{1:04d}.jpg'.format(seed_path1, idx1),
                                '{0}_{1:04d}.jpg'.format(seed_path, idx2))]

        read_rows += n_matched_pair

    return matched_list, unmatched_list


def evaluate_id_lfw(dataset_path, cleanir, dsize=(64, 64)):
    """Evaluates CLEANIR model by using LFW dataset

    Arguments:
        dataset_path {str} -- lfw dataset path
        cleanir {Cleanir} -- Cleanir instance

    Keyword Arguments:
        dsize {tuple} -- size of cropped face (default: {(64, 64)})

    Returns:
        dict -- evaluation results
    """

    print('Reading LFW dataset pair information..')
    matched_list, unmatched_list = read_lfw_pair_info(dataset_path)

    print('Loading FaceNet..')
    facenet = FaceNet()

    o_dists = []
    m0_dists = []
    m30_dists = []
    m60_dists = []
    m90_dists = []
    m120_dists = []
    m150_dists = []
    m180_dists = []

    print('Loading and modifying LFW dataset images..')
    for face1_path, face2_path in tqdm(matched_list):
        face1 = crop_face_from_file(face1_path, dsize)
        face2 = crop_face_from_file(face2_path, dsize)

        deid = cleanir.get_deid_single_axis_func(face1)

        ems = facenet.embeddings([face1, face2, deid(180), deid(90), deid(0),
                                  deid(30), deid(60), deid(120), deid(150)])
        o_dists += [facenet.compute_distance(ems[0], ems[1])]
        m180_dists += [facenet.compute_distance(ems[2], ems[1])]
        m90_dists += [facenet.compute_distance(ems[3], ems[1])]
        m0_dists += [facenet.compute_distance(ems[4], ems[1])]
        m30_dists += [facenet.compute_distance(ems[5], ems[1])]
        m60_dists += [facenet.compute_distance(ems[6], ems[1])]
        m120_dists += [facenet.compute_distance(ems[7], ems[1])]
        m150_dists += [facenet.compute_distance(ems[8], ems[1])]

    results = {'threshold': [], 'original': [], '0': [], '30': [], '60': [],
               '90': [], '120': [], '150': [], '180': []}

    print('Thresholding..')
    for threshold in tqdm(np.arange(0.1, 2.0, 0.1)):
        results['threshold'].append(threshold)
        results['original'].append(np.sum(np.array(o_dists) < threshold) / len(o_dists))
        results['180'].append(np.sum(np.array(m180_dists) < threshold) / len(m180_dists))
        results['90'].append(np.sum(np.array(m90_dists) < threshold) / len(m90_dists))
        results['0'].append(np.sum(np.array(m0_dists) < threshold) / len(m0_dists))
        results['30'].append(np.sum(np.array(m30_dists) < threshold) / len(m30_dists))
        results['60'].append(np.sum(np.array(m60_dists) < threshold) / len(m60_dists))
        results['120'].append(np.sum(np.array(m120_dists) < threshold) / len(m120_dists))
        results['150'].append(np.sum(np.array(m150_dists) < threshold) / len(m150_dists))

    return results
