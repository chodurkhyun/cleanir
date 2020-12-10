import numpy as np
import cv2
import pandas as pd
import keras
import os


class VGG2DataGenerator(keras.utils.Sequence):

    def __init__(self, path, 
                 batch_size=32,
                 img_size=(64, 64),
                 shuffle=True, norm='-1to1',
                 pair=False,
                 has_label_files=False):

        self._batch_size = batch_size
        self._img_size = img_size
        self._shuffle = shuffle
        self._norm = norm
        self._pair = pair
        self._has_label_files = has_label_files

        img_files = []
        labels = []
        label = 0
        for root, _, files in os.walk(path):
            if len(files) > 0:
                img_files_in_current_path = [os.path.join(root, f) for f in files if f[-3:] == 'jpg']
                img_files += img_files_in_current_path
                labels += [label] * len(img_files_in_current_path)

                label += 1

        self.n_classes = label
        self._indexes = np.arange(len(img_files))
        self._img_files = np.array(img_files)

        if has_label_files:
            self._labels = np.array([filepath.replace('jpg', 'npy') for filepath in img_files])
        else:
            self._labels = np.array(labels)

        if shuffle:
            np.random.shuffle(self._indexes)

    def __len__(self):
        """
        return the number of batches per epoch
        """
        return len(self._img_files) // self._batch_size

    def __getitem__(self, index):
        """
        generate one batch of data
        """
        idxs = self._indexes[index * self._batch_size:(index + 1) * self._batch_size]
        X, y = self.__data_generation(self._img_files[idxs], self._labels[idxs], self._norm)

        if self._pair:
            idxs2 = np.random.choice(self._indexes, self._batch_size,
                                     replace=False)

            for i in range(self._batch_size):
                for _ in range(len(self._img_files)):  # To avoid infinite loop
                    if self._labels[idxs[i]] != self._labels[idxs2[i]]:
                        break
                    else:
                        idxs2[i] = np.random.choice(self._indexes, 1)
                    
            X2, y2 = self.__data_generation(self._img_files[idxs2], self._labels[idxs2], self._norm)

            return X, y, X2, y2

        else:
            return X, y

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._indexes)

    def __data_generation(self, files, labels, norm, pair=False):
        imgs = [cv2.resize(cv2.imread(f)[:, :, ::-1], self._img_size) for f in files]

        if norm == '-1to1':
            X = np.asarray(imgs, dtype='float32') / 127.5 - 1  # -1 ~ 1
        elif norm == '0to1':
            X = np.asarray(imgs, dtype='float32') / 255.0  # 0 ~ 1
        else:
            X = np.asarray(imgs, dtype='uint8')

        if self._has_label_files:
            y = np.array([np.load(filepath) for filepath in labels])
        else:
            y = keras.utils.to_categorical(labels, num_classes=self.n_classes)

        return X, y


def image_generator(files, label_file, batch_size=32):
    while True:
        batch_paths = np.random.choice(a=files, size=batch_size)
        batch_input = []
        batch_output = []

        for input_path in batch_paths:
            input_img = cv2.imread(input_path)
            input_img = input_img.astype('float32') / 127.5 - 1
            img_id = int(input_path.split('/')[-1].split('.')[0])
            output = label_file.loc[img_id].values
            batch_input += [input_img]
            batch_output += [output]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        yield batch_x, batch_y


def auto_encoder_generator(files, batch_size=32):
    while True:
        batch_paths = np.random.choice(a=files, size=batch_size)
        batch_input = []
        batch_output = []

        for input_path in batch_paths:
            input_img = cv2.imread(input_path)
            input_img = input_img.astype('float32') / 127.5 - 1
            output = input_img
            batch_input += [input_img]
            batch_output += [output]

        batch_x = np.array(batch_input)
        batch_y = np.array(batch_output)

        #yield batch_x, batch_y
        yield (batch_x, None)
