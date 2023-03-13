from matplotlib import pyplot as plt
import numpy as np
import math
import os
import sys
import pickle
from scipy import ndimage
from skimage.exposure import adjust_gamma, rescale_intensity
from skimage.color import rgb2gray, rgba2rgb
from utils import *


def get_balanced_training_set_BASIC(pos_train, neg_train, epoch):
    np.random.seed(epoch)
    np.random.shuffle(pos_train)
    np.random.shuffle(neg_train)
    nb_tiles_per_class = np.min([len(pos_train), len(neg_train)])
    pos_train_balanced = pos_train[:nb_tiles_per_class]
    neg_train_balanced = neg_train[:nb_tiles_per_class]
    return pos_train_balanced, neg_train_balanced


# gets a training and validation set where there is as much + than - in each set
def get_training_validation_sets_BASIC(path_to_dataset, overlap_data):
    overlap_str =""
    if overlap_data==True:
        overlap_str = "_overlap"
    else:
        overlap_str = "_no_overlap"
    np.random.seed(1)
    #Training set
    train_path_positive_pairs_X_Y = os.path.join(path_to_dataset, "training"+overlap_str, "core_pos_tiles")
    train_pairs_pos = generate_pair_data_names(train_path_positive_pairs_X_Y, ["img.png", "annot.png"])
    np.random.shuffle(train_pairs_pos)

    train_path_negative_pairs_X_Y = os.path.join(path_to_dataset, "training"+overlap_str, "far_neg_tiles")
    train_pairs_neg = generate_pair_data_names(train_path_negative_pairs_X_Y, ["img.png", "annot.png"])
    np.random.shuffle(train_pairs_neg)

    #Validation
    val_path_positive_pairs_X_Y = os.path.join(path_to_dataset, "validation", "core_pos_tiles")
    val_pairs_pos = generate_pair_data_names(val_path_positive_pairs_X_Y, ["img.png", "annot.png"])
    np.random.shuffle(val_pairs_pos)

    val_path_negative_pairs_X_Y = os.path.join(path_to_dataset, "validation", "far_neg_tiles")
    val_pairs_neg = generate_pair_data_names(val_path_negative_pairs_X_Y, ["img.png", "annot.png"])
    np.random.shuffle(val_pairs_neg)

    nb_tiles_per_class_validation = np.min([len(val_pairs_pos), len(val_pairs_neg)])
    val_pairs_pos = val_pairs_pos[:nb_tiles_per_class_validation]
    val_pairs_neg = val_pairs_neg[:nb_tiles_per_class_validation]

    return train_pairs_pos,train_pairs_neg,val_pairs_pos, val_pairs_neg


class DataGenerator_v2():
    """Reads dataset files (images & annotations) and prepare training & validation batches"""

    def __init__(self, batch_size, validation_size, tile_size, path_to_dataset, data_augmentation, overlap_data):

        self.batch_size = batch_size  # TODO
        self.validation_size = validation_size  # TODO
        self.tile_size = tile_size
        self.path_to_dataset = path_to_dataset
        self.data_augmentation = data_augmentation
        self.overlap_data = overlap_data

        self.pos_train, self.neg_train, self.pos_val, self.neg_val = get_training_validation_sets_BASIC(self.path_to_dataset, self.overlap_data)
        self.nb_tiles_val = len(self.pos_val) + len(self.neg_val)
        self.nb_tiles_train_per_class = np.min([len(self.pos_train), len(self.neg_train)])
        self.nb_tiles_train_balanced = 2 * self.nb_tiles_train_per_class
        self.batches_per_epoch = self.nb_tiles_train_balanced // self.batch_size
        print("nb of batches per epoch:", self.batches_per_epoch)
        print("exiting DataGenerator()")

    def next_balanced_batch_BASIC(self, n_epochs):
        print("entered in next_balanced_batch_BASIC")
        for e in range(n_epochs):
            pos_train_balanced, neg_train_balanced = get_balanced_training_set_BASIC(self.pos_train, self.neg_train, e)
            print("new epoch --> e = ", e)
            print("len(pos_train) : ", len(self.pos_train))
            print("len(neg_train) : ", len(self.neg_train))
            print("len(pos_train_balanced) : ", len(pos_train_balanced))
            print("len(neg_train_balanced) : ", len(neg_train_balanced))
            batch_nb_in_epoch = 0
            index_in_batch = 0
            batch_x = np.zeros((self.batch_size,) + self.tile_size + (3,))
            batch_y = np.zeros((self.batch_size,) + self.tile_size)

            for p, n in zip(pos_train_balanced, neg_train_balanced):
                if batch_nb_in_epoch == self.batches_per_epoch:
                    print(
                        "entered in break because we reached the number of batches per epoch, with batch_nb_in_epoch=",
                        batch_nb_in_epoch)
                    break
                tile_rgb_pos = load_img(p[0])
                tile_mask_pos = load_annot(p[1])
                tile_rgb_neg = load_img(n[0])
                tile_mask_neg = load_annot(n[1])

                batch_x[index_in_batch] = tile_rgb_pos
                batch_x[index_in_batch + 1] = tile_rgb_neg
                batch_y[index_in_batch] = tile_mask_pos
                batch_y[index_in_batch + 1] = tile_mask_neg

                index_in_batch += 2

                if index_in_batch >= self.batch_size:
                    # the batch is fully loaded with + and - tiles. We can proceed to data augmentation and yield the result
                    index_in_batch = 0
                    batch_nb_in_epoch += 1
                    if self.data_augmentation == True:
                        yield self._augment(batch_x, batch_y)
                    else:
                        yield batch_x, batch_y

    def get_validation_data_BASIC(self):
        print("entered in get_validation_data()")
        while (True):
            gen_size = self.batch_size
            val_x = np.zeros((gen_size,) + self.tile_size + (3,))
            val_y = np.zeros((gen_size,) + self.tile_size)
            index_in_gen = 0
            for p, n in zip(self.pos_val, self.neg_val): #TODO: make sure there is an even number of + and -
                tile_rgb_pos = load_img(p[0])
                tile_mask_pos = load_annot(p[1])
                tile_rgb_neg = load_img(n[0])
                tile_mask_neg = load_annot(n[1])

                val_x[index_in_gen] = tile_rgb_pos
                val_x[index_in_gen + 1] = tile_rgb_neg
                val_y[index_in_gen] = tile_mask_pos
                val_y[index_in_gen + 1] = tile_mask_neg
                index_in_gen += 2
                if index_in_gen == gen_size:
                    index_in_gen = 0
                    yield val_x, val_y

    @staticmethod
    def _augment(batch_x, batch_y):
        """Data augmentation:
        Horizontal/Vertical symmetry
        Random rotation
        Gamma correction
        Random noise
        """
        # Vertical symmetry
        if (np.random.random() < 0.5):
            batch_x = batch_x[:, ::-1, :, :]
            batch_y = batch_y[:, ::-1, :]
        # Horizontal symmetry
        if (np.random.random() < 0.5):
            batch_x = batch_x[:, :, ::-1, :]
            batch_y = batch_y[:, :, ::-1]
        # Rotation; since each tile is 100% from one class, no need to rotate the annotation
        if (np.random.random() < 0.5):
            rot_angle = np.random.randint(0, 45)
            for i in range(len(batch_x)):
                im = ndimage.rotate(batch_x[i], rot_angle, order=5, reshape=False, mode='reflect')
                # the 5th order interpolation could lead to negative rare negative values. Rescaling is needed to get values back to the range 0,1
                im = rescale_intensity(im, in_range=(0, 1))
                batch_x[i] = im

        # Gamma correction
        gamma = (np.random.random() - 0.5) * 2
        if gamma < 0:
            gamma = 1 / (1 - gamma)
        else:
            gamma = 1 + gamma

        batch_x_ = batch_x.copy()
        for i in range(len(batch_x)):
            batch_x_[i] = adjust_gamma(batch_x[i], gamma=gamma)

        # Random noise
        batch_x_ += np.random.normal(0, 0.02, size=batch_x.shape)

        return batch_x_, batch_y


def test_access_tiles():
    path_to_dataset = data_path = os.path.join(os.path.dirname(os.getcwd()), "DataThesis",
                                               "mag10_512_512_tissue_tiles_strict")
    path_positive_pairs_X_Y = os.path.join(path_to_dataset, "core_pos_tiles")
    pairs_pos = generate_pair_data_names(path_positive_pairs_X_Y, ["img.png", "annot.png"])
    path_negative_pairs_X_Y = os.path.join(path_to_dataset, "far_neg_tiles")
    pairs_neg = generate_pair_data_names(path_negative_pairs_X_Y, ["img.png", "annot.png"])

    mask = load_annot(pairs_pos[0][1])
    print(mask)
    print(mask.shape)

if __name__ == '__main__':
    test_access_tiles()
