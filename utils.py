import numpy as np
import os
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from PIL import Image
from data_extraction import get_image, get_image_and_anno, imageWithOverlay
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.color import rgb2gray, rgba2rgb
from skimage.morphology import opening, closing, disk
from skimage.transform import resize, downscale_local_mean


#new background mask addition
from skimage.color import rgb2hsv
import tensorflow as tf
from skimage.measure import label, regionprops

def load_img(src_path):
    img = rgba2rgb(plt.imread(src_path)) #[:, :, :3]
    return img

def load_annot(src_path):
    #annot = rgb2gray(rgba2rgb(plt.imread(src_path))) > 0
    im = Image.open(src_path)
    return (np.array(im))>0

def save_img(img, img_name):
    plt.imsave(img_name, img, cmap="gray")
    return None
def save_annot(annot, annot_name):
    annot = annot.astype("bool")
    im = Image.fromarray(annot)
    im.save(annot_name)
    #plt.imsave(annot_name, annot, cmap="gray")
    return None
def save_img_annot_overlay(im_path, ann_path, name, dest_path):
    #rgbpath= path = im_path #os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "dataset_more_precise", "17cu065073-Ki67.ndpi")
    #annotpath= path = ann_path #os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "dataset_more_precise", "17cu065073-Ki67.ndpi.ndpa")
    im, ann = get_image_and_anno(im_path, ann_path, verbose = True)
    img_overlay = imageWithOverlay(im, ann)

    save_img(img_overlay, os.path.join(dest_path, name+ "_overlay.png"))
    save_img(im, os.path.join(dest_path, name + "_im.png"))
    save_annot(ann, os.path.join(dest_path, name+ "_annot.png"))

"""
def show_prob_map_image(Y):
    plt.figure()
    ax = plt.gca()
    im = ax.imshow(Y)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.imshow(Y, cmap=plt.cm.gray)
    plt.colorbar(im, cax=cax)
    plt.show()
    return None
"""

def show_heatmap(Y_prob):
    # Taken from https://matplotlib.org/stable/gallery/axes_grid1/simple_colorbar.html#sphx-glr-gallery-axes-grid1-simple-colorbar-py
    ax = plt.subplot()
    im = ax.imshow(Y_prob)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.show()
    """
    hm = sns.heatmap(Y, annot=False, xticklabels=False, yticklabels=False)
    plt.show()
    plt.close()
    #hm.figure.savefig("seaborn_heatmap_test.png")
    """
    return None
def save_probmap(Y_prob, name):
    # Taken from https://matplotlib.org/stable/gallery/axes_grid1/simple_colorbar.html#sphx-glr-gallery-axes-grid1-simple-colorbar-py
    ax = plt.subplot()
    im = ax.imshow(Y_prob)
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(name, dpi=300, quality = 95)
    plt.close()
    """
    hm = sns.heatmap(Y, annot=False, xticklabels=False, yticklabels=False)
    hm.figure.savefig(name + ".png")
    plt.close()
    """
    return None


def generate_pair_data_ndpi_ndpa(folder_path, distinguishers, verbose=False):
    image_files = sorted(
        [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(distinguishers[0])
        ]
    )
    annotation_files = sorted(
        [
            os.path.join(folder_path, fname + ".ndpa")
            for fname in image_files
        ]
    )
    """
    annotation_files = sorted(
        [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(distinguishers[1])
        ]
    )
    """
    if verbose:
        for img_path, annot_path in zip(image_files, annotation_files):
            print(img_path, "|", annot_path)


    pair = [[image_files[i], annotation_files[i]] for i in range(len(image_files))]
    return pair


def generate_pair_data_names(folder_path, distinguishers, verbose=False):
    image_files = sorted(
        [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(distinguishers[0])
        ]
    )
    annotation_files = sorted(
        [
            os.path.join(folder_path, fname)
            for fname in os.listdir(folder_path)
            if fname.endswith(distinguishers[1])
        ]
    )

    if verbose:
        for img_path, annot_path in zip(image_files, annotation_files):
            print(img_path, "|", annot_path)


    pair = [[image_files[i], annotation_files[i]] for i in range(len(image_files))]
    return pair
    """
def extract_tissue_mask(input, down_factor=20, verbose=True):
    disk_size = 20
    if verbose: print("Converting input to grayscale")
    gray_input = rgb2gray(input)
    if verbose: print("Downscalding input")
    grayscale_downscaled_img = downscale_local_mean(gray_input, (down_factor, down_factor))
    if verbose: print("Finding optimal threshold")
    #grayscale_img = rgb2gray(downscaled_img)
    automatic_thres = threshold_otsu(grayscale_downscaled_img)
    if verbose: print("Generating a mask based on the optimal threshold")
    tissue_mask = grayscale_downscaled_img<automatic_thres
    if verbose: print("Cleaning the mask")
    tissue_mask_ = opening(closing(tissue_mask, disk(disk_size)), disk(disk_size)) #expensive operation. must be done on downscaled image
    if verbose: print("finished cleaning")
    #bg = resize(bg, (rgb.shape[0],rgb.shape[1]))<0.5
    if verbose: print("Resizing the mask to the original shape")
    tissue_mask_resized = resize(tissue_mask_, (input.shape[0], input.shape[1]))
    """
    """
    plt.figure()

    plt.subplot(5,1,1)
    plt.imshow(grayscale_downscaled_img, cmap=plt.cm.gray)
    plt.subplot(5,1,2)
    plt.imshow(tissue_mask, cmap=plt.cm.gray)
    plt.subplot(5,1,3)
    plt.imshow(tissue_mask_, cmap=plt.cm.gray)
    plt.subplot(5,1,4)
    plt.imshow(tissue_mask_resized, cmap=plt.cm.gray)
    plt.show()
    """
    return tissue_mask_resized

def convert_pred_to_mask_backup(Y_pred, shape, classification_segmentation="segmentation"):
    pred_mask = np.zeros(shape)
    if classification_segmentation=="segmentation":
        for y in range(shape[0]):
            for x in range(shape[1]):
                pred_mask[y][x] = np.argmin(Y_pred[y][x])
        pred_mask = pred_mask==0
    elif classification_segmentation=="classification":
        #TODO: implement the fact that if the tile is predicted positive, pred_mask is filled with 1s. If predicted negative, pred_mask is filled with 0s
        pass
    return pred_mask
"""
def convert_pred_to_mask_BASIC(Y_prob, threshold=0.5):
    pred_mask = Y_prob>thresh
    return pred_mask
"""

def convert_pred_to_mask(Y_pred, shape, classification_segmentation="segmentation"):
    thresh = 0.5
    pred_mask = np.zeros(shape)
    if classification_segmentation=="segmentation":
        for y in range(shape[0]):
            for x in range(shape[1]):
                pred_mask[y][x] = Y_pred[y][x][1]
        pred_mask = pred_mask>thresh
    elif classification_segmentation=="classification":
        #TODO: implement the fact that if the tile is predicted positive, pred_mask is filled with 1s. If predicted negative, pred_mask is filled with 0s
        pass
    return pred_mask

"""
def convert_pred_to_probability_map_BASIC(Y_pred, down_factor=4):
    Y_prob_map = Y_pred[:,:,1]
    Y_prob_map_downscaled = downscale_local_mean(Y_prob_map, (down_factor, down_factor))
    return Y_prob_map_downscaled
"""

def convert_pred_to_probability_map(Y_pred, shape, classification_segmentation="segmentation"):
    pred_mask = np.zeros(shape)
    if classification_segmentation=="segmentation":
        for y in range(shape[0]):
            for x in range(shape[1]):
                pred_mask[y][x] = Y_pred[y][x][1] #only one prob is enough to get a prob map
    elif classification_segmentation=="classification":
        #TODO: implement the fact that if the tile is predicted positive, pred_mask is filled with 1s. If predicted negative, pred_mask is filled with 0s
        pass
    return pred_mask

def get_column(pairs, column_index):
    column = [pairs[i][column_index] for i in range(len(pairs))]
    return column

def compare_img_annot(img, annot):
    plt.figure()
    plt.subplot(2,1,1)
    plt.imshow(img)
    plt.subplot(2,1,2)
    plt.imshow(annot, cmap=plt.cm.gray)
    plt.show()
    #name_pred_file = os.path.join("testfolder", str(it))
    #plt.savefig(name_pred_file)
    return None

def fullprint(*args):
    from pprint import pprint
    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)
    pprint(*args)
    np.set_printoptions(**opt)


if __name__ == '__main__':
    coucou = 0
