"""" The raw data consists of .ndpi whose slide images are too voluminous to be handled with DL models. 
This functions aims at preparing the data and transform it into tiles while keeping track of 
where they came from (image and coordinates on the image)""""

import os
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2hsv, rgb2gray
from skimage.filters import threshold_otsu
from skimage.morphology import opening, closing, disk
from skimage.transform import resize, downscale_local_mean
#import cv2
import sys
import pickle
import data_extraction
from utils import *


def get_coord_valid_tiles_BASIC(annot_img, tissue_mask, min_tissue_area, tile_size, overlap):
    counter_valid_tiles = 0
    if overlap==False:
        nb_tiles_y = annot_img.shape[0]//tile_size[0]
        nb_tiles_x = annot_img.shape[1]//tile_size[1]
    elif overlap==True:
        nb_tiles_y = annot_img.shape[0] // int(tile_size[0]/2)
        nb_tiles_x = annot_img.shape[1] // int(tile_size[1]/2)
    coords_corner_valid_tiles = []
    tile_temp = np.zeros(tile_size)
    for y in range(nb_tiles_y):
        for x in range(nb_tiles_x):
            if overlap==False:
                y_up = y*tile_size[0]
                y_down = (y+1)*tile_size[0]
                x_left = x*tile_size[1]
                x_right = (x+1)*tile_size[1]
            elif overlap==True:
                y_up = y*int(tile_size[0]/2)
                y_down = y_up + tile_size[0]
                x_left = x *int(tile_size[1]/2)
                x_right = x_left + tile_size[1]

            tile_temp = tissue_mask[y_up:y_down, x_left:x_right]
            #condition to accept tissue tile
            if tile_temp.sum() > min_tissue_area:
                coords_corner_valid_tiles.append([y_up, x_left])
                counter_valid_tiles+=1

    print("Number of valid tiles : ", counter_valid_tiles)
    return np.array(coords_corner_valid_tiles)

def generate_tiles_from_mask(tiles_folder_path, img_name, annot_name, img, annot_img, tissue_mask, tile_size, min_tissue_percentage_area, min_tumor_percentage_area, dist_border, overlap=False):
    # Note: crops the last pixels if the image is not a multiple of the tile size

    counter_pos_close_border =0
    counter_core_pos_tiles = 0
    counter_neg_close_border = 0
    counter_far_neg_tiles = 0
    min_tissue_area = min_tissue_percentage_area * tile_size[0] * tile_size[1]
    min_tumor_area = min_tumor_percentage_area*tile_size[0]*tile_size[1]

    coords_corner_valid_tiles = get_coord_valid_tiles_BASIC(annot_img, tissue_mask, min_tissue_area, tile_size, overlap)

    i = 0
    for leftmost_corner in coords_corner_valid_tiles:
        tile_img = img[leftmost_corner[0]:leftmost_corner[0] + tile_size[0], leftmost_corner[1]:leftmost_corner[1] + tile_size[1]]
        tile_annot = (annot_img[leftmost_corner[0]:leftmost_corner[0] + tile_size[0], leftmost_corner[1]:leftmost_corner[1] + tile_size[1]])>0
        annot_tile_with_surroundings = (annot_img[leftmost_corner[0] - int(dist_border*tile_size[0]):leftmost_corner[0] + int((1+dist_border)* tile_size[0]), leftmost_corner[1] - int(dist_border*tile_size[1]):leftmost_corner[1] + int((1+dist_border)* tile_size[1])]) > 0

        tumor_pix = annot_tile_with_surroundings.shape[0]*annot_tile_with_surroundings.shape[1]
        area_tumor = (tile_annot>0).sum()
        if area_tumor>=min_tumor_area:
            #the tile contains enough tumor. Will be labelled as positive, but we still need to determine wether it is at the core of the tumor
            # or if it is on the border of the tumor
            if annot_tile_with_surroundings.sum()>= 0.99*tumor_pix:
                # Core positive tiles are positive tiles that are not next to the tumor boundary (far enough from the tumor border)
                tile_img_name = os.path.join(tiles_folder_path, "core_pos_tiles", os.path.basename(img_name) + "_tile_" + str(i) + "_img" +".png")
                tile_annot_name = os.path.join(tiles_folder_path, "core_pos_tiles", os.path.basename(annot_name) + "_tile_" + str(i) + "_annot" + ".png")
                save_img(tile_img, tile_img_name)
                save_annot(tile_annot, tile_annot_name)
                counter_core_pos_tiles+=1
            else:
                counter_pos_close_border+=1
        elif annot_tile_with_surroundings.any()==False:
            counter_far_neg_tiles+=1
            tile_img_name = os.path.join(tiles_folder_path, "far_neg_tiles", os.path.basename(img_name) + "_tile_" + str(i) + "_img" + ".png")
            tile_annot_name = os.path.join(tiles_folder_path, "far_neg_tiles", os.path.basename(annot_name) + "_tile_" + str(i) + "_annot" + ".png")
            save_img(tile_img, tile_img_name)
            save_annot(tile_annot, tile_annot_name)
        else:
            counter_neg_close_border+=1
        i+=1

    print("Number of core pos tiles: ", counter_core_pos_tiles)
    print("Number of pos tiles close to the tumor border: ", counter_pos_close_border)
    print("Number of far neg tiles: ", counter_far_neg_tiles)
    print("Number of neg tiles close to the tumor border: ", counter_neg_close_border)

    return None



def generate(dic):
    print("Data preparation of :  ", dic)
    train_val_str = dic['train_val_str']
    mag_factor = dic['mag_factor']
    dist_border = dic['dist_border']
    tile_size = dic['tile_size']
    overlap = dic['overlap']

    source_folder_name = train_val_str + '_set'
    context_resolution_folder_name = "mag" + str(mag_factor) + "_" + str(tile_size[0]) + "_" + str(tile_size[1])
    tiles_folder_name = train_val_str
    if overlap == True:
        overlap_str = "_overlap"
    elif overlap == False:
        overlap_str = "_no_overlap"

    if train_val_str=='training':
        tiles_folder_name += overlap_str

    #down_factor = mag_factor
    min_tissue_percentage_area = 0.85
    min_tumor_percentage_area = 1.
    current_wd = os.getcwd()


    dataset_folder_path = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", source_folder_name) #dataset_more_precise
    prepared_dataset_path = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", context_resolution_folder_name, tiles_folder_name)

    print("Input dataset: ", dataset_folder_path)
    print("Output dataset: ", prepared_dataset_path)

    pair_WSI_annotations = generate_pair_data_ndpi_ndpa(dataset_folder_path, [".ndpi", ".ndpa"], verbose=True)
    print("The tiling will start now")
    for files in pair_WSI_annotations:
        WSI_name = files[0]
        annot_name = files[1]
        print("Loading " + os.path.basename(WSI_name) + " and " + os.path.basename(annot_name))
        WSI_img, annot_img, tissue_mask = data_extraction.get_image_anno_tissue(WSI_name, annot_name, mag=mag_factor, verbose=True)#WSI_img, annot_img = data_extraction.get_image_and_anno(WSI_name, annot_name, mag=mag_factor, verbose=True)
        #tissue_mask = get_tissue_mask_BASIC(WSI_name, annot_img.shape, mag_factor)
        generate_tiles_from_mask(prepared_dataset_path, WSI_name, annot_name, WSI_img, annot_img, tissue_mask, tile_size, min_tissue_percentage_area, min_tumor_percentage_area, dist_border, overlap=overlap)

def main():
    gen_tiles = []

    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 10, 'dist_border': 0.5, 'tile_size': [512, 512], 'overlap': False})
    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 10, 'dist_border': 0.5, 'tile_size': [512, 512], 'overlap': True})
    gen_tiles.append({"train_val_str": 'validation', 'mag_factor': 10, 'dist_border': 0.5, 'tile_size': [512, 512], 'overlap': False})

    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 10, 'dist_border': 1, 'tile_size': [256, 256], 'overlap': False})
    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 10, 'dist_border': 1, 'tile_size': [256, 256], 'overlap': True})
    gen_tiles.append({"train_val_str": 'validation', 'mag_factor': 10, 'dist_border': 1, 'tile_size': [256, 256], 'overlap': False})

    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 10, 'dist_border': 2, 'tile_size': [128, 128], 'overlap': False})
    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 10, 'dist_border': 2, 'tile_size': [128, 128], 'overlap': True})
    gen_tiles.append({"train_val_str": 'validation', 'mag_factor': 10, 'dist_border': 2, 'tile_size': [128, 128], 'overlap': False})

    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 5, 'dist_border': 0.5, 'tile_size': [256, 256], 'overlap': False})
    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 5, 'dist_border': 0.5, 'tile_size': [256, 256], 'overlap': True})
    gen_tiles.append({"train_val_str": 'validation', 'mag_factor': 5, 'dist_border': 0.5, 'tile_size': [256, 256], 'overlap': False})

    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 5, 'dist_border': 1, 'tile_size': [128, 128], 'overlap': False})
    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 5, 'dist_border': 1, 'tile_size': [128, 128], 'overlap': True})
    gen_tiles.append({"train_val_str": 'validation', 'mag_factor': 5, 'dist_border': 1, 'tile_size': [128, 128], 'overlap': False})

    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 5, 'dist_border': 2, 'tile_size': [64, 64], 'overlap': False})
    gen_tiles.append({"train_val_str": 'training', 'mag_factor': 5, 'dist_border': 2, 'tile_size': [64, 64], 'overlap': True})
    gen_tiles.append({"train_val_str": 'validation', 'mag_factor': 5, 'dist_border': 2, 'tile_size': [64, 64], 'overlap': False})

    #generate(gen_tiles)
    for g in gen_tiles:
        generate(g)

if __name__ == '__main__':
    main()
