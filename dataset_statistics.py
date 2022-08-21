from utils import *
import numpy as np
import os
import data_extraction
import matplotlib.pyplot as plt

def dataset_analysis(dataset_path):
    mag_factor = 5  # TODO change it to 20 when on Windows
    down_factor = mag_factor
    dataset_folder_path = dataset_path #os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "dataset_more_precise") #dataset_more_precise
    pair_WSI_annotations = generate_pair_data_ndpi_ndpa(dataset_folder_path, [".ndpi", ".ndpa"], verbose=True)
    nb_WSI = len(pair_WSI_annotations)
    print("nb_WSI: ", nb_WSI)
    size_x_ls = np.zeros(nb_WSI)
    size_y_ls = np.zeros(nb_WSI)
    tissue_on_WSI_ls = np.zeros(nb_WSI)
    tumor_on_WSI_ls = np.zeros(nb_WSI)
    tumor_on_tissue_ls = np.zeros(nb_WSI)
    structuredArr = []
    dtype = [('Name', (np.str_, 20)), ('size_x', np.float64), ('size_y', np.float64), ('tissue_on_WSI', np.float64), ('tumor_on_WSI', np.float64), ('tumor_on_tissue', np.float64)]
    i=0
    for files in pair_WSI_annotations:
        WSI_name = files[0]
        annot_name = files[1]
        print("Loading " + os.path.basename(WSI_name) + " and " + os.path.basename(annot_name))
        WSI_img, annot_img, tissue_mask = data_extraction.get_image_anno_tissue(WSI_name, annot_name, mag=mag_factor, verbose=True)
        annot_img = (annot_img>0)*tissue_mask
        #annot_adjusted = annot_img * tissue_mask
        """
        #tissue_mask = get_tissue_mask_BASIC(WSI_name, annot_img.shape, mag_factor, verbose = True)
        #overlay = imageWithOverlay(WSI_img, tissue_mask)
        #overlay_img_annot = imageWithOverlay(WSI_img, annot_img)
        #overlay_img_annot_adjusted = imageWithOverlay(WSI_img, annot_adjusted)
        path_ = os.path.join(os.getcwd(), 'test_folder6', os.path.basename(WSI_name))
        save_img(overlay, path_ + "_overlay.jpeg")
        save_img(overlay_img_annot, path_ + "_overlay_img_annot.jpeg")
        save_img(overlay_img_annot_adjusted, path_ + "_overlay_img_annot_adjusted.jpeg")
        #save_img(WSI_img, path_ + "_img.jpeg")
        #save_annot(annot_img, path_ + "_mask.png")
        #save_annot(tissue_mask, path_ + "_tissue_mask.png")
        """

        #Stats
        size_x_ls[i] = WSI_img.shape[0]*4 #size at x20 mag if mag=5 here
        size_y_ls[i] = WSI_img.shape[1]*4 #size at x20 mag if mag=5 here
        area_WSI = WSI_img.shape[0]*WSI_img.shape[1]
        area_tissue = tissue_mask.sum()
        area_tumor = annot_img.sum()
        #area_tumor_on_tissue = (annot_img*tissue_mask)/sum()
        tissue_on_WSI_ls[i] = area_tissue/area_WSI
        tumor_on_WSI_ls[i] = area_tumor/area_WSI
        tumor_on_tissue_ls[i] = area_tumor/area_tissue
        print("size_x_ls[i]: ", size_x_ls[i])
        print("size_y_ls[i]: ", size_y_ls[i])
        print("area_WSI: ",area_WSI)
        print("area_tissue: ", area_tissue)
        print("area_tumor: ", area_tumor)
        print("tissue_on_WSI_ls[i]", tissue_on_WSI_ls[i])
        print("tumor_on_WSI_ls[i]", tumor_on_WSI_ls[i])
        print("tumor_on_tissue_ls[i]", tumor_on_tissue_ls[i])
        if i==0:
            structuredArr = np.array([(os.path.basename(WSI_name), size_x_ls[i], size_y_ls[i], tissue_on_WSI_ls[i], tumor_on_WSI_ls[i], tumor_on_tissue_ls[i])], dtype=dtype)
        else:
            structuredArr = np.append(structuredArr, np.array([(os.path.basename(WSI_name),size_x_ls[i], size_y_ls[i], tissue_on_WSI_ls[i], tumor_on_WSI_ls[i], tumor_on_tissue_ls[i])], dtype=dtype), axis=0)
        print(WSI_img.shape)
        print(annot_img.shape)
        print(tissue_mask.shape)
        i+=1
        """
        plt.figure()
        plt.subplot(4, 1, 1)
        plt.imshow(WSI_img)
        plt.subplot(4, 1, 2)
        plt.imshow(overlay)
        plt.subplot(4, 1, 3)
        plt.imshow(tissue_mask, cmap=plt.cm.gray)
        plt.subplot(4, 1, 4)
        plt.imshow(annot_img, cmap=plt.cm.gray)
        plt.show()
        """
    #Results
    fullprint(size_x_ls)
    fullprint(size_y_ls)
    fullprint(tissue_on_WSI_ls)
    fullprint(tumor_on_WSI_ls)
    fullprint(tumor_on_tissue_ls)
    avg_size_x = np.mean(size_x_ls)
    avg_size_y = np.mean(size_y_ls)
    avg_tissue_on_WSI = np.mean(tissue_on_WSI_ls)
    avg_tumor_on_WSI = np.mean(tumor_on_WSI_ls)
    avg_tumor_on_tissue = np.mean(tumor_on_tissue_ls)

    stdev_size_x = np.std(size_x_ls)
    stdev_size_y = np.std(size_y_ls)
    stdev_tissue_on_WSI = np.std(tissue_on_WSI_ls)
    stdev_tumor_on_WSI = np.std(tumor_on_WSI_ls)
    stdev_tumor_on_tissue = np.std(tumor_on_tissue_ls)
    print("-------------- statistical results --------------------")
    print("nb_WSI= ", nb_WSI)
    print("size_x= ", avg_size_x, " +- ", stdev_size_x)
    print("size_y= ", avg_size_y, " +- ", stdev_size_y)
    print("tissue_on_WSI= ", avg_tissue_on_WSI, " +- ", stdev_tissue_on_WSI)
    print("tumor_on_WSI= ", avg_tumor_on_WSI, " +- ", stdev_tumor_on_WSI)
    print("tumor_on_tissue= ", avg_tumor_on_tissue, " +- ", stdev_tumor_on_tissue)
    structuredArr = np.append(structuredArr, np.array([('mean all', avg_size_x, avg_size_y, avg_tissue_on_WSI, avg_tumor_on_WSI, avg_tumor_on_tissue)], dtype=dtype), axis=0)
    structuredArr = np.append(structuredArr, np.array([('stdev all', stdev_size_x, stdev_size_y, stdev_tissue_on_WSI, stdev_tumor_on_WSI, stdev_tumor_on_tissue)], dtype=dtype), axis=0)
    name_csv = os.path.join(os.getcwd(), 'dataset_stats_folder', 'dataset_stats_' + os.path.basename(dataset_path) +'.csv')
    #np.savetxt('dataset_stats_' + os.path.basename(dataset_path) +'.csv', (avg_size_x,stdev_size_x, avg_size_y,stdev_size_y, avg_tissue_on_WSI,stdev_tissue_on_WSI, avg_tumor_on_WSI, stdev_tumor_on_WSI, avg_tumor_on_tissue, stdev_tumor_on_tissue), delimiter=',')
    np.savetxt(name_csv, structuredArr, delimiter=',', fmt=['%s', '%f', '%f', '%f', '%f', '%f'], header='Name,size_x,size_y,tissue_on_WSI,tumor_on_WSI,tumor_on_tissue', comments='')

dataset_path = dataset_folder_path = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "data_set_precise")
dataset_analysis(dataset_path)