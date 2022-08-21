import sys
import os
import numpy as np
from scipy.stats import mode
from matplotlib import pyplot as plt
from skimage.transform import resize, downscale_local_mean
from skimage.color import rgb2gray, rgba2rgb
import tensorflow as tf

from Model import Model

#from dataset_prep import generate_pair_data_names, extract_tissue_mask
from utils import *
from data_extraction import get_image_anno_tissue

def get_conf_matrix_only_tissue_BASIC(Y_pred_mask,Y_gt, tissue_mask):
    # Confusion matrix, only computed where there is tissue
    TP = float(((Y_gt == True) * (Y_pred_mask == True) * (tissue_mask==True)).sum())
    FP = float(((Y_gt == False) * (Y_pred_mask == True)* (tissue_mask==True)).sum())
    FN = float(((Y_gt == True) * (Y_pred_mask == False)* (tissue_mask==True)).sum())
    TN = float(((Y_gt == False) * (Y_pred_mask == False)* (tissue_mask==True)).sum())
    return np.array([TP, FP, TN, FN])


def one_prediction_pass(model, img, tissue_mask, coords_begin, tile_size, min_tissue_area, overlap = False):
    Y_pred_prob = np.zeros(tissue_mask.shape)
    #confusion_matrix_WSI= np.zeros(4)
    nb_tiles_y = (img.shape[0]-coords_begin[0]) // tile_size[0]
    nb_tiles_x = (img.shape[1]-coords_begin[1]) // tile_size[1]
    print("nb_tiles_x= ", nb_tiles_x)
    print("nb_tiles_y= ", nb_tiles_y)
    for y in range(nb_tiles_y):
        for x in range(nb_tiles_x):
            y_up = y * tile_size[0] + coords_begin[0]
            y_down = (y + 1) * tile_size[0] + coords_begin[0]
            x_left = x * tile_size[1] + coords_begin[1]
            x_right = (x + 1) * tile_size[1] + coords_begin[1]
            tile_tissue_mask = tissue_mask[y_up:y_down, x_left:x_right]
            if tile_tissue_mask.sum() > min_tissue_area:
                # the tile contains tissue --> prediction
                X = np.zeros((1,) + tile_size + (3,))
                tissue_tile = (img[y_up:y_down, x_left:x_right]) / 255
                X[0] = tissue_tile
                pred_tissue_tile = model.predict(X)[0]  # tissue_tile:
                pred_tile_prob = pred_tissue_tile[:, :,1]  # convert_pred_to_probability_map_BASIC(pred_tissue_tile)
                #pred_tile_mask = pred_tile_prob > 0.5
                Y_pred_prob[y_up:y_down, x_left:x_right] = pred_tile_prob #np.rint(pred_tile_prob*100).astype('uint8')
                x=0
                #gt_tile_mask = Y_gt[y_up:y_down, x_left:x_right]
                #confusion_matrix_WSI += get_contribution_confusion_matrix(gt_tile_mask, pred_tile_mask)
            else:
                # the tile doesn't contain tissue --> tile considered as non-tumorous
                # nothing to do because initialized to zero
                pass
    return Y_pred_prob #, Y_pred_prob#, confusion_matrix_WSI #if overlap=True, Y_pred is a prob map and Y_pred_prob is irrelevant


def get_div_factor(shape, tile_size):
    div_factor = np.ones(shape)*4
    #corners of ones
    div_factor[0:int(tile_size[0]/2), 0:int(tile_size[1]/2)] = 1
    div_factor[div_factor.shape[0]- int(tile_size[0] / 2):, 0:int(tile_size[1] / 2)] = 1
    div_factor[0:int(tile_size[0] / 2), div_factor.shape[1]-int(tile_size[1] / 2):] = 1
    div_factor[div_factor.shape[0] - int(tile_size[0] / 2):,div_factor.shape[1]-int(tile_size[1] / 2):] = 1
    #alleys of twos
    div_factor[int(tile_size[0] / 2):div_factor.shape[0] - int(tile_size[0] / 2), 0:int(tile_size[1] / 2)] = 2
    div_factor[int(tile_size[0] / 2):div_factor.shape[0] - int(tile_size[0] / 2), div_factor.shape[1] - int(tile_size[1] / 2):] = 2
    div_factor[0:int(tile_size[0] / 2), int(tile_size[1] / 2):div_factor.shape[1]-int(tile_size[1] / 2)] = 2
    div_factor[div_factor.shape[0] - int(tile_size[0] / 2):,int(tile_size[1] / 2):div_factor.shape[1]-int(tile_size[1] / 2)] = 2
    return div_factor

def get_contribution_confusion_matrix(Y_gt, Y_pred):
    # Confusion matrix
    TP = float(((Y_gt == True) * (Y_pred == True)).sum())
    FP = float(((Y_gt == False) * (Y_pred == True)).sum())
    FN = float(((Y_gt == True) * (Y_pred == False)).sum())
    TN = float(((Y_gt == False) * (Y_pred == False)).sum())
    return np.array([TP, FP, TN, FN])
def get_metrics_BASIC(confusion_matrix):
    """Compute evaluation metrics (detection precision/recall + segmentation MCC) on a
    pair of ground truth labels / predicted labels.
    """
    # Confusion matrix
    TP = confusion_matrix[0]
    FP = confusion_matrix[1]
    TN = confusion_matrix[2]
    FN = confusion_matrix[3]

    # Metrics
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    if (TP + FP)!=0:
        precision = TP/(TP + FP)
    else:
        precision = 0
    if (TN+FP)!=0:
        specificity = TN/(TN + FP)
    else:
        specificity = 0
    if (TP+FN) !=0:
        recall = TP/(TP + FN)
    else:
        recall = 0
    if (precision + recall)!=0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0
    if (TP + FN + FP)!=0:
        iou = TP / (TP + FN + FP)
    else:
        iou=0
    if np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))!=0:
        MCC = ((TP * TN) - (FP * FN)) /(np.sqrt(TP + FP) * np.sqrt(TP + FN)* np.sqrt(TN + FP) * np.sqrt(TN + FN))
    else:
        MCC = 0
    # Printing the results
    print("TP", TP)
    print("FP", FP)
    print("FN", FN)
    print("TN", TN)
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Specificity: ", specificity)
    print("Recall: ", recall)
    print("f1: ", f1)
    print("iou: ", iou)
    print("MCC: ", MCC)

    return [accuracy, precision, specificity, recall, f1, iou, MCC]

def predict_WSI_BASIC(model, img, tissue_mask, Y_gt, tile_size, min_tissue_percentage_area=0.3, overlap=False):
    min_tissue_area = min_tissue_percentage_area * tile_size[0] * tile_size[1]
    if overlap == False:
        coords_begin = [0,0]
        Y_pred_prob = one_prediction_pass(model, img, tissue_mask, coords_begin, tile_size, min_tissue_area)
        Y_pred_mask = (Y_pred_prob>0.5)
    elif overlap==True:
        Y_pred_prob = np.zeros(tissue_mask.shape)
        #confusion_matrix_WSI = np.zeros(4)
        div_factor = get_div_factor(Y_gt.shape, tile_size)
        coords = [[0, 0], [0, int(tile_size[1] / 2)], [int(tile_size[0] / 2), 0], [int(tile_size[0] / 2), int(tile_size[1] / 2)]]
        for coords_begin in coords:
            Y_pred_temp = one_prediction_pass(model, img, tissue_mask, coords_begin, tile_size, min_tissue_area) #, confusion_matrix_WSI_temp
            Y_pred_prob+= Y_pred_temp
            #confusion_matrix_WSI += confusion_matrix_WSI_temp
        Y_pred_prob/=div_factor
        Y_pred_mask = (Y_pred_prob>0.5)

    if (Y_pred_prob.shape[0]>31600 or Y_pred_prob.shape[1]>31600): #requires too much memory (only happens for the biggest file) -> put a dummy prob map
        Y_pred_prob_downscaled = np.array([[1,0],[0,1]])
    else:
        Y_pred_prob_downscaled = downscale_local_mean(Y_pred_prob, (4, 4)) #(4,4) to downscale with a factor 4

    Y_pred_mask*= tissue_mask.astype('bool')

    # Modif for getting cropped imgs on erroneous WSIs
    save_folder_path = os.path.join(os.path.dirname(os.getcwd()), "Images mémoire", "cropped images") #Chap 7 Results

    name = "20CU017239-Ki67.ndpi"
    upper_left = [1,1]
    lower_right = [6105,6705]

    img = []
    Y_gt = []
    tissue_mask = []
    #cropped_pred_mask = Y_pred_mask[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]].astype("uint8")
    # save cropped predictions
    save_annot(Y_pred_mask[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]].astype("uint8"), os.path.join(save_folder_path, name + "_cropped_pred_mask.png"))

    upper_left = [int(upper_left[0]/ 4), int(upper_left[1]/4)]
    lower_right = [int(lower_right[0]/4), int(lower_right[1]/4)]
    #cropped_prob_map = Y_pred_prob_downscaled[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]
    save_probmap(Y_pred_prob_downscaled[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]], os.path.join(save_folder_path, name + "_cropped_prob_map.png"))  # _downscaled
    # #confusion_matrix_WSI = get_conf_matrix_only_tissue_BASIC(Y_pred_mask, Y_gt, tissue_mask)
    #metrics_WSI = get_metrics_BASIC(confusion_matrix_WSI)
    return Y_pred_mask.astype("uint8"), Y_pred_prob_downscaled #, metrics_WSI #Y_pred_prob_downscaled

class Evaluator:
    #[accuracy, precision, specificity, recall, f1, iou, MCC]
    @classmethod

    def predict_WSI_and_evaluate_BASIC(cls, model, path_test_folder, path_pred_folder, tile_size, mag_factor = 10, overlap = False):
        metrics_test_set_summary = []
        pred_model_name = os.path.basename(path_pred_folder)
        with open(f"{path_pred_folder}_test_metrics.txt", 'w') as fp:
            print(f"Prediction of {pred_model_name}: ", file=fp)
            print("Testing perfomance:", file=fp)
            print("name, accuracy, precision, specificity, recall, f1, iou, MCC", file=fp)

        path_pairs_X_Y = generate_pair_data_ndpi_ndpa(path_test_folder, [".ndpi", ".ndpa"], verbose=True)
        X_files_path = get_column(path_pairs_X_Y, 0) #[path_pairs_X_Y[i][0] for i in range(len(path_pairs_X_Y))]
        Y_files_path = get_column(path_pairs_X_Y, 1)
        for X_path, Y_path in zip(X_files_path, Y_files_path):
            print("Predicting " + X_path + "\n")
            X, Y, tissue_mask = get_image_anno_tissue(X_path, Y_path, mag=mag_factor, verbose=False)

            basename = os.path.basename(X_path)
            path_name = os.path.join(path_pred_folder, basename)
            save_img(X, path_name + "_img.jpeg") #"_img.png"
            save_annot(Y, path_name + "_mask.png")
            save_annot(tissue_mask, path_name + "_tissue_mask.png")
            Y_pred_mask, Y_pred_prob_downscaled = predict_WSI_BASIC(model, X, tissue_mask, Y>0, tile_size, 0.3, overlap=overlap) # , metrics_WSI#  Y_pred_prob,, Y_pred_prob_downscaled
            confusion_matrix_WSI = get_conf_matrix_only_tissue_BASIC(Y_pred_mask, Y>0, tissue_mask)
            metrics_WSI = get_metrics_BASIC(confusion_matrix_WSI)
            metrics_test_set_summary.append(metrics_WSI)
            with open(f"{path_pred_folder}_test_metrics.txt", 'a') as fp:
                print(basename, ",", metrics_WSI[0], ",", metrics_WSI[1], ",", metrics_WSI[2], ",", metrics_WSI[3], ",",
                      metrics_WSI[4], ",", metrics_WSI[5], ",", metrics_WSI[6], file=fp)

            #print("Y_pred_prob_downscaled --> ", Y_pred_prob_downscaled)
            #print("Y_pred_prob_downscaled type --> ", type(Y_pred_prob_downscaled))
            #print("Y_pred_prob_downscaled shape ---> ", Y_pred_prob_downscaled.shape)
            save_annot(Y_pred_mask, path_name + "_pred_mask.png")
            save_probmap(Y_pred_prob_downscaled, path_name + "_prob_map.png") #_downscaled
            #metrics += [cls._get_metrics_WSI(Y > 0, Y_pred_mask)]

            X=[]
            Y=[]
            tissue_mask = []
            Y_pred_mask = []
            Y_pred_prob = []

        #metrics_test_set_summary = np.mean(np.asarray(metrics), axis=0)
        return metrics_test_set_summary

def test_pred():
    mag = 5 #10
    tile_size = (128,128) #(256,256)
    overlap_pred = True
    experiment_model = "Seminalmag5_128_128_no_overlap_augmentation_lr0.0001_eps1e-07_e100_pat10"
    clf_name = experiment_model +".hdf5"
    path_to_model = os.path.join(os.getcwd(),"models", clf_name) # "back_up_prev_exp",
    path_test_folder = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "only_one_WSI_annot") # "test_set"
    pred_folder_name = experiment_model
    if overlap_pred==True:
        pred_folder_name = pred_folder_name + "_overlap_pred"
    else:
        pred_folder_name = pred_folder_name

    path_pred_folder = os.path.join(os.getcwd(), "test_folder", pred_folder_name) #os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "predictions", pred_folder_name)
    os.makedirs(path_pred_folder, exist_ok=True)  # Creates the folder for the predictions. If it already exists, overwrites it
    print("path to model ---> ", path_to_model)
    print("path_test_folder ---> ", path_test_folder)
    print("path_to_predictions ---> ", path_pred_folder)

    model = Model(tile_size, path_to_model, loadFrom=path_to_model)
    #metrics = Evaluator.predict_WSI_and_evaluate(model, path_test_folder, path_pred_folder, path_metrics_folder, tile_size, mag_factor=10, classification_segmentation="segmentation")
    metrics = Evaluator.predict_WSI_and_evaluate_BASIC(model, path_test_folder, path_pred_folder, tile_size, mag_factor=mag, overlap = overlap_pred)
    #(cls, model, path_test_folder, path_pred_folder, tile_size, mag_factor = 1.25, overlap = False):
    print("mean metrics: ", metrics)

def test_tissue_mask():
    tile_size = (256,256) #(512,512)
    #model_name = "base_model_tile.hdf5"
    model_name = "mag10_256_256_no_overlap_no_augmentation_lr0.0001_eps1e-07_e2_pat5.hdf5"
    path_to_model = os.path.join(os.getcwd(),"models", model_name)
    print("path to model ---> ", path_to_model)
    path_test_folder =  os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "only_one_WSI_annot")
    path_pred_folder =  os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "predictions")
    #path_metrics_folder = path_pred_folder#os.path.join(os.getcwd(), "fake_test_set")
    model = Model(tile_size, path_to_model, loadFrom=path_to_model)
    Evaluator.predict_WSI_and_evaluate_BASIC(model, path_test_folder, path_pred_folder,tile_size, mag_factor = 10)

def small_test():
    tile_size = (256, 256)  # (512,512)
    # model_name = "base_model_tile.hdf5"
    model_name = "UnetSimple_model_256_256_mag10_BASIC_full_data_augm_lr_1e-4.hdf5"
    path_to_model = os.path.join(os.getcwd(), "models", model_name)
    print("path to model ---> ", path_to_model)
    path_test_folder = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "test_set")
    path_pred_folder = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "predictions")
    path_metrics_folder = path_pred_folder  # os.path.join(os.getcwd(), "fake_test_set")
    model = Model(tile_size, path_to_model, loadFrom=path_to_model)

def test_model_on_test_set(test_conditions):
    mag = test_conditions['mag']#5 #10
    tile_size = test_conditions['tile_size']#(128,128) #(256,256)
    overlap_pred = test_conditions['overlap_pred'] #False
    experiment_model = test_conditions['experiment_model']#"Seminal_mag5_128_128_overlap_augmentation_lr0.0001_eps1e-07_e100_pat10"
    path_test_folder = test_conditions['path_test_folder']#os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "only_one_WSI_annot") # "test_set"
    path_predictions_folder = test_conditions['path_predictions_folder']
    clf_name = experiment_model +".hdf5"
    path_to_model = os.path.join(os.getcwd(),"models", clf_name) # "back_up_prev_exp",
    pred_folder_name = experiment_model
    if overlap_pred==True:
        pred_folder_name = pred_folder_name + "_overlap_pred"
    else:
        pred_folder_name = pred_folder_name

    path_pred_model_folder = os.path.join(path_predictions_folder, pred_folder_name) #os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "predictions", pred_folder_name)
    os.makedirs(path_pred_model_folder, exist_ok=True)  # Creates the folder for the predictions. If it already exists, overwrites it
    print("path to model ---> ", path_to_model)
    print("path_test_folder ---> ", path_test_folder)
    print("path_to_predictions ---> ", path_pred_model_folder)

    model = Model(tile_size, path_to_model, loadFrom=path_to_model)
    # metrics = Evaluator.predict_WSI_and_evaluate(model, path_test_folder, path_pred_folder, path_metrics_folder, tile_size, mag_factor=10, classification_segmentation="segmentation")
    metrics_summary = Evaluator.predict_WSI_and_evaluate_BASIC(model, path_test_folder, path_pred_model_folder, tile_size,
                                                       mag_factor=mag, overlap=overlap_pred)
    metrics_mean = np.mean(np.asarray(metrics_summary), axis=0)
    metrics_std = np.std(np.asarray(metrics_summary), axis=0)
    print("mean metrics: ", metrics_mean)
    with open(f"{path_pred_model_folder}_test_metrics_summary.txt", 'w') as fp:
        print(f"Prediction of {os.path.basename(path_pred_model_folder)} (summary): ", file=fp)
        print("Testing perfomance (summary):", file=fp)
        print("name, accuracy, precision, specificity, recall, f1, iou, MCC", file=fp)
        print("Mean", ",", metrics_mean[0], ",", metrics_mean[1], ",", metrics_mean[2], ",", metrics_mean[3], ",",
              metrics_mean[4], ",", metrics_mean[5], ",", metrics_mean[6], file=fp)
        print("Std", ",", metrics_std[0], ",", metrics_std[1], ",", metrics_std[2], ",", metrics_std[3], ",",
              metrics_std[4], ",", metrics_std[5], ",", metrics_std[6], file=fp)


def results_on_test_set():
    #test_pred()
    #test_tissue_mask()
    mag = 5  # 10
    tile_size = (64, 64)  # (256,256)
    overlap_pred = True
    experiment_model = "Seminal_mag5_64_64_overlap_augmentation_lr0.0001_eps1e-07_e100_pat10"
    path_test_folder = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "test_set")
    path_predictions_folder = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "predictions")

    test_conditions = {'mag': mag, 'tile_size': tile_size, 'overlap_pred': overlap_pred,'experiment_model': experiment_model,
                       'path_test_folder': path_test_folder, 'path_predictions_folder': path_predictions_folder}
    test_model_on_test_set(test_conditions)

def pred_erroneous_tiles():
    from skimage.color import rgb2gray
    folder_path = os.path.join("D:\\Nicolas", "Sauvegarde pred et metrics", "Seminal_mag5_256_256_overlap_augmentation_lr0.0001_eps1e-07_e100_pat10_overlap_pred")
    name_WSI = "20CU017239-Ki67" # "19h11173-Ki67"  "20CU003108-Ki67" "20CU006014-Ki67 ndpi"  "20CU017239-Ki67"
    annot_path = os.path.join(folder_path, name_WSI+".ndpi_mask.png")
    pred_path = os.path.join(folder_path, name_WSI+".ndpi_pred_mask.png")
    tissue_path = os.path.join(folder_path, name_WSI+".ndpi_tissue_mask.png")

    annot_mask = load_annot(annot_path)
    pred_mask = load_annot(pred_path)
    tissue_mask = (rgb2gray(load_img(tissue_path)))>0.5
    pred_mask_corrected = pred_mask*tissue_mask
    save_annot(pred_mask_corrected, pred_path + "_corrected.png")

    confusion_matrix = get_conf_matrix_only_tissue_BASIC(pred_mask,annot_mask, tissue_mask)
    metrics = get_metrics_BASIC(confusion_matrix)
    print(metrics)
    return None

def retrieve_cropped_imgs():
    #TODO: once cropping is ok, do the pred with overlapping predictions
    mag_factor = 5
    tile_size = (128,128)
    path_to_model = os.path.join(os.getcwd(),"models", "Seminal_mag5_128_128_overlap_augmentation_lr0.0001_eps1e-07_e100_pat10.hdf5")
    model = Model(tile_size, path_to_model, loadFrom=path_to_model)
    folder_path = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "test_set")
    """
    19h11173-Ki67.ndpi with [1,1] [3089,3965]
    20CU003108-Ki67.ndpi with [1,1] [3985,4793]
    20CU006014-Ki67 ndpi.ndpi with [1,1] [6000,7665]
    20CU017239-Ki67.ndpi with  [1,1] [6105,6705]
    """

    name = "20CU017239-Ki67.ndpi"
    upper_left = [1,1]
    lower_right = [6105,6705]

    mag_factor = 5
    fpath = os.path.join(folder_path, name)  # "20CU003064-Ki67.ndpi"
    apath = os.path.join(folder_path, name+".ndpa")  # "20CU003064-Ki67.ndpi.ndpa"
    save_folder_path = os.path.join(os.path.dirname(os.getcwd()), "Images mémoire", "cropped images")

    print("Predicting " + name + "\n")
    X, Y, tissue_mask = get_image_anno_tissue(fpath, apath, mag=mag_factor, verbose=False)
    #cropped_img = X[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]
    #cropped_mask = Y[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]]
    save_img(X[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]], os.path.join(save_folder_path, name + "_cropped_img.png"))
    save_annot(Y[upper_left[0]:lower_right[0], upper_left[1]:lower_right[1]], os.path.join(save_folder_path,name + "_cropped_mask.png"))
    cropped_img= []
    cropped_mask=[]
    #save_img(X, path_name + "_img.jpeg")  # "_img.png"
    #save_annot(Y, path_name + "_mask.png")
    #save_annot(tissue_mask, path_name + "_tissue_mask.png")
    Y_pred_mask, Y_pred_prob_downscaled = predict_WSI_BASIC(model, X, tissue_mask, Y > 0, tile_size, 0.3, overlap=True)  # , metrics_WSI#  Y_pred_prob,, Y_pred_prob_downscaled


def model_info():
    mag_factor = 5
    tile_size = (128, 128)
    path_to_model = os.path.join(os.getcwd(),"models", "Seminal_mag5_128_128_overlap_augmentation_lr0.0001_eps1e-07_e100_pat10.hdf5")
    model1 = Model(tile_size, path_to_model, loadFrom=path_to_model)
    model1.print()
if __name__ == '__main__':
    model_info()

