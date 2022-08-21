import sys
import os
from matplotlib import pyplot as plt
import numpy as np
import tensorflow as tf
import datetime
from DataGenerator_v2 import DataGenerator_v2
from UnetSeminalModel import UnetSeminalModel
from Evaluator import Evaluator


def train_experiment(experiment):

    Model = experiment['model']
    model_str = experiment['model_str']
    mag_factor = experiment.get('mag_factor', 10)
    lr = experiment.get('lr', 1e-4)
    eps = experiment.get('eps', 1e-7)
    overlap_data = experiment.get('overlap_data', False)
    data_augmentation = experiment.get('data_augmentation', True)
    overlap_pred = experiment.get('overlap_pred', False)
    max_epochs = experiment.get('max_epochs', 100)
    patience = experiment.get('patience', 5)
    image_size = experiment.get('image_size', (256,256))
    batch_size = experiment.get('batch_size', 16)
    validation_size = experiment.get('validation_size', 0.2)

    data = "mag" + str(mag_factor) + "_" + str(image_size[0]) + "_" + str(image_size[1])
    data_folder_name = data
    if overlap_data==True:
        data += "_overlap"
    else:
        data += "_no_overlap"
    if data_augmentation==True:
        name_model = data + '_augmentation' + '_lr' + str(lr) + '_eps' + str(eps) + '_e' + str(max_epochs) + '_pat' + str(patience)
    else:
        name_model = data + '_no_augmentation' + '_lr' + str(lr) + '_eps' + str(eps) + '_e' + str(max_epochs) + '_pat' + str(patience)

    clf_name = os.path.join(os.getcwd(), "models",  model_str + name_model) #"lighter_"+

    path_to_dataset = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", data_folder_name)
    path_to_test_set = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "test_set")
    #path_to_predictions = os.path.join(os.path.dirname(os.getcwd()), "DataThesis", "predictions", os.path.basename(clf_name))
    #os.makedirs(path_to_predictions, exist_ok=True)  # Creates the folder for the predictions. If it already exists, overwrites it

    print("Path to dataset provided: ", path_to_dataset)
    print("Path to test set provided: ", path_to_test_set)
    #print("Path to predictions provided: ", path_to_predictions)
    print("clf name provided:   ", clf_name)

    #Tensorboard
    log_dir = os.path.join(os.getcwd(), "tensorboard", os.path.basename(clf_name) +"_"+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    tf.keras.backend.clear_session()

    # Training
    print("about to enter data generator")
    generator = DataGenerator_v2(batch_size, validation_size, image_size, path_to_dataset, data_augmentation, overlap_data)
    print("generator ready")

    print("about to enter model")
    model = Model(image_size, clf_name, lr=lr, eps=eps)



    print("about to enter history (model.fit())")
    history = model.fit(max_epochs, generator, tensorboard_callback, patience=patience)
    print("exited history (model.fit())")
    np.save(f"{clf_name}_history.npy", history.history, allow_pickle=True)
    print(history)

    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(history.history['val_loss'], 'r-', label="validation")
    plt.plot(history.history['loss'], 'b-', label="training")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(history.history['val_matthews_correlation'], 'r-', label="validation")
    plt.plot(history.history['matthews_correlation'], 'b-', label="training")
    plt.ylabel("MCC")
    plt.xlabel("epochs")
    plt.legend(loc='lower right')
    plt.savefig(f'{clf_name}_history.png')


    # Compute & save metrics
    print("about to compute the different metrics")
    metrics = {}
    i_bm = np.argmin(np.asarray(history.history['val_loss']))  # index best model
    print("index best model -->", i_bm)
    print("Training metrics")
    metrics["training"] = {"loss": history.history['loss'][i_bm], 'accuracy': history.history['accuracy'][i_bm],
                           'precision': history.history['precision'][i_bm], 'specificity': history.history['specificity'][i_bm],
                           'recall': history.history['recall'][i_bm], 'f1': history.history['f1'][i_bm],
                           'iou': history.history['iou'][i_bm], 'matthews_correlation': history.history['matthews_correlation'][i_bm]}
    print(metrics["training"])
    print("Validation metrics")
    metrics["validation"] = {"loss": history.history['val_loss'][i_bm],
                             'accuracy': history.history['val_accuracy'][i_bm],
                             'precision': history.history['val_precision'][i_bm],
                             'specificity': history.history['val_specificity'][i_bm],
                             'recall': history.history['val_recall'][i_bm], 'f1': history.history['val_f1'][i_bm],
                             'iou': history.history['val_iou'][i_bm], 'matthews_correlation': history.history['val_matthews_correlation'][i_bm]}
    print(metrics["validation"])

    """ 
    print("Test metrics")
    test_metrics= Evaluator.predict_WSI_and_evaluate_BASIC(model, path_to_test_set, path_to_predictions, image_size, mag_factor=mag_factor, overlap=overlap_pred)
    metrics["test"] = {'accuracy': test_metrics[0], 'precision': test_metrics[1], 'specificity': test_metrics[2],
                       'recall': test_metrics[3], 'f1': test_metrics[4], 'iou': test_metrics[5], 'matthews_correlation': test_metrics[6]}
    print(metrics["test"])
    np.save(f"{clf_name}_metrics.npy", metrics, allow_pickle=True)
    """


    with open(f"{clf_name}_metrics.txt", 'w') as fp:
        print("Number of pos tiles available: ", len(generator.pos_val) + len(generator.pos_train), file=fp)
        print("Number of neg tiles available: ", len(generator.neg_val) + len(generator.neg_train), file=fp)
        print("Nb tiles training per class: ", generator.nb_tiles_train_per_class, file=fp)
        print("Nb tiles training balanced (both classes): ", generator.nb_tiles_train_balanced, file=fp)
        print("Nb tiles validation balanced (both classes): ", generator.nb_tiles_val, file=fp)
        print(" ---- ", file=fp)
        print("Training perfomance:", file=fp)
        print("Loss\tAccuracy\tPrecision\tSpecificity\tRecall\tf1\tiou\tMCC", file=fp)
        print(metrics["training"]["loss"], metrics["training"]["accuracy"], metrics["training"]["precision"], metrics["training"]["specificity"], metrics["training"]["recall"], metrics["training"]["f1"], metrics["training"]["iou"], metrics["training"]["matthews_correlation"],file=fp)
        print(" ---- ", file=fp)
        print("Validation perfomance:", file=fp)
        print("Loss\tAccuracy\tPrecision\tSpecificity\tRecall\tf1\tiou\tMCC", file=fp)
        print(metrics["validation"]["loss"], metrics["validation"]["accuracy"], metrics["validation"]["precision"],
              metrics["validation"]["specificity"], metrics["validation"]["recall"], metrics["training"]["f1"],
              metrics["training"]["iou"], metrics["validation"]["matthews_correlation"], file=fp)
        
        #print(" ---- ", file=fp)
        #print("Test perfomance:", file=fp)
        #print("Accuracy\tPrecision\tSpecificity\tRecall\tf1\tiou\tMCC", file=fp)
        #print(metrics["test"]["accuracy"], metrics["test"]["precision"],
        #      metrics["test"]["specificity"], metrics["test"]["recall"], metrics["test"]["f1"],
        #      metrics["test"]["iou"], metrics["test"]["matthews_correlation"],file=fp)

def main():

    experiments = [
        {'model': UnetSeminalModel,
         'model_str': 'TEST_',
         'mag_factor': 5,
         'image_size': (128, 128),
         'overlap_data': True,
         'data_augmentation': True,
         'overlap_pred': False,
         'lr': 1e-4,
         'eps': 1e-7,
         'max_epochs': 100,
         'patience': 10}
]

    for experiment in experiments:
        print(f"Training {experiment}")
        train_experiment(experiment)

if __name__ == '__main__':
    main()