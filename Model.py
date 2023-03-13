import tensorflow as tf
from skimage.measure import label, regionprops
import numpy as np
import tensorflow.keras.backend as K

def recall(y_true, y_pred):
    # converting the prediction to numpy to deduce the mask and then converting it back to a tensor
    y_pred = tf.convert_to_tensor(np.argmax(y_pred.numpy(), axis=3), dtype=tf.float32)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall_keras = true_positives / (possible_positives + K.epsilon())
    return recall_keras


def precision(y_true, y_pred):
    #converting the prediction to numpy to deduce the mask and then converting it back to a tensor
    y_pred = tf.convert_to_tensor(np.argmax(y_pred.numpy(), axis=3), dtype=tf.float32)

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_keras = true_positives / (predicted_positives + K.epsilon())
    return precision_keras

def specificity(y_true, y_pred):
    #Taken from https://medium.com/analytics-vidhya/custom-metrics-for-keras-tensorflow-ae7036654e05
    # converting the prediction to numpy to deduce the mask and then converting it back to a tensor
    y_pred = tf.convert_to_tensor(np.argmax(y_pred.numpy(), axis=3), dtype=tf.float32)
    tn = K.sum(K.round(K.clip((1 - y_true) * (1 - y_pred), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    return tn / (tn + fp + K.epsilon())

def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def iou(y_true, y_pred): #TODO
    # tp/tp+FN+FP
    # converting the prediction to numpy to deduce the mask and then converting it back to a tensor
    y_pred = tf.convert_to_tensor(np.argmax(y_pred.numpy(), axis=3), dtype=tf.float32)
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    fp = K.sum(K.round(K.clip((1 - y_true) * y_pred, 0, 1)))
    fn = K.sum(K.round(K.clip(y_true * (1-y_pred), 0, 1)))
    iou_keras = tp/(tp+fn+fp)
    return iou_keras

def matthews_correlation(y_true, y_pred):
    # converting the prediction to numpy to deduce the mask and then converting it back to a tensor
    y_pred = tf.convert_to_tensor(np.argmax(y_pred.numpy(), axis=3), dtype=tf.float32)
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


class Model():
    """Build & use DCNN model.
    Includes post-processing.
    """

    def __init__(self, image_size, clf_name, loadFrom=None, lr=1e-4, eps=1e-8):
        """Load existing model from hdf5 file or build it from scratch."""
        self.image_size = image_size
        self.clf_name = clf_name
        if( loadFrom == None ):
            self.set_model()
            opt = tf.keras.optimizers.Adam(
                learning_rate=lr,
                epsilon=eps,
                name='Adam')
            self.model.compile(
                optimizer=opt,
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=[tf.keras.losses.SparseCategoricalCrossentropy(name='crossentropy'), 'accuracy', precision, specificity, recall, f1, iou, matthews_correlation], #tf.keras.losses.BinaryCrossentropy(name='crossentropy'),tf.keras.metrics.BinaryAccuracy(name = 'accuracy'),tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5, name='IoU'), tf.keras.metrics.Precision(name = 'precision'), tf.keras.metrics.Recall(name = 'recall'),tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
                run_eagerly=True)
        else:
            self.model = tf.keras.models.load_model(loadFrom, compile=False, custom_objects={'leaky_relu': tf.nn.leaky_relu})

    def print(self):
        """Display model summary"""
        self.model.summary()

    def plot(self):
        """Generates plot of model architecture and saves it to model.png file"""
        tf.keras.utils.plot_model(self.model, show_shapes=True)

    def save(self, fname):
        """Save whole model to file"""
        self.model.save(fname)

    def fit(self, n_epochs, dataset, tensorboard_callback, patience=15):
        """Fit the model on the dataset with EarlyStopping on the validation crossentropy"""
        print("Model: entered in fit() ")
        validation_data = None
        callbacks = []
        callbacks = [tf.keras.callbacks.ModelCheckpoint(f"{self.clf_name}.hdf5", save_best_only=True),
                     tf.keras.callbacks.EarlyStopping(monitor='val_crossentropy', patience=patience), #val_crossentropy
                     tensorboard_callback] #modif pour apres faire un generateur sur validation_data

        print("about to enter into return self.model.fit()")
        return self.model.fit(
            dataset.next_balanced_batch_BASIC(n_epochs),
            epochs=n_epochs,
            steps_per_epoch=dataset.batches_per_epoch,
            validation_data=dataset.get_validation_data_BASIC(),
            validation_steps= dataset.nb_tiles_val//dataset.batch_size,
            callbacks=callbacks
            )


    def predict(self, data):
        return self.model.predict(data)

    @staticmethod
    def post_process(pred, min_area=100):
        print("entered into Model.post_process()")
        """Label binary mask, then remove small objects & close holes"""
        pred_mask = np.argmax(pred, axis=2)
        lab = label(pred_mask)
        for obj in regionprops(lab):
            if( obj.area < min_area ):
                lab[lab==obj.label] = 0
            else:
                region = lab[obj.bbox[0]:obj.bbox[2],obj.bbox[1]:obj.bbox[3]]
                region[obj.filled_image] = obj.label

        return lab
