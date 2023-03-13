import tensorflow as tf

from Model import Model

#inspired by https://pyimagesearch.com/2022/02/21/u-net-image-segmentation-in-keras/
def conv_block(x, size_filters):
    print("dans conv_block")
    x_ = tf.keras.layers.Conv2D(size_filters, 3, activation=tf.nn.leaky_relu, padding='same', kernel_initializer="he_normal")(x)
    x_ = tf.keras.layers.Conv2D(size_filters, 3, activation=tf.nn.leaky_relu, padding='same', kernel_initializer="he_normal")(x_)
    return x_ #tf.keras.layers.concatenate([x,x_]) #handmade short-skip connection


def downsample_block(x, size_filters):
    print("entre dans downsample_block")
    x = conv_block(x, size_filters)
    p = tf.keras.layers.MaxPool2D(2)(x)
    #p = tf.keras.layers.Dropout(0.1)(p)
    return x, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    print("entre dans upsample block")
    #
    x = tf.keras.layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = tf.keras.layers.concatenate([x, conv_features])
    #x = tf.keras.layers.Dropout(0.1)(x)
    # Conv2D twice with ReLU activation
    x = conv_block(x, n_filters)
    return x


class UnetSeminalModel(Model):

    def __init__(self, image_size, clf_name, loadFrom=None, lr=1e-4, eps=1e-8):
        super().__init__(image_size, clf_name, loadFrom=loadFrom, lr=lr, eps=eps)



    def set_model(self):
        n_filters = 64
        inputs = tf.keras.Input(shape=self.image_size+(3,))
        
        x1, p1 = downsample_block(inputs, n_filters)
        x2, p2 = downsample_block(p1, n_filters*2)
        x3, p3 = downsample_block(p2, n_filters*4)
        x4, p4 = downsample_block(p3, n_filters*8)
        bottom = conv_block(p4, n_filters*16)
        u4 = upsample_block(bottom, x4, n_filters*8)
        u3 = upsample_block(u4, x3, n_filters*4)
        u2 = upsample_block(u3, x2, n_filters*2)
        u1 = upsample_block(u2, x1, n_filters)
        
        outputs = tf.keras.layers.Conv2D(2, 1, activation=tf.nn.softmax, padding='same', kernel_initializer="he_normal")(u1) #2,1 #added padding same recently
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs)
        self.model.summary()
