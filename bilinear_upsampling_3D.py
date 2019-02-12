import keras.backend as K
import tensorflow as tf
from keras.layers import *

def resize_images_bilinear(X, height_factor=1, width_factor=1, depth_factor=1, target_height=None, target_width=None, target_depth=None, data_format='default'):
    '''Resizes the images contained in a 4D tensor of shape
    - [batch, channels, height, width] (for 'channels_first' data_format)
    - [batch, height, width, channels] (for 'channels_last' data_format)
    by a factor of (height_factor, width_factor). Both factors should be
    positive integers.
    '''
    if data_format == 'default':
        data_format = K.image_data_format()
    if data_format == 'channels_first':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[2:4]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.transpose(X, [0, 4, 2, 3, 1])
        S = K.int_shape(X)
        x = tf.reshape(X, [S[1], S[2], S[3],S[4]])
        X = tf.image.resize_bilinear(X, new_shape)
        X = tf.reshape(X, [S[0], S[1], target_height, target_with, S[4]])
        X = tf.transpose(X, [0, 4, 2, 3, 1])
        if target_height and target_width and target_depth:
            X.set_shape((None, None, target_height, target_width, target_depth))
        else:
            X.set_shape((None, None, original_shape[2] * height_factor, original_shape[3] * width_factor, original_shape[4]*depth_factor))
        return X
    elif data_format == 'channels_last':
        original_shape = K.int_shape(X)
        if target_height and target_width:
            new_shape = tf.constant(np.array((target_height, target_width)).astype('int32'))
        else:
            new_shape = tf.shape(X)[1:3]
            new_shape *= tf.constant(np.array([height_factor, width_factor]).astype('int32'))
        X = tf.transpose(X, [0, 3, 1, 2, 4])
        S = K.int_shape(X)
        X = tf.squeeze(X,[0])
        X = tf.image.resize_bilinear(X, new_shape)
        X = tf.transpose(X,[1, 2, 0, 3])
        X = tf.expand_dims(X,0)
        if target_height and target_width and target_depth:
            X.set_shape((None, target_height, target_width, target_depth, None))
        else:
            X.set_shape((None, original_shape[1] * height_factor, original_shape[2] * width_factor, original_shape[3] * depth_factor,None))
        return X
    else:
        raise Exception('Invalid data_format: ' + data_format)

class BilinearUpSampling3D(Layer):
    def __init__(self, size=(1, 1, 1), target_size=None, data_format='default', **kwargs):
        if data_format == 'default':
            data_format = K.image_data_format()
        self.size = tuple(size)
        if target_size is not None:
            self.target_size = tuple(target_size)
        else:
            self.target_size = None
        assert data_format in {'channels_last', 'channels_first'}, 'data_format must be in {tf, th}'
        self.data_format = data_format
        self.input_spec = [InputSpec(ndim=4)]
        super(BilinearUpSampling3D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_first':
            width = int(self.size[0] * input_shape[2] if input_shape[2] is not None else None)
            height = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            depth = int(self.size[1] * input_shape[4] if input_shape[4] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
                depth = self.target_size[2]
            return (input_shape[0],
                    input_shape[1],
                    width,
                    height,depth)
        elif self.data_format == 'channels_last':
            width = int(self.size[0] * input_shape[1] if input_shape[1] is not None else None)
            height = int(self.size[1] * input_shape[2] if input_shape[2] is not None else None)
            depth = int(self.size[1] * input_shape[3] if input_shape[3] is not None else None)
            if self.target_size is not None:
                width = self.target_size[0]
                height = self.target_size[1]
                depth = self.target_size[2]
            return (input_shape[0],
                    width,
                    height,
                    depth,
                    input_shape[4])
        else:
            raise Exception('Invalid data_format: ' + self.data_format)

    def call(self, x, mask=None):
        if self.target_size is not None:
            return resize_images_bilinear(x, target_height=self.target_size[0], target_width=self.target_size[1], target_depth=self.target_size[2], data_format=self.data_format)
        else:
            return resize_images_bilinear(x, height_factor=self.size[0], width_factor=self.size[1],target_depth=self.target_size[2], data_format=self.data_format)

    def get_config(self):
        config = {'size': self.size, 'target_size': self.target_size}
        base_config = super(BilinearUpSampling3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
