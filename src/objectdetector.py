#mobile net object detection

import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2


class ObjectDetector():
    """
    Object detection with pretrained mobilenet v2
    Crop the image to the player
    """
    IMAGE_SHAPE = (224,224) #size required by mobilenet v2 obj detection; doc says image can be variable size

    def __init__(self):
        # https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2
        self.detector = hub.load("https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/TensorFlow2/fpnlite-320x320/1")
        #print(self.detector.inputs)
        #mobilenet_v2 ="https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4"
        #self.clf = tf.keras.Sequential([hub.KerasLayer(mobilenet_v2, 
        #                                               input_shape=self.IMAGE_SHAPE+(3,))
        #                               ])
    
    def prepImage(self, im: np.ndarray) -> tf.Tensor:
        """
        Resize to the required shape for imagenet and 
        normalize and convert to tensor
        """
        im = np.array(im)
        #im = cv2.resize(im, self.IMAGE_SHAPE) 
        #im = im/255.0
        im = tf.convert_to_tensor(im, dtype=tf.float32)
        im = tf.image.resize(im, self.IMAGE_SHAPE)
        im = tf.cast(im, tf.uint8)
        im = tf.expand_dims(im, axis=0)
        return im
    
    def identifyPlayer(self):
        pass

    def cropToPlayer(self):
        pass

    #TODO this is set up for a single image, but performance would be improved by processing images in batch
    def run(self, im: np.ndarray):
        im = self.prepImage(im=im)
        result = self.detector(im)
        class_ids = result["detection_classes"]
        result = {key:value.numpy() for key,value in result.items()}

        print("Found %d objects." % len(result["detection_scores"]))
        #need to crop the image to the box with the player classification and return the cropped image
        #if it isn't detected, don't want to keep the image
        
        return result
