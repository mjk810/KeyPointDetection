import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from src.keypointmapping import KeyPointMapping

class KeyPointDetector():
    """
    Generate images with key points from the images in the images folder
    """
    IMAGE_SHAPE = (256,256)

    def __init__(self):
        model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
        self.keyPointDetector = model.signatures['serving_default']
        self.mapping = KeyPointMapping()
        self._keyPointImages = []
        self._keyPointOverlayImages = []

    def generateKeyPoints(self, im: np.ndarray, saveOverlay: str) -> None:
        """Method that will process the image to include keypoints and lines

        Args:
            im (np.ndarray): a cv2 image
            saveOverlay (str): True or False (str not bool) as to
            whether the overlay images should be saved or not

        Returns:
            None: No return value
        """
        im = self._prepImage(im = im)
        outputs = self.keyPointDetector(im)

        outputArray = outputs['output_0'].numpy() #the output array is shape 1, 6, 56; thre are 17 keypoints with y, x, confidence, + 5 elements for the bounding box
        reshapedArray = outputArray[0][0][:51].reshape(17,3)
        
        #TODO this should be a method so it isn't duplicated
        f = plt.figure(figsize=(5, 5))
        new_plot = f.add_subplot(111)
        self._addKeyPoints(subPlot = new_plot, reshapedArray = reshapedArray)
        plt.axis([0, 256, 256, 0])
        plt.axis('off')
        self._keyPointImages.append(f)
        plt.close()

        #create the overlay plot
        if saveOverlay == 'True':
            f2 = plt.figure(figsize=(5, 5))
            new_plot2 = f2.add_subplot(111)
            self._addKeyPoints(subPlot = new_plot2, reshapedArray = reshapedArray)
            new_plot2.imshow(np.squeeze(im))
            plt.axis([0, 256, 256, 0])
            plt.axis('off')
            self._keyPointOverlayImages.append(f2)
            plt.close()
        
        return None
    
    def _addKeyPoints(self, subPlot: plt.Axes, reshapedArray: np.ndarray) -> plt.Axes:
        """Add the keypoints and lines to the image

        Args:
            subPlot (plt.Axes): A matplotlib subplot
            reshapedArray (np.ndarray): _description_

        Returns:
            plt.Axes: The axes updated with the key points
        """
        #add keypoints
        for i in range(17):
            subPlot.plot(reshapedArray[i][1]*self.IMAGE_SHAPE[0], reshapedArray[i][0]*self.IMAGE_SHAPE[1], marker='o', color="red")
        #add the lines connecting the keypoints
        for l in self.mapping.LINES:
            indOne = self.mapping.KEY_POINT_NAMES.index(l[0])
            indTwo = self.mapping.KEY_POINT_NAMES.index(l[1])
            subPlot.plot([reshapedArray[indOne][1]*self.IMAGE_SHAPE[0], reshapedArray[indTwo][1]*self.IMAGE_SHAPE[0]],
                     [reshapedArray[indOne][0]*self.IMAGE_SHAPE[1], reshapedArray[indTwo][0]*self.IMAGE_SHAPE[1]],
                      color='k',linestyle='-')
            
        return subPlot
    
    def _prepImage(self, im: np.ndarray) -> tf.Tensor:
        """
        Resize to the required shape for imagenet and 
        normalize and convert to tensor
        """
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.array(im)
        im = tf.convert_to_tensor(im, dtype=tf.int32)
        im = tf.image.resize(im, self.IMAGE_SHAPE)
        im = tf.cast(im, tf.int32)
        im = tf.expand_dims(im, axis=0)
        return im

    @property
    def keyPointImages(self):
        return self._keyPointImages

    @property
    def keyPointOverlayImages(self):
        return self._keyPointOverlayImages