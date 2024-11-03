import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from src.keypointmapping import KeyPointMapping

#TODO this will be slow bc it is separated from the video parser
#it would be better if this was combined with the video parser
# so that the images aren't looped over multiple times or saved multiple times
# should be called from the video parser method 
class KeyPointDetector():
    """
    Generate images with key points from the images in the images folder
    """
    IMAGE_SHAPE = (256,256)

    def __init__(self):
        model = hub.load("https://www.kaggle.com/models/google/movenet/TensorFlow2/multipose-lightning/1")
        self.keyPointDetector = model.signatures['serving_default']
        self.mapping = KeyPointMapping()
        self.keyPointImages = []
        self.keyPointOverlayImages = []
    #TODO - return the images from this function so that they can be saved from the videoparser
    #or add them to a list and store as a class variable then just get them later and save them
    def generateKeyPoints(self, im: np.ndarray, keyPointPath: os.PathLike, overlayPath: os.PathLike, imNumber: int) -> None:
        #plt.imshow(np.squeeze(im.astype(np.uint8)))
        im = self._prepImage(im = im)
        outputs = self.keyPointDetector(im)

        outputArray = outputs['output_0'].numpy() #the output array is shape 1, 6, 56; thre are 17 keypoints with y, x, confidence, + 5 elements for the bounding box
        reshapedArray = outputArray[0][0][:51].reshape(17,3)
        plt.figure(figsize=(5, 5))
        
        #add keypoints
        for i in range(17):
            plt.plot(reshapedArray[i][1]*self.IMAGE_SHAPE[0], reshapedArray[i][0]*self.IMAGE_SHAPE[1], marker='o', color="red")
        #add the lines connecting the keypoints
        for l in self.mapping.LINES:
            indOne = self.mapping.KEY_POINT_NAMES.index(l[0])
            indTwo = self.mapping.KEY_POINT_NAMES.index(l[1])
            plt.plot([reshapedArray[indOne][1]*self.IMAGE_SHAPE[0], reshapedArray[indTwo][1]*self.IMAGE_SHAPE[0]],
                     [reshapedArray[indOne][0]*self.IMAGE_SHAPE[1], reshapedArray[indTwo][0]*self.IMAGE_SHAPE[1]],
                      color='k',linestyle='-')
            
            #save the frames with the keypoints? will have to create the directory
            #append the images to the self.frames list, then after 
        #generate a gif from the list of frames
        plt.axis([0, 256, 256, 0])
        plt.axis('off')
        #plt.show()
        plt.savefig(os.path.join(keyPointPath, 'frame_' + str(imNumber)+'.jpg'))
        
        #plt.imshow(np.squeeze(im))
        #plt.show()
        if overlayPath is not None:
            plt.imshow(np.squeeze(im))
            plt.savefig(os.path.join(overlayPath, 'frame_' + str(imNumber)+'.jpg'))
        #plt.show()
        plt.close()
        return None
    
    def _prepImage(self, im: np.ndarray) -> tf.Tensor:
        """
        Resize to the required shape for imagenet and 
        normalize and convert to tensor
        """
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = np.array(im)
        #im = cv2.resize(im, self.IMAGE_SHAPE) 
        #im = im/255.0
        im = tf.convert_to_tensor(im, dtype=tf.int32)
        im = tf.image.resize(im, self.IMAGE_SHAPE)
        im = tf.cast(im, tf.int32)
        im = tf.expand_dims(im, axis=0)
        return im

