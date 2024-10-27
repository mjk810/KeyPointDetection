import cv2
import os
import numpy as np
from src.objectdetector import ObjectDetector
from src.keypointdetector import KeyPointDetector


class VideoParser():
    """
    Load the video and convert it to single frames and save
    Use to create single frames from the video; 
    """
    DIR_PATH = 'Videos'
    OUT_PATH = 'Images'
    IMAGE_SHAPE = (256,256) 

    def __init__(self):
        self.filesToLoad: list[str] = os.listdir(self.DIR_PATH)
        #self.objectDetector = ObjectDetector()
        #self.keyPointDetector = KeyPointDetector()

    def processVideos(self) -> None:
        for f in self.filesToLoad:
            saveDir = os.path.join(self.OUT_PATH, f.split('.')[0]) #This is something like Images/Vid_1
            os.makedirs(saveDir, exist_ok=True)
            
            video = cv2.VideoCapture(os.path.join(self.DIR_PATH, f))
            counter, success = 0, 1

            while success:
                success, im = video.read() #need to reshape; reduce size? current 828, 1792, 3
                if im is not None:
                    im = self.formatImage(im=im)
                    #im = cv2.flip(im, 0)
                    #im = self.objectDetector.run(im=im)
                    #self.keypoints = self.keyPointDetector.generateKeyPoints(im=im)
                    self.saveImage(saveDir=saveDir, im=im, counter=counter) #TODO maybe have a separate class to save these
                
                    counter+=1
            #TODO could generate the gif right here is saving keypoint images as they are read
    #TODO - this method should be removed
    def formatImage(self, im: os.PathLike) -> np.ndarray:
        im = cv2.flip(im, 0)
        im = cv2.flip(im, 1)
        im = cv2.resize(im, self.IMAGE_SHAPE) 

        return im
    
    def saveImage(self, saveDir: os.PathLike, im: np.ndarray, counter: int) -> None:
        cv2.imwrite(os.path.join(saveDir, 'frame_{}.jpg'.format(counter)), im)





