import cv2
import os
import numpy as np
from src.objectdetector import ObjectDetector
from src.keypointdetector import KeyPointDetector
import matplotlib.pyplot as plt

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
        self.keyPointDetector = KeyPointDetector()

    def processVideos(self) -> None:
        
        for f in self.filesToLoad:
            saveDir = os.path.join(self.OUT_PATH, f.split('.')[0]) #This is something like Images/Vid_1
            os.makedirs(saveDir, exist_ok=True)
            
            video = cv2.VideoCapture(os.path.join(self.DIR_PATH, f))
            counter, success = 0, 1

            while success:
                success, im = video.read() #need to reshape; reduce size? current 828, 1792, 3
                #im = rawim.astype(float)
                
                #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                #plt.imshow(im)
                #cv2.waitKey(1)

                if im is not None:
                    im = self.formatImage(im=im)
                    #plt.imshow(im)
                    #plt.show()
                    #im = cv2.flip(im, 0)
                    #im = self.objectDetector.run(im=im)
                    #self.keypoints = self.keyPointDetector.generateKeyPoints(im=im)
                    
                    #TODO handle this differently; if you want to save the orig frame, pass something in
                    #self.saveImage(saveDir=saveDir, im=im, counter=counter) #TODO maybe have a separate class to save these
                    #imRead = cv2.imread(os.path.join('Images','Vid_1','frame_{}.jpg'.format(counter)))
                    #im = cv2.cvtColor(imRead, cv2.COLOR_BGR2RGB)
                    #plt.imshow(imRead)
                    
                    #cv2.imshow('image', im)
                    #if cv2.waitKey(1):
                    #    break
                    self.keyPointDetector.generateKeyPoints(im=im,
                                    keyPointPath=os.path.join(saveDir, 'KeyPoints'),
                                    overlayPath=os.path.join(saveDir, 'KeyPointsOverlay'),
                                    imNumber = counter)
                    counter+=1
            video.release()
            cv2.destroyAllWindows()    

        #plt.imshow(frames[0])
        #plt.show()
            #TODO could generate the gif right here is saving keypoint images as they are read
    #TODO - this method should be removed
    def formatImage(self, im: os.PathLike) -> np.ndarray:
        #im = cv2.flip(im, 0)
        #im = cv2.flip(im, 1)
        im = cv2.resize(im, self.IMAGE_SHAPE) 
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

        return im
    
    def saveImage(self, saveDir: os.PathLike, im: np.ndarray, counter: int) -> None:
        cv2.imwrite(os.path.join(saveDir, 'frame_{}.jpg'.format(counter)), im)





