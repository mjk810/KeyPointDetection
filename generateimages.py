from src.videoparser import VideoParser
from src.keypointdetector import KeyPointDetector
import os
import cv2
import matplotlib.pyplot as plt
from src.generategif import GenerateGif

#manually deleting frames that don't have person, so need to comment this out momentarily
#videoParser = VideoParser()
#videoParser.processVideos()

'''
#key point detector; keeping separate now so that the images without a person can be manually removed
#this could be automated using object detection in the videoparser class
#will loop over files in dir, but the dir should/could be passed as an argument; but that won't be necessary 
#if this is moved into the videoparser class
keyPoints = KeyPointDetector()
filesToLoad: list[str] = os.listdir(os.path.join('Images','Vid_1'))
#create the output directory for the keypoints
keyPointDirectory = os.path.join('Images', 'Vid_1', 'KeyPoints')
os.makedirs(keyPointDirectory, exist_ok=True)
#make a directory for the overlay
overlayDirectory = os.path.join('Images', 'Vid_1', 'KeyPointsOverlay')
os.makedirs(overlayDirectory, exist_ok=True)
#can this model process more than one image at a time? if so, that would be faster than looping here
for f in filesToLoad:
    im = cv2.imread(os.path.join('Images','Vid_1',f))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    #plt.imshow(im)
    #plt.show()
    keyPoints.generateKeyPoints(im=im, keyPointPath=os.path.join(keyPointDirectory,f.split('.')[0] + '.png'),
                                overlayPath=os.path.join(overlayDirectory,f.split('.')[0] + '.png')) #pass in save dir
'''
#could create runners for each of these (gneerate images and keypoint detectors), then from this file,
#pass in true or false if it needs to run or not as a config? or just pass the condition for now

#read the keypoint files and create a gif
gifGenerator = GenerateGif()
gifGenerator.create(imdir = os.path.join('Images','Vid_1','KeyPointsOverlay'))
