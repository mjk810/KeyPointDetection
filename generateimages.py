from src.videoparser import VideoParser
from src.keypointdetector import KeyPointDetector
import os
import cv2
import matplotlib.pyplot as plt
from src.generategif import GenerateGif
import json


def run():
    #load config file
    print(os.getcwd())
    with open(os.path.join("src", "configuration", "config.json"), 'r') as f:
        config = json.load(f)
    
    #manually deleting frames that don't have person, so need to comment this out momentarily
    videoParser = VideoParser(config = config)
    videoParser.processVideos()

    if config["createGif"]=='True':
        #read the keypoint files and create a gif TODO integrate this so that all the files don't need to be read again
        gifGenerator = GenerateGif()
        gifGenerator.create(imdir = os.path.join('Images','Vid_1','KeyPointsOverlay'))

#could create runners for each of these (gneerate images and keypoint detectors), then from this file,
#pass in true or false if it needs to run or not as a config? or just pass the condition for now



if __name__=="__main__":
    run()