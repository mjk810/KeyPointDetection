from src.videoparser import VideoParser
from src.generategif import GenerateGif
import os
import json


def run():
    #load config file
    with open(os.path.join("src", "configuration", "config.json"), 'r') as f:
        config = json.load(f)
    
    #manually deleting frames that don't have person, so need to comment this out momentarily
    videoParser = VideoParser(config = config)
    videoParser.processVideos()

    if config["createGif"]=='True':
        #read the keypoint files and create a gif TODO integrate this so that all the files don't need to be read again
        gifGenerator = GenerateGif()
        gifGenerator.create(imdir = os.path.join('Images','Vid_1','KeyPointsOverlay'))
        gifGenerator.create(imdir = os.path.join('Images','Vid_1','KeyPoints'))



if __name__=="__main__":
    run()