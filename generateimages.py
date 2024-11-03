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




if __name__=="__main__":
    run()