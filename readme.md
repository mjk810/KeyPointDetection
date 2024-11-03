## Key Point Detection

This repo will process a video to generate video frames with keypoints marked.

The process:
- A video (mp4 of mov) is read from the Videos folder and separated into frames
    - To save the frames, saveFrames should be set to True in config.json
- The tensorflow movenet model is used to detect keypoints and the keypoints and lines are draw on the frame using matplotlib
- The keypoint figure is saved each time
- The keypoint overlay images are saved if saveOverlay is set to True in the config file
- Gifs of the keypoint images and the overlay images are saved if createGif is set to True in config.json

The config file is used to specify whether to save the frames, the keypoint overlay, and the gif

## To get started
##### Create a virtual environment
`python -m venv .venv`
 ##### Activate the venv
`PowerShell -ExecutionPolicy Bypass` <br>
`.venv/Scripts/activate` <br>
 ##### Install dependencies
`pip install -r requirements.txt`
##### Save a video to the Videos folder
##### Run generateimages.py