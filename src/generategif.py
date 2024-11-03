import cv2
import imageio #https://imageio.readthedocs.io/en/stable/
import os
import numpy as np

class GenerateGif():

    def generateGifFromFile(self, imdir: os.PathLike) -> None:
        images = []
        #read all the files from the directory
        filesToLoad: list[str] = os.listdir(imdir)
        for f in filesToLoad:
            im = cv2.imread(os.path.join(imdir,f))
            if im is not None:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                images.append(im)
        
        #turn into gif
        #save gif to the directory
        imageio.mimwrite(os.path.join(imdir,'pitch.gif'), images)

    def generateGifFromList(self, imageList: list, imdir: os.PathLike, config: dict[str, str]) -> None:
        figList = []
        if config['createGif'] == 'True':
            #Have to convert figure to open cv image; having to loop, this probably isnt any faster than reading the files
            for im in imageList:
                im.canvas.draw()

                # convert canvas to image
                img = np.fromstring(im.canvas.tostring_rgb(), dtype=np.uint8, sep='')
                img  = img.reshape(im.canvas.get_width_height()[::-1] + (3,))
                figList.append(img)
            imageio.mimwrite(os.path.join(imdir,'pitch.gif'), figList)