import cv2
import imageio #https://imageio.readthedocs.io/en/stable/
import os

class GenerateGif():

    def create(self, imdir: os.PathLike) -> None:
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