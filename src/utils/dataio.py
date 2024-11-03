import matplotlib.pyplot as plt
import os
import numpy as np
import cv2


class DataIO():

    @staticmethod
    def makeNewDirectory(self):
        pass

    @staticmethod
    def saveFile(self):
        pass

    @staticmethod
    def saveImage(im: np.ndarray, filePath: os.PathLike, counter: int) -> None:
        cv2.imwrite(os.path.join(filePath, 'Frame_'+str(counter)+'.jpg'), im)

    @staticmethod
    def saveImages(images: list, filePath: os.PathLike) -> None:
        """
        Save a list of images to the specified filepath
        """
        os.makedirs(filePath, exist_ok=True)
        for idx, fig in enumerate(images):
            fname = os.path.join(filePath, 'frame_' + str(idx)+'.jpg')
            fig.savefig(fname)