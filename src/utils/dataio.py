import matplotlib.pyplot as plt
import os


class DataIO():

    @staticmethod
    def makeNewDirectory(self):
        pass

    @staticmethod
    def saveFile(self):
        pass

    @staticmethod
    def saveImages(images: list, filePath: os.PathLike) -> None:
        """
        Save a list of images to the specified filepath
        """
        for idx, fig in enumerate(images):
            fname = os.path.join(filePath, 'frame_' + str(idx)+'.jpg')
            fig.savefig(fname)