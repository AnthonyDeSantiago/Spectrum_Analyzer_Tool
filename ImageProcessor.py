# Class for segregating image processing utility functions
import cv2
import numpy as np
from collections import deque


class ImageProcessor:
    """
    A class for performing various image processing

    Basically, I imagining this class to hold all or atleast most of the functionality for image processing
    Currently it only has two empty functions.
    """
    def __init__(self, sequence_length=8):
        self.sequence_length = sequence_length
        self.frame_deque = deque(maxlen=sequence_length)

    
    def isolateGrid(self, image):
        """
        Perform image processing to isolate the grid

        Args:
            image (numpy arry...I think?): Cropped image of the spectrum analyzer screen
        """
        # Perform image processing to isolate the grid
        return
    
    def isolateSignal(self, image):
        """
        Perform image processing to isolate the signal

        Args:
            image (numpy array...I think?): Cropped image of the spectrum analyzer screen
        """
        # Perform image processing to isolate the signal
        return
    
    def getBackground(self):
        background = np.median(np.array(self.frame_deque), axis=0).astype(np.uint8)
        return background
    
    def append(self, frame):
        self.frame_deque.append(frame)

    def isEmpty(self):
        isEmpty = not self.frame_deque
        return isEmpty
    
    def performDifferencing(self, background):
        resultants = []
    
        for i in self.frame_deque:
            result = cv2.absdiff(background, i)
            _, result = cv2.threshold(result, 50, 255, cv2.THRESH_BINARY)
            result = cv2.dilate(result, None, iterations=2)
            resultants.append(result)

        if len(resultants) > 0:
            cv2.imshow("resultants", resultants[0])
        return resultants
    
    def performSumming(self, resultants):
        sum = np.zeros_like(resultants[0])
        for i in resultants:
            sum = cv2.add(i, sum)
            cv2.imshow("sum", sum)
        return sum
    
