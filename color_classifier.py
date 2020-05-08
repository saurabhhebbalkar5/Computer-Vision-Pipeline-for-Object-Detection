# import the necessary packages
import numpy as np
import argparse
import cv2
from multiprocessing import Queue

# Class to detect color of the car
class Color_Classifier():

	def __init__(self):
		# define the list of boundaries for each color
		self.boundaries = {
					'red': ([117,0,112], [255,19,255]),
					'black': ([0, 0, 0], [18,18,57]),
					'white':([0, 0, 180], [10, 20, 255]),
					'blue':([90,75,20],[125,255,255]),
					'silver':([95,10,20],[115,50,255])}

	def detect_color(self, image):
		# convert to hsv format
		hsv = cv2.cvtColor(np.float32(image), cv2.COLOR_RGB2HSV)
		max= 0
		for key, value in self.boundaries.items():
			lower = np.array(value[0], dtype = "uint8")
			upper = np.array(value[1], dtype = "uint8")
			# create mask in range of the color obtained
			mask = cv2.inRange(hsv, lower, upper)
			# get the pixel count of each color obtained
			pixel_count = np.sum(mask)
			# max pixel color is returned as the color of the car
			if (max < pixel_count):
				max = pixel_count
				color = key
		return color