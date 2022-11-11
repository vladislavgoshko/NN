from random import seed
from random import randrange
from random import random
from random import randint
from csv import reader
from csv import writer
import csv
from types import new_class
import numpy as np
import cv2 
from math import exp, sqrt
import os
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import matplotlib.pyplot as plt 

path=os.path.dirname(os.path.abspath(__file__))
img111 = cv2.imread("C:\\Users\\ferru\\OneDrive\\Desktop\\5semester\\NN\\4lab\\train\\table clocks\\table (1).jpg", cv2.IMREAD_GRAYSCALE)
print(len(img111))