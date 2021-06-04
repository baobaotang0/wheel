import math
import numpy
import cv2
from matplotlib import pyplot



def new_plot(l: list, style=None):
    if l is not []:
        if style:
            if isinstance(l[0], float) or isinstance(l[0], int):
                pyplot.plot(l[0], l[1], style)
            else:
                pyplot.plot([p[0] for p in l], [p[1] for p in l], style)
        else:
            if isinstance(l[0], float) or isinstance(l[0], int):
                pyplot.plot(l[0], l[1])
            else:
                pyplot.plot([p[0] for p in l], [p[1] for p in l])

def plot_mosaic(pictures):
    from matplotlib import pyplot
    pyplot.figure(figsize=(20, 5))
    c = pyplot.pcolormesh(pictures, cmap='magma')
    pyplot.colorbar(c)
    pyplot.axis("equal")
    pyplot.show()

def plot_opencv(picture):
    cv2.imshow('detected hough_wheel', picture)
    cv2.waitKey(0)

def add_opencv_circle(picture,circle_para, color):
    cv2.circle(picture, (circle_para[0], circle_para[1]), circle_para[2], color, 2)
    cv2.circle(picture, (circle_para[0], circle_para[1]), 2, color, 3)

def build_cicle(center: list, radius: float, lineNum=12):
    theta = numpy.linspace(0, math.pi * 2, lineNum)
    res = []
    for i in theta:
        res.append([center[0] + math.cos(i) * radius, center[1] + math.sin(i) * radius])
    return res