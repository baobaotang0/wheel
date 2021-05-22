import math
import os

import cv2
import numpy

from math_tools import *
from ply_reader import *
from matplotlib import pyplot

pixel_size = 0.02
extention = 3
wheel_diam_range = [0.58 / 2, 0.90 / 2]
pixel_wheel_diam_rang = [int(wheel_diam_range[0] / pixel_size * extention),
                         math.ceil(wheel_diam_range[1] / pixel_size * extention)]
print(pixel_wheel_diam_rang)
for path in path_iter():
    floor = ply_loader(path)
    floor = [[i[0], i[2]] for i in floor]

    p_min, p_max = get_min_max(floor)
    print(p_min, p_max)
    up_car, down_car = cut_cloud(floor, boundary={1:[0, None]},need_rest=True)
    car = []
    if len(down_car)> 10000:
        car += down_car
    else:
        p_min[1] = 0
    if len(up_car) > 10000:
        car += up_car
    else:
        p_max[1] = 0

    mosaic_matrix = pixel(car, pixel_size, p_min, p_max, darkest=1, extention=extention)
    # mosaic_matrix = reverse_black_white(mosaic_matrix)
    #
    pyplot.figure(figsize=(20,5))
    c = pyplot.pcolormesh(mosaic_matrix, cmap='magma')
    pyplot.colorbar(c)
    pyplot.axis("equal")
    pyplot.show()

    img = mosaic_matrix
    empyt_img_bw = numpy.array([numpy.array([[0] for j in range(len(mosaic_matrix[0]))], dtype=numpy.uint8)
                                for i in range(len(mosaic_matrix))], dtype=numpy.uint8)
    empyt_img_c = numpy.array(
        [numpy.array([[0, 0, 0] for j in range(len(mosaic_matrix[0]))], dtype=numpy.uint8)
         for i in range(len(mosaic_matrix))], dtype=numpy.uint8)
    def kernel_n(n):
        return numpy.ones((n, n), dtype=numpy.uint8)
    dilate = cv2.dilate(img, kernel_n(5), iterations=1)
    erosion = cv2.erode(dilate, kernel_n(5), iterations=1)
    # erosion = cv2.erode(erosion, kernel_n(3), iterations=1)
    erosion = cv2.bitwise_not(erosion)
    ss = numpy.hstack((img, erosion))
    cv2.imshow('cleaner', ss)
    cv2.waitKey(0)
    contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    hierarchy = numpy.squeeze(hierarchy)

    contours_idx = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 1000:
            contours_idx.append(i)
            color = (255, 102, 102)
            shadow_bw = cv2.drawContours(empyt_img_bw, contours, i, color=255, thickness=-1)
            shadow_c = cv2.drawContours(empyt_img_c, contours, i, color=color, thickness=-1)

    cv2.imshow('car', empyt_img_c)
    cv2.waitKey(0)