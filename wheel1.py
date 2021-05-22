import math
import os

import cv2
import numpy

from math_tools import build_cicle, new_plot, interpolate_by_stepLen, interpolate_by_pixel
from ply_reader import *
from matplotlib import pyplot

def get_min_max_3d(car):
    p_min = car[0].copy()
    p_max = car[0].copy()
    for p in car:
        for i in range(3):
            if p[i] < p_min[i]:
                p_min[i] = p[i]
            if p[i] > p_max[i]:
                p_max[i] = p[i]
    return p_min, p_max


def cut_2dcar(car: list, idx: int, limit: list):
    res = []
    for p in car:
        if limit[0] <= p[idx] <= limit[1]:
            res.append([p[0], p[1]])
    return res


def pixel(car: list, pixel_size, p_min: list, p_max: list, darkest: float, extention=1, colored=False):
    resolution = [math.ceil((p_max[0] - p_min[0]) / pixel_size), math.ceil((p_max[1] - p_min[1]) / pixel_size)]
    res = [[0 for j in range(extention * (resolution[0] + 1))] for i in range(extention * (resolution[1] + 1))]
    for p in car:
        i = int((p[0] - p_min[0]) / pixel_size)
        j = int((p[1] - p_min[1]) / pixel_size)
        for k in range(extention):
            for l in range(extention):
                res[extention * j + k][extention * i + l] += 1
    for i in range(resolution[1]):
        for j in range(resolution[0]):
            if res[extention * i][extention * j] > darkest:
                for k in range(extention):
                    for l in range(extention):
                        res[extention * i + k][extention * j + l] = 255
            else:
                for k in range(extention):
                    for l in range(extention):
                        res[extention * i + k][extention * j + l] = int(
                            255 / darkest * res[extention * i][extention * j])
    if colored:
        for i in range(len(res)):
            for j in range(len(res[0])):
                res[i][j] = [res[i][j]] * 3
    res = numpy.array(res, dtype=numpy.uint8)
    return res


def takeFirst(elem):
    return elem[0]
a =[[0,0],[1,1]]
a.sort(key=takeFirst)

if __name__ == '__main__':
    pixel_size = 0.02
    extention = 3
    wheel_diam_range = [0.58 / 2, 0.90 / 2]
    pixel_wheel_diam_rang = [int(wheel_diam_range[0] / pixel_size * extention),
                             math.ceil(wheel_diam_range[1] / pixel_size * extention)]
    print(pixel_wheel_diam_rang)
    folder_path = "cars/"
    car_id = os.listdir(folder_path)
    for i in car_id:
        # if i not in ["car_17.npy"]:
        #     continue
        if i.endswith("npy"):
            print(i)
            path2d = "cars/" + i
            with open(path2d, 'rb') as f_pos:
                car = list(numpy.load(f_pos))
                p_min, p_max = get_min_max_3d(car)
                mid = (p_min[2] + p_max[2]) / 2
                half_car = cut_2dcar(car, idx=2, limit=[mid, p_max[2]])
                p_max[1] = 0.95
                half_car = cut_2dcar(half_car, idx=1, limit=[p_min[1], p_max[1]])
                # vtktool.vtk_show(car)
                mosaic_matrix = pixel(half_car, pixel_size, p_min, p_max, darkest=1, extention=extention)
                # pyplot.figure(figsize=(20,5))
                # c = pyplot.pcolormesh(mosaic_matrix, cmap='magma')
                # pyplot.colorbar(c)
                # pyplot.axis("equal")
                # pyplot.show()

                img = mosaic_matrix
                empyt_img_bw = numpy.array([numpy.array([[0] for j in range(len(mosaic_matrix[0]))], dtype=numpy.uint8)
                                            for i in range(len(mosaic_matrix))], dtype=numpy.uint8)
                empyt_img_c = numpy.array(
                    [numpy.array([[0, 0, 0] for j in range(len(mosaic_matrix[0]))], dtype=numpy.uint8)
                     for i in range(len(mosaic_matrix))], dtype=numpy.uint8)
                kernel_2 = numpy.ones((2, 2), dtype=numpy.uint8)
                kernel_3 = numpy.ones((3, 3), dtype=numpy.uint8)
                kernel_4 = numpy.ones((4, 4), dtype=numpy.uint8)
                dilate = cv2.dilate(img, kernel_2, iterations=1)
                erosion = cv2.erode(dilate, kernel_3, iterations=1)
                ss = numpy.hstack((img, erosion))
                # cv2.imshow('cleaner', ss)
                # cv2.waitKey(0)
                # image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
                hierarchy = numpy.squeeze(hierarchy)
                drop_wheel = []
                contours_idx = {"wheel":[],"car":[]}
                for i in range(len(contours)):
                    if cv2.contourArea(contours[i]) > 1000:
                        (x, y), radius = cv2.minEnclosingCircle(contours[i])
                        if pixel_wheel_diam_rang[0] < radius < pixel_wheel_diam_rang[1] and \
                                pixel_wheel_diam_rang[0] < y < pixel_wheel_diam_rang[1]:
                            contours_idx["wheel"].append(i)
                            color = (153, 255, 255)
                            drop_wheel.append([x, y, radius])
                            (x, y, radius) = numpy.int0((x, y, radius))
                            cv2.circle(empyt_img_c, (x, y), radius, (0, 0, 255), 2)
                            cv2.circle(empyt_img_c, (x, y), 2, (0, 0, 255), 3)
                        else:
                            contours_idx["car"].append(i)
                            color = (255,102,102)
                        shadow_bw = cv2.drawContours(empyt_img_bw, contours, i, color=255, thickness=-1)
                        shadow_c = cv2.drawContours(empyt_img_c, contours, i, color=color, thickness=-1)

                print("drop wheel", drop_wheel)
                # cv2.imshow('detected hough_wheel', empyt_img_c)
                # cv2.waitKey(0)
                # 如果还是没有找齐两个轮子，则对等高线进行圆拟合
                if len(drop_wheel)<2:
                    y_limit = [None for j in range(len(mosaic_matrix[0]))]
                    for i in contours_idx["car"]:
                        car_part = [i[0] for i in contours[i]]
                        car_part.append(car_part[0])
                        car_part = interpolate_by_pixel(car_part)
                        for p in car_part:
                            if y_limit[p[0]] is None or p[1] < y_limit[p[0]]:
                                y_limit[p[0]] = p[1]

                    y_limit_continuous = []
                    for i in range(len(y_limit)):
                        if y_limit[i]:
                            y_limit_continuous.append([i, y_limit[i]])
                    y_limit_continuous = interpolate_by_stepLen(y_limit_continuous, 5)
                    pyplot.plot(y_limit,"rs-")
                    new_plot(y_limit_continuous, "b*-")
                    v = []
                    for i in range(len(y_limit_continuous)-1):
                        p1 = y_limit_continuous[i]
                        p2 = y_limit_continuous[i+1]
                        local_v = (p2[1] - p1[1]) / (p2[0] - p1[0])
                        int_x = round((p2[0] + p1[0]) / 2)
                        if v == [] or v[-1][0] != int_x :
                            v.append([int_x, local_v])
                        elif abs(v[-1][1]) < abs(local_v):
                            v[-1][1] = local_v
                    v = interpolate_by_pixel(v, False)
                    v_head = [[i, v[0][1]] for i in range(v[0][0])]
                    v= v_head+v
                    from scipy.signal import savgol_filter
                    savgol_v = savgol_filter([i[1] for i in v], 11, 5)
                    pyplot.plot(savgol_v, "g-")

                    new_plot(v)






                    # 将轮廓线分成左右两半，并找最低点
                    mid = (p_max[0] - p_min[0]) / pixel_size * extention / 2
                    outline_split = [[], []]
                    lowest = [[], []]
                    print(len(y_limit))
                    for i in range(len(y_limit)):
                        if y_limit[i]:
                            p = [i, y_limit[i]]
                            if i < mid:
                                switch = 0
                            else:
                                switch = 1
                            outline_split[switch].append(p)
                            if lowest[switch]:
                                if lowest[switch][0][1] > p[1]:
                                    print(lowest[switch])
                                    print(p)
                                    lowest[switch].clear()
                                    lowest[switch].append(p)
                                elif lowest[switch][0][1] == p[1]:
                                    lowest[switch].append(p)
                            else:
                                lowest[switch].append(p)
                    print("low", lowest)
                    suppose_center_x = [sum([i[0] for i in lowest[0]]) / len(lowest[0]),
                                        sum([i[0] for i in lowest[1]]) / len(lowest[1])]
                    # 如果左轮已经找到，则左边不找，如果右轮已经找到，则右边不找
                    if len(drop_wheel) == 1:
                        if drop_wheel[0][0] > mid:
                            switch_list = [0]
                        else:
                            switch_list = [1]
                    else:
                        switch_list = [0, 1]
                    print(switch_list)

                    # 找轮子
                    fitting_circle = []


                    def sum_error(point_list: list, x: float, y: float, r: float):
                        res = 0
                        for p in point_list:
                            one_err = (r - math.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)) ** 2
                            res += one_err
                        return res


                    from scipy.optimize import minimize

                    fitting_points = [[], []]
                    for switch in switch_list:
                        left,right = v[round(suppose_center_x[switch])], v[round(suppose_center_x[switch])]
                        for i in range(max(int(suppose_center_x[switch] - pixel_wheel_diam_rang[0]*0.5),0),
                                        int(suppose_center_x[switch] - pixel_wheel_diam_rang[1]*1.2), -1):
                            if left[1] > v[i][1]:
                                left = v[i-1] # 寻找斜率最低点，左侧n个点通常是开始上升的地方，n粗略取1
                        for i in range(int(suppose_center_x[switch] + pixel_wheel_diam_rang[0] * 0.5),
                                    min(int(suppose_center_x[switch] + pixel_wheel_diam_rang[1] * 1.2),len(v))):
                            if right[1] < v[i][1]:
                                right = v[i+1] # 寻找斜率最高点，右侧n个点通常是开始下降的地方，n粗略取1
                        print()
                        new_plot([suppose_center_x[switch],0],"go")
                        new_plot(left,"ro")
                        new_plot(right, "ro")
                        new_plot(v[int(suppose_center_x[switch] + pixel_wheel_diam_rang[0] * 0.5)],"yo")
                        new_plot(v[max(int(suppose_center_x[switch] - pixel_wheel_diam_rang[0]*0.5),0)], "yo")
                        new_plot(v[int(suppose_center_x[switch] - pixel_wheel_diam_rang[1]*1.2)], "yo")
                        new_plot(v[min(int(suppose_center_x[switch] + pixel_wheel_diam_rang[1] * 1.2),len(v))], "yo")

                        fitting_points[switch] = numpy.array([[i,y_limit[i]]for i in range(left[0],right[0]+1)])
                        print()
                        para_estimate = numpy.array([round(suppose_center_x[switch]), pixel_wheel_diam_rang[1],
                                                     pixel_wheel_diam_rang[1]])
                        print("**", para_estimate)
                        a = minimize(lambda para_list: sum_error(point_list=fitting_points[switch],
                                                                  x=para_list[0], y=para_list[1], r=para_list[2]),
                                      x0 = para_estimate,
                                      bounds=((left[0] + pixel_wheel_diam_rang[0]*0.5, right[0] - pixel_wheel_diam_rang[0]*0.5),
                                              (pixel_wheel_diam_rang[0], pixel_wheel_diam_rang[1]),
                                              (pixel_wheel_diam_rang[0], pixel_wheel_diam_rang[1]))
                                      )
                        print(a.x)
                        # if a[1] - a[2] > lowest[switch][0][1]:
                        #     a[1] = lowest[switch][0][1] + a[2]
                        fitting_circle.append([a.x[0], a.x[1], a.x[2]])
                    print("fc", fitting_circle)


                    outline = []
                    for i in contours_idx["car"]:
                        outline += [i[0] for i in contours[i]]
                    new_plot(outline)
                    for i in range(len(fitting_circle)):
                        # new_plot(fitting_points[switch], "y")
                        new_plot(build_cicle([fitting_circle[i][0], fitting_circle[i][1]], fitting_circle[i][2]))
                        new_plot(build_cicle([fitting_circle[i][0], fitting_circle[i][1]], 3))
                    pyplot.axis("equal")
                    pyplot.show()
                    fitting_circle = numpy.array(fitting_circle, dtype=numpy.uint16)
                    for c in fitting_circle:
                        cv2.circle(empyt_img_c, (c[0], c[1]), c[2], (0, 100, 0), 2)
                        cv2.circle(empyt_img_c, (c[0], c[1]), 2, (0, 100, 0), 3)
                #
                # cv2.imshow('detected hough_wheel', empyt_img_c)
                # cv2.waitKey(0)
