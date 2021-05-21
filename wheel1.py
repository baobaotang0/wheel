import math
import os

import cv2
import numpy

from math_tools import build_cicle, new_plot, interpolate_by_x
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
                count = 0
                drop_wheel = []
                big_contours_idx = []
                biggest_contours = None
                biggest_contours_idx = None
                for i in range(len(contours)):
                    if cv2.contourArea(contours[i]) > 1000:
                        big_contours_idx.append(i)
                        count += 1
                        shadow_bw = cv2.drawContours(empyt_img_bw, contours, i, color=255, thickness=-1)
                        shadow_c = cv2.drawContours(empyt_img_c, contours, i, color=255, thickness=-1)
                        if cv2.contourArea(contours[i]) < 10000:
                            (x, y), radius = cv2.minEnclosingCircle(contours[i])
                            if radius < 60:
                                drop_wheel.append([x, y, radius])
                                (x, y, radius) = numpy.int0((x, y, radius))
                                cv2.circle(empyt_img_c, (x, y), radius, (0, 0, 255), 2)
                                cv2.circle(empyt_img_c, (x, y), 2, (0, 0, 255), 3)
                        else:
                            biggest_contours =[i[0] for i in  contours[i]]
                            biggest_contours.append(biggest_contours[0])
                            biggest_contours_idx = i
                            biggest_contours = interpolate_by_x(biggest_contours)
                            y_limit = [None for j in range(len(mosaic_matrix[0]))]
                            for p in biggest_contours:
                                if y_limit[p[0]] and p[1] < y_limit[p[0]] or y_limit[p[0]] is None:
                                    y_limit[p[0]] = p[1]
                            pyplot.plot(y_limit,"r*-")
                            from scipy.signal import savgol_filter
                            savgol = savgol_filter(y_limit, 31, 3)
                            pyplot.plot(savgol, "g-")

                            pyplot.axis("equal")
                            # pyplot.show()


                print("drop wheel", drop_wheel)

                real_wheel = []

                # 根据等高线查找霍夫圆，这一步的目的是和上一步通过腐蚀剥离的块的外接圆比较，如果两者相似度高，则认为外接圆是正确的轮子。
                # 因此如果，上一步没有剥离出外接圆，则没有必要进行霍夫圆查找

                hough_wheel = cv2.HoughCircles(empyt_img_bw, cv2.HOUGH_GRADIENT, dp=1, minDist=300,
                                               param1=50, param2=5, minRadius=pixel_wheel_diam_rang[0],
                                               maxRadius=pixel_wheel_diam_rang[1])
                if hough_wheel is not None:
                    hough_wheel = numpy.uint16(numpy.around(hough_wheel))
                    for i in hough_wheel[0, :]:
                        cv2.circle(empyt_img_c, (i[0], i[1]), i[2], (0, 255, 0), 2)
                        cv2.circle(empyt_img_c, (i[0], i[1]), 2, (0, 255, 0), 3)
                        for dw in drop_wheel:
                            if abs(dw[1] - i[1]) < (dw[2] + i[2]) / 2 * 0.3 and \
                                    abs(dw[2] - i[2]) < min(dw[2], i[2]) * 0.5 and \
                                    abs(dw[0] - i[0]) < (dw[2] + i[2]) / 2 * 0.5:
                                real_wheel.append(dw)
                                break
                # 如果只比对除了一个正确的轮子，则以这个正确的轮子为基准比较其他的外接圆和霍夫圆，如果相似度够高，则取为另一个轮子
                if len(real_wheel) == 1:
                    sample_wheel = real_wheel[0]
                    print("real wheel11111", real_wheel)
                    drop_wheel.remove(sample_wheel)
                    for w in drop_wheel:
                        if abs(w[0] - sample_wheel[0]) > 300 and abs(w[1] - sample_wheel[1]) < sample_wheel[2] * 0.2 \
                                and abs(w[2] - sample_wheel[2]) < sample_wheel[2] * 0.2:
                            real_wheel.append(w)
                            drop_wheel.remove(w)
                            break
                    if len(real_wheel) < 2:
                        for w in hough_wheel[0, :]:
                            if abs(w[0] - sample_wheel[0]) > 300 and abs(w[1] - sample_wheel[1]) < sample_wheel[
                                2] * 0.2 \
                                    and abs(w[2] - sample_wheel[2]) < sample_wheel[2] * 0.1:
                                real_wheel.append(w)
                                break
                real_wheel = numpy.array(real_wheel, dtype=numpy.uint16)
                for w in real_wheel:
                    cv2.circle(empyt_img_c, (w[0], w[1]), w[2],
                               (0, 255, 255), 2)
                    cv2.circle(empyt_img_c, (w[0], w[1]), 2, (0, 255, 255), 3)
                if len(real_wheel) == 2:
                    continue
                # 如果还是没有找齐两个轮子，则对等高线进行圆拟合
                if len(real_wheel) < 2:
                    outline = []
                    # outline = [j[0] for j in biggest_contours.tolist()]
                    # 把等高线的点提出来，作为轮廓线
                    for i in big_contours_idx:
                        outline += [j[0] for j in contours[i].tolist()]
                    # 将轮廓线分成左右两半，并找最低点
                    mid = (p_max[0] - p_min[0]) / pixel_size * extention / 2
                    outline_split = [[], []]
                    lowest = [[], []]
                    highest = [[], []]
                    for p in outline:
                        if p[0] < mid:
                            switch = 0
                        else:
                            switch = 1
                        outline_split[switch].append(p)
                        if lowest[switch]:
                            if lowest[switch][0][1] > p[1]:
                                lowest[switch].clear()
                                lowest[switch].append(p)
                            elif lowest[switch][0][1] == p[1]:
                                lowest[switch].append(p)
                        else:
                            lowest[switch].append(p)
                        if p[1] < 90:
                            if highest[switch]:
                                if highest[switch][0][1] < p[1]:
                                    highest[switch].clear()
                                    highest[switch].append(p)
                                elif highest[switch][0][1] == p[1]:
                                    highest[switch].append(p)
                            else:
                                highest[switch].append(p)
                    print("low", lowest)
                    print("high", highest)

                    # 如果左轮已经找到，则左边不找，如果右轮已经找到，则右边不找
                    if len(real_wheel) == 1:
                        if real_wheel[0][0] > mid:
                            switch_list = [0]
                        else:
                            switch_list = [1]
                    else:
                        switch_list = [0, 1]
                    suppose_center_x = [sum([i[0] for i in lowest[0]]) / len(lowest[0]),
                                        sum([i[0] for i in lowest[1]]) / len(lowest[1])]
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
                        # 将
                        for p in outline_split[switch]:
                            if abs(p[0] - suppose_center_x[switch]) < (pixel_wheel_diam_rang[1] +
                                                                       pixel_wheel_diam_rang[0]) / 2 and \
                                    p[1] < pixel_wheel_diam_rang[1]:
                                fitting_points[switch].append(p)

                        p = numpy.array(fitting_points[switch])
                        para_estimate = numpy.array([round(suppose_center_x[switch]), pixel_wheel_diam_rang[1],
                                                     pixel_wheel_diam_rang[1]])
                        print("**", para_estimate)
                        a = minimize(lambda para_list: sum_error(point_list=fitting_points[switch],
                                                                  x=para_list[0], y=para_list[1], r=para_list[2]),
                                      x0 = para_estimate,
                                      bounds=((suppose_center_x[switch] - (pixel_wheel_diam_rang[1] + pixel_wheel_diam_rang[0]) / 2,
                                               suppose_center_x[switch] + (pixel_wheel_diam_rang[1] + pixel_wheel_diam_rang[0]) / 2),
                                              (pixel_wheel_diam_rang[0], pixel_wheel_diam_rang[1]),
                                              (pixel_wheel_diam_rang[0], pixel_wheel_diam_rang[1]))
                                      )
                        print(a.x)
                        # if a[1] - a[2] > lowest[switch][0][1]:
                        #     a[1] = lowest[switch][0][1] + a[2]
                        fitting_circle.append([a.x[0], a.x[1], a.x[2]])
                    print("fc", fitting_circle)


                    new_plot(outline)
                    for i in range(len(fitting_circle)):
                        new_plot(fitting_points[switch], "r")
                        new_plot(build_cicle([fitting_circle[i][0], fitting_circle[i][1]], fitting_circle[i][2]))
                    pyplot.axis("equal")
                    pyplot.show()
                    fitting_circle = numpy.array(fitting_circle, dtype=numpy.uint16)
                    for c in fitting_circle:
                        cv2.circle(empyt_img_c, (c[0], c[1]), c[2], (255, 255, 0), 2)
                        cv2.circle(empyt_img_c, (c[0], c[1]), 2, (255, 255, 0), 3)

                cv2.imshow('detected hough_wheel', empyt_img_c)
                cv2.waitKey(0)
