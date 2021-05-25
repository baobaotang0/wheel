import cv2
from matplotlib import pyplot

from math_tools import *


def sum_error(point_list, x: float, y: float, r: float):
    res = 0
    for p in point_list:
        one_err = (r - math.sqrt((x - p[0]) ** 2 + (y - p[1]) ** 2)) ** 2
        res += one_err
    return res


class WholeWheelFinder:
    def __init__(self, car: list):
        self.car = car
        self.p_min, self.p_max = get_min_max(car)
        self.mid = (self.p_min[2] + self.p_max[2]) / 2
        # 网格化车侧面的点云，用二值图像表示
        self.p_max[1] = 0.95
        down_car = cut_cloud(self.car, boundary={1: [None, self.p_max[1]]})  # 剪掉车的上半部分，省计算
        right_car, left_car = cut_cloud(down_car, boundary={2: [self.mid, None]}, need_rest=True)
        self.right_car = SideWheelFinder(right_car, self.p_min, self.p_max)
        self.left_car = SideWheelFinder(left_car, self.p_min, self.p_max)
        self.right_car.find_wheel()
        self.left_car.find_wheel()
        self.pixel_wheel = {1: self.left_car.wheels, 2: self.right_car.wheels}
        self.real_wheel = {1:[],2:[]}
        for bot, pw in self.pixel_wheel.items():
            for w in pw:
                self.real_wheel[bot].append([w[i]/self.right_car.extention*self.right_car.pixel_size for i in range(3)])


    def get_wheel(self):
        return self.real_wheel




class SideWheelFinder:
    pixel_size = 0.02
    extention = 3
    wheel_diam_range = [0.58 / 2, 0.90 / 2]
    pixel_wheel_diam_rang = [int(wheel_diam_range[0] / pixel_size * extention),
                             math.ceil(wheel_diam_range[1] / pixel_size * extention)]
    wheels: list
    y_limit: list
    switch_list: list


    def __init__(self, car: list, p_min: list, p_max: list):
        self.p_min, self.p_max = p_min, p_max
        self.mid = (p_max[0] - p_min[0]) / self.pixel_size * self.extention / 2
        self.img = pixel(car, self.pixel_size, self.p_min, self.p_max, darkest=1, extention=self.extention)
        self.empty_img_c = numpy.array([numpy.array([[0, 0, 0] for j in range(len(self.img[0]))], dtype=numpy.uint8)
                                        for i in range(len(self.img))], dtype=numpy.uint8)
        self.drop_wheels = []
        self.fitting_wheels = []
        self.v = []
        self.car_contours = []
        self.bds_range={}
        self.bds= {}


    def find_wheel(self):
        self.find_drop_wheel()
        if self.drop_wheels != 2:
            self.get_lower_edge()
            self.estimate_circle_center()
            from scipy.optimize import minimize
            for switch in self.switch_list:
                fitting_points = self.locate_fitting_points(switch)
                para_estimate = numpy.array([round(self.suppose_center_x[switch]), self.pixel_wheel_diam_rang[1],
                                             self.pixel_wheel_diam_rang[1]])
                a = minimize(lambda para_list: sum_error(point_list=fitting_points,
                                                         x=para_list[0], y=para_list[1], r=para_list[2]),
                             x0=para_estimate,
                             bounds=((self.bds[switch][0] + self.pixel_wheel_diam_rang[0] * 0.5,
                                      self.bds[switch][1] - self.pixel_wheel_diam_rang[0] * 0.5),
                                     (self.pixel_wheel_diam_rang[0], self.pixel_wheel_diam_rang[1]),
                                     (self.pixel_wheel_diam_rang[0], self.pixel_wheel_diam_rang[1]))
                             )
                self.fitting_wheels.append([a.x[0], a.x[1], a.x[2]])
                self.add_opencv_circle(numpy.int0((a.x[0], a.x[1], a.x[2])),(0,255,0))
            if len(self.fitting_wheels) > 1:
                self.wheels = self.fitting_wheels
            else:
                self.wheels = self.fitting_wheels + self.drop_wheels
        else:
            self.wheels = self.drop_wheels

    def find_drop_wheel(self):
        dilate = cv2.dilate(self.img, kernel_n(n=2), iterations=1)
        erosion = cv2.erode(dilate, kernel_n(n=3), iterations=1)
        whole_contours = []
        contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        for i in range(len(contours)):
            # print(cv2.contourArea(contours[i]))
            if cv2.contourArea(contours[i]) > 1000:
                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                if self.pixel_wheel_diam_rang[0] < radius < self.pixel_wheel_diam_rang[1] and \
                        self.pixel_wheel_diam_rang[0] < y < self.pixel_wheel_diam_rang[1] and \
                        abs(x - self.mid) > self.pixel_wheel_diam_rang[0] and\
                        cv2.contourArea(contours[i]) > 3000:
                    color = (153, 255, 255)
                    self.drop_wheels.append([x, y, radius])
                    (x, y, radius) = numpy.int0((x, y, radius))
                    self.add_opencv_circle((x, y, radius), (0, 0, 255))
                else:
                    self.car_contours.append(contours[i])
                    color = (255, 102, 102)
                whole_contours.append(contours[i])
                cv2.drawContours(self.empty_img_c, contours, i, color=color, thickness=-1)
        # 判断还缺前后哪个轮子
        if len(self.drop_wheels) == 2:
            self.switch_list = []
        elif len(self.drop_wheels) == 1:
            if self.drop_wheels[0][0] > self.mid:
                self.switch_list = [0]
            else:
                self.switch_list = [1]
        else:
            if len(self.drop_wheels) > 2:
                self.car_contours = whole_contours
            self.switch_list = [0, 1]

        # print(self.switch_list)
        # print("drop wheel", self.drop_wheel)
        # cv2.imshow('detected hough_wheel', self.empty_img_c)
        # cv2.waitKey(0)

    def get_lower_edge(self):
        self.y_limit = [None for j in range(len(self.img[0]))]
        for car_part in self.car_contours:
            car_part = car_part = [i[0] for i in car_part.tolist()]
            car_part.append(car_part[0])
            car_part = interpolate_by_pixel(car_part)
            for p in car_part:
                if self.y_limit[p[0]] is None or p[1] < self.y_limit[p[0]]:
                    self.y_limit[p[0]] = p[1]
        y_limit_continuous = []
        for i in range(len(self.y_limit)):
            if self.y_limit[i]:
                y_limit_continuous.append([i, self.y_limit[i]])
        y_limit_continuous = interpolate_by_stepLen(y_limit_continuous, 5)
        for i in range(len(y_limit_continuous) - 1):
            p1 = y_limit_continuous[i]
            p2 = y_limit_continuous[i + 1]
            local_v = (p2[1] - p1[1]) / (p2[0] - p1[0])
            int_x = round((p2[0] + p1[0]) / 2)
            if self.v == [] or self.v[-1][0] != int_x:
                self.v.append([int_x, local_v])
            elif abs(self.v[-1][1]) < abs(local_v):
                self.v[-1][1] = local_v
        self.v = interpolate_by_pixel(self.v, False)
        v_head = [[i, self.v[0][1]] for i in range(self.v[0][0])]  # 微分后会差两位，补全
        self.v = v_head + self.v
        # pyplot.plot(self.y_limit, "r-+")
        # new_plot(y_limit_continuous, "b*-")
        # new_plot(self.v)

    def estimate_circle_center(self):
        lowest = [[], []]
        for i in range(len(self.y_limit)):
            if self.y_limit[i] is not None:
                p = [i, self.y_limit[i]]
                if i < self.mid:
                    switch = 0
                else:
                    switch = 1
                if lowest[switch]:
                    if lowest[switch][0][1] > p[1]:
                        lowest[switch].clear()
                        lowest[switch].append(p)
                    elif lowest[switch][0][1] == p[1]:
                        lowest[switch].append(p)
                else:
                    lowest[switch].append(p)
        self.suppose_center_x = []
        for i in range(2):
            if lowest[i]:
                self.suppose_center_x.append(sum([j[0] for j in lowest[i]]) / len(lowest[i]))
            else:
                self.suppose_center_x.append(None)


    def locate_fitting_points(self, switch):
        left = [self.v[round(self.suppose_center_x[switch])], self.v[round(self.suppose_center_x[switch])]]
        right = [self.v[round(self.suppose_center_x[switch])], self.v[round(self.suppose_center_x[switch])]]
        for i in range(max(int(self.suppose_center_x[switch] - self.pixel_wheel_diam_rang[0] * 0.5), 0),
                       int(self.suppose_center_x[switch] - self.pixel_wheel_diam_rang[1] * 1.2), -1):
            if left[0][1] < self.v[i][1]:
                left[0] = self.v[i]
            if left[1][1] > self.v[i][1]:
                left[1] = self.v[i]
        for i in range(int(self.suppose_center_x[switch] + self.pixel_wheel_diam_rang[0] * 0.5),
                       min(int(self.suppose_center_x[switch] + self.pixel_wheel_diam_rang[1] * 1.2), len(self.v))):
            if right[0][1] < self.v[i][1]:
                right[0] = self.v[i]
            if right[1][1] > self.v[i][1]:
                right[1] = self.v[i]
        self.bds_range[switch] = [left, right]
        bds = [None, None]
        if left[0][0] > left[1][0] and abs(left[1][1] / left[0][1]) > 1.5:
            bds[0] = left[1][0] - 1
        else:
            bds[0] = left[0][0] + 1
        if right[0][0] > right[1][0] and abs(right[0][1] / right[1][1]) > 1.5:
            bds[1] = right[0][0] + 1
        else:
            bds[1] = right[1][0] - 1
        self.bds[switch] = bds
        fitting_points = numpy.array([[i, self.y_limit[i]] for i in range(bds[0], bds[1] + 1)])
        return fitting_points

    def plot_mosaic(self):
        pyplot.figure(figsize=(20, 5))
        c = pyplot.pcolormesh(self.img, cmap='magma')
        pyplot.colorbar(c)
        pyplot.axis("equal")
        pyplot.show()

    def plot_fitting_situation(self, switch):
        pyplot.plot(self.y_limit, "r-+")
        for i in self.car_contours:
            new_plot(i)
        new_plot([self.suppose_center_x[switch], 0], "co")
        left, right = self.bds_range[switch]
        new_plot(left, "ro")
        new_plot(right, "ro")
        new_plot([self.v[self.bds[switch][0]], self.v[self.bds[switch][1]]], "b^")
        new_plot(self.v[int(self.suppose_center_x[switch] + self.pixel_wheel_diam_rang[0] * 0.5)], "yo")
        new_plot(self.v[max(int(self.suppose_center_x[switch] - self.pixel_wheel_diam_rang[0] * 0.5), 0)], "yo")
        new_plot(self.v[int(self.suppose_center_x[switch] - self.pixel_wheel_diam_rang[1] * 1.2)], "yo")
        new_plot(self.v[min(int(self.suppose_center_x[switch] + self.pixel_wheel_diam_rang[1] * 1.2), len(self.v) - 1)],
                 "yo")
        pyplot.axis("equal")
        pyplot.show()

    def add_opencv_circle(self, circle_para, color):
        cv2.circle(self.empty_img_c, (circle_para[0], circle_para[1]), circle_para[2], color, 2)
        cv2.circle(self.empty_img_c, (circle_para[0], circle_para[1]), 2, color, 3)

    def plot_opencv(self):
        cv2.imshow('detected hough_wheel', self.empty_img_c)
        cv2.waitKey(0)
