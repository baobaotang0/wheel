import cv2
import math
import numpy
from matplotlib import pyplot
from scipy.optimize import minimize
from math_tools import get_min_max, cut_cloud, pixel, kernel_n, plot_mosaic, interpolate_by_pixel, new_plot, \
    interpolate_by_stepLen, split_list, plot_opencv, add_opencv_circle, get_lower_edge, sum_error, build_cicle


class QuarterWheelFinder:
    pixel_size = 0.02
    extention = 3
    wheel_diam_range = [0.55 / 2, 0.90 / 2]
    pixel_wheel_diam_rang = [int(wheel_diam_range[0] / pixel_size * extention),
                             math.ceil(wheel_diam_range[1] / pixel_size * extention)]
    area_filter = 1000  # 不能小于2个像素点，这样边界会只有3个点而无法进行圆拟合
    circle_area_filter = pixel_wheel_diam_rang[0] ** 2 * math.pi / 2

    def __init__(self, car: list, is_left: bool):
        self.car = car
        self.p_min, self.p_max = get_min_max(self.car)
        self.p_max_pixel = [(self.p_max[i] - self.p_min[i]) / self.pixel_size * self.extention for i in range(3)]
        self.proportion = max(self.p_max_pixel[0], self.p_max_pixel[1]) / (
                    min(self.p_max_pixel[0], self.p_max_pixel[1]) / 2)
        # 车高于地面5cm的俯视图轮廓，用于对车轮进行粗定位
        wheel_shadow = cut_cloud(self.car, boundary={1: [None, 0.05 + self.p_min[1]]})
        floor = [[i[0], i[2]] for i in wheel_shadow]
        self.img_ver = pixel(floor, self.pixel_size, [self.p_min[0], self.p_min[2]], [self.p_max[0], self.p_max[2]],
                             darkest=1, extention=self.extention)
        if is_left:
            self.img_ver = cv2.flip(self.img_ver, 0)
        self.img_ver = cv2.dilate(self.img_ver, kernel_n(n=2), iterations=1)
        self.img_ver = cv2.erode(self.img_ver, kernel_n(n=3), iterations=1)
        # debug用的俯视图的opencv画布
        self.empty_img_ver_c = numpy.array(
            [numpy.array([[0, 0, 0] for j in range(len(self.img_ver[0]))], dtype=numpy.uint8)
             for i in range(len(self.img_ver))], dtype=numpy.uint8)
        # 车的侧视图
        self.img_side = pixel(car, self.pixel_size, self.p_min, self.p_max, darkest=1, extention=self.extention)
        self.img_side = cv2.dilate(self.img_side, kernel_n(n=2), iterations=1)
        self.img_side = cv2.erode(self.img_side, kernel_n(n=3), iterations=1)
        # plot_mosaic(self.img_side)
        # debug用的侧视图的opencv画布
        self.empty_img_side_c = numpy.array(
            [numpy.array([[0, 0, 0] for j in range(len(self.img_side[0]))], dtype=numpy.uint8)
             for i in range(len(self.img_side))], dtype=numpy.uint8)

        self._drop_wheel = []
        self._fitting_wheel = []
        self.fake_wheel_x = []
        self.prior_wheel_x = []
        self.waiting_list_wheel_x = []
        self._y_limit_continuous = []
        self._v = []
        self._img_side_contours = []
        self._img_ver_contours = []
        self._estimate_center_ver = []
        self._estimate_center_side = []
        self._estimate_center_match = []
        self._bds_range = []
        self._bds = []

        self._find_wheel()
        self._match_wheel()



    def get_wheel(self):
        if self.prior_wheel_x:
            return self.prior_wheel_x
        elif self.waiting_list_wheel_x:
            return self.waiting_list_wheel_x
        else:
            return self.fake_wheel_x

    def _match_wheel(self):
        def _pixel_to_real(l):
            for idx, i in enumerate(l):
                l[idx] = [j / self.extention * self.pixel_size for j in i]
                l[idx][0] = l[idx][0] + self.p_min[0]
                l[idx][1] = l[idx][1] + self.p_min[1]
        def _takeForth(elem):
            return elem[3]

        _pixel_to_real(self._drop_wheel)
        _pixel_to_real(self._fitting_wheel)

        used_set = []
        for dw in self._drop_wheel:
            for fw in self._fitting_wheel:
                if fw in used_set:
                    continue
                if abs(dw[0]-fw[0]) < 0.02:
                    self.prior_wheel_x.append((dw[0]+fw[0])/2)
                    used_set.append(fw)
                    used_set.append(dw)

        self._fitting_wheel.sort(key=_takeForth)

        for fw in self._fitting_wheel:
            if fw not in used_set:
                self.waiting_list_wheel_x.append(fw[0])
                break

        if not self.waiting_list_wheel_x:
            for dw in self._drop_wheel:
                if dw in used_set:
                    continue
                else:
                    self.waiting_list_wheel_x.append(dw[0])

        if not (self.prior_wheel_x or self.waiting_list_wheel_x or self.fake_wheel_x):
            self.fake_wheel_x.append((self.p_min[0]+self.p_max[0])/2)

        # print("prior",self.prior_wheel_x)
        # print("waitting list", self.waiting_list_wheel_x)
        # print("fake",self.fake_wheel_x)

    def _find_wheel(self):
        self._estimate_wheel_center_by_vertical_view()
        self._find_drop_wheel()
        if self._img_side_contours:
            self._find_fitting_wheel()

    def _estimate_wheel_center_by_vertical_view(self):
        """靠俯视图毛估估一个位置范围"""
        # 分离俯视图的块，并把所有找到的块存在self._img_ver_contours
        contours, hierarchy = cv2.findContours(self.img_ver, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        color_list = [(255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (0, 0, 255), (148, 0, 211)]
        color_list = [(i[2], i[1], i[0]) for i in color_list]
        count = 0
        for i in range(len(contours)):
            if cv2.contourArea(contours[i]) > 10:
                self._img_ver_contours.append(contours[i])
                cv2.drawContours(self.empty_img_ver_c, contours, i, color=color_list[count % 6], thickness=-1)
                count += 1
        # 如果self._img_ver_contours不为空，
        if self._img_ver_contours:
            # 对contour找外接矩形，方便找边界
            contour_rect = []
            for i in range(len(self._img_ver_contours)):
                x, y, w, h = cv2.boundingRect(self._img_ver_contours[i])
                contour_rect.append([i, x, y, w, h])

            # 找最低点最低的块lowest
            def take_y(elem):
                return elem[2]

            contour_rect.sort(key=take_y)
            lowest = contour_rect[0]
            # 从其他块里找最低点不高于lowest最高点的块，把这些块都存在self._estimate_center_ver
            self._estimate_center_ver.append(lowest)
            for i in contour_rect[1:]:
                x, y, w, h = i[1:]
                if y > lowest[2] + lowest[4]:
                    break
                else:
                    self._estimate_center_ver.append(i)
            # print(self._estimate_center_ver)
            # plot_opencv(self.empty_img_ver_c)

    def _find_drop_wheel(self):
        contours, hierarchy = cv2.findContours(self.img_side, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            (rect_x, rect_y), l, angle = cv2.minAreaRect(contours[i])
            if area > self.area_filter and 1 / self.proportion < l[0] / l[1] < self.proportion and \
                    rect_y - l[1] / 2 < self.p_max_pixel[1] / 2:
                self._img_side_contours.append(contours[i])
                cv2.drawContours(self.empty_img_side_c, contours, i, color=(255, 102, 102), thickness=-1)
                (x, y), radius = cv2.minEnclosingCircle(contours[i])
                if y - radius < 0 and rect_y - l[1] / 2 <= 0 and area < radius ** 2 * math.pi * 0.7:
                    y = radius = (y ** 2 + radius ** 2) / 2 / y
                if self.pixel_wheel_diam_rang[0] < radius < self.pixel_wheel_diam_rang[1] and \
                        self.pixel_wheel_diam_rang[0] < y < self.pixel_wheel_diam_rang[1] and \
                        self.pixel_wheel_diam_rang[0] < x < self.p_max_pixel[0] - self.pixel_wheel_diam_rang[0] and \
                        area > self.circle_area_filter:
                    cv2.drawContours(self.empty_img_side_c, contours, i, color=(153, 255, 255), thickness=-1)
                    self._drop_wheel.append([x, y, radius])
                    (x, y, radius) = numpy.int0((x, y, radius))
                    add_opencv_circle(self.empty_img_side_c, (x, y, radius), (255, 0, 0))

        for i in range(len(self._img_ver_contours)):
            cv2.drawContours(self.empty_img_side_c, self._img_ver_contours, i, color=(0, 0, 255), thickness=-1)
        # plot_opencv(self.empty_img_side_c)

    def _find_fitting_wheel(self):
        self._y_limit = get_lower_edge(len(self.img_side[0]), self._img_side_contours)
        self._estimate_wheel_center_by_side_view()
        self._fit_curve()

    def _estimate_wheel_center_by_side_view(self):
        lowest = []
        for i in range(len(self._y_limit)):
            if self._y_limit[i] is not None:
                p = [i, self._y_limit[i]]
                if lowest == [] or lowest[0][1] == round(p[1]):
                    lowest.append(p)
                elif lowest[0][1] > round(p[1]):
                    lowest.clear()
                    lowest.append(p)
        # pyplot.plot(self._y_limit, "sg-")
        # new_plot(lowest, "or")
        lowest = split_list(lowest)
        self._estimate_center_side = [(sum([j[0] for j in i]) / len(i)) for i in lowest]
        for i in self._estimate_center_ver:
            for j in self._estimate_center_side:
                if i[1] < j < i[1] + i[3]:
                    self._estimate_center_match.append(j)
        if not self._estimate_center_match:
            self._estimate_center_match = self._estimate_center_side
        # new_plot([[i, self._y_limit[round(i)]] for i in res], "*y")
        # for i in self._estimate_center_ver:
        #     new_plot([[i[1],i[2]],[i[1]+i[3],i[2]]],"b^-")
        # pyplot.axis("equal")
        # pyplot.show()

    def _fit_curve(self):
        """圆拟合"""
        """把离散的self._y_limit插值成连续的self._y_limit_continuous，并求每个像素距离的斜率self._v"""
        for i in range(len(self._y_limit)):
            if self._y_limit[i] is not None:
                self._y_limit_continuous.append([i, self._y_limit[i]])
        self._y_limit_continuous = interpolate_by_stepLen(self._y_limit_continuous, 5)
        for i in range(len(self._y_limit_continuous) - 1):
            p1 = self._y_limit_continuous[i]
            p2 = self._y_limit_continuous[i + 1]
            local_v = (p2[1] - p1[1]) / (p2[0] - p1[0])
            int_x = round((p2[0] + p1[0]) / 2)
            if self._v == [] or self._v[-1][0] != int_x:
                self._v.append([int_x, local_v])
            elif abs(self._v[-1][1]) < abs(local_v):
                self._v[-1][1] = local_v
        self._v = interpolate_by_pixel(self._v, False)
        v_head = [[i, self._v[0][1]] for i in range(self._v[0][0])]  # 微分后会差几位，补全
        self._v = v_head + self._v
        v_tail = [[i, self._v[-1][1]] for i in range(len(self._v),len(self._y_limit))]
        self._v = self._v + v_tail
        """尝试对self._estimate_center_match列表中每一个预测圆形的位置进行圆拟合"""
        for estimate_x in self._estimate_center_match:
            """找到左右两侧距离中心self.pixel_wheel_diam_rang[0] * 0.5 到 x0 - self.pixel_wheel_diam_rang[1] * 1.2 范围内，
                下缘线的斜率的最高点和最低点"""
            x0 = int(estimate_x)
            left = [self._v[x0], self._v[x0]]
            right = [self._v[x0], self._v[x0]]
            for i in range(int(x0 - self.pixel_wheel_diam_rang[0] * 0.5),
                           max(int(x0 - self.pixel_wheel_diam_rang[1] * 1.2), 0), -1):
                if left[0][1] <= self._v[i][1]:
                    left[0] = self._v[i]
                if left[1][1] >= self._v[i][1]:
                    left[1] = self._v[i]
            for i in range(int(x0 + self.pixel_wheel_diam_rang[0] * 0.5),
                           min(int(x0 + self.pixel_wheel_diam_rang[1] * 1.2), len(self._v))):
                if right[0][1] <= self._v[i][1]:
                    right[0] = self._v[i]
                if right[1][1] >= self._v[i][1]:
                    right[1] = self._v[i]
            """根据斜率的最值点找边界线，通常车的下缘线是两段水平的直线连接在一个下半圆的两侧，下半圆的左右端点处的轮廓线接近竖直，
            所以左端点处的斜率应该是非常大的负值，而右端点处的斜率应该是非常大的正值"""
            if left == [self._v[x0], self._v[x0]] or right == [self._v[x0], self._v[x0]]:
                fitting_points = None # 如果找不到斜率的直接让被拟合点为None，后续会报错
            else:
                self._bds_range.append([left, right])
                bds = [None, None]
                if int(x0 - self.pixel_wheel_diam_rang[1] * 1.2) <= 0 and abs(left[0][1]) < 10 and abs(left[1][1]) < 10:
                    # 如果搜索范围已经到达了边界，而且找到的最值都不太大，则认为这些最值是噪声点造成的，直接取边界
                    bds[0] = 0
                elif left[0][0] > left[1][0] and (left[1][1] == 0 or abs(left[0][1] / left[1][1]) > 1.5):
                    # 通常轮子和车壳间会有缝隙，导致下半圆的左端点处的斜率先升再降，
                    # 又因为噪声点会让数据不准确，用于做左端点判断标准的最低点会比较小，则用最高点的右边一个点
                    bds[0] = left[0][0] + 1
                else:
                    bds[0] = left[1][0] - 1
                if int(x0 + self.pixel_wheel_diam_rang[1] * 1.2) >= len(self._v) and abs(right[0][1]) < 10 and abs(
                        right[1][1]) < 10:
                    bds[1] = len(self._v) - 1
                elif right[0][0] > right[1][0] and (right[0][1] == 0 or abs(right[1][1] / right[0][1]) > 1.5):
                    bds[1] = right[1][0] - 1
                else:
                    bds[1] = right[0][0] + 1
                self._bds.append(bds)
                fitting_points = numpy.array([[i, self._y_limit[i]] for i in range(bds[0], bds[1] + 1)])
            """预测一个初值，求误差最小的圆方程"""
            para_estimate = numpy.array([round(estimate_x), self.pixel_wheel_diam_rang[1],
                                                 self.pixel_wheel_diam_rang[1]])
            try:
                a = minimize(lambda para_list: sum_error(point_list=fitting_points,
                                                         x=para_list[0], y=para_list[1], r=para_list[2]),
                             x0=para_estimate,
                             bounds=((self._bds[-1][0] + self.pixel_wheel_diam_rang[0] * 0.5,
                                      self._bds[-1][1] - self.pixel_wheel_diam_rang[0] * 0.5),
                                     (self.pixel_wheel_diam_rang[0], self.pixel_wheel_diam_rang[1]),
                                     (self.pixel_wheel_diam_rang[0], self.pixel_wheel_diam_rang[1]))
                             )

                x = [a.x[0], a.x[1], a.x[2], a.fun]
                color = (0,0,255)
                self._fitting_wheel.append(x)
                # print(x, a.fun)
                # self.plot_fitting_situation()
            except:
                print("fitting failure")
                r = (self.pixel_wheel_diam_rang[0] + self.pixel_wheel_diam_rang[1]) / 2
                x = [estimate_x, r, r]
                self.fake_wheel_x.append(estimate_x/ self.extention * self.pixel_size + self.p_min[0])
                color = (0, 255,0)
            add_opencv_circle(self.empty_img_side_c, numpy.int0(x), color)
        # plot_opencv(self.empty_img_side_c)

    def plot_fitting_situation(self):
        pyplot.plot(self._y_limit, "r-+")
        new_plot(self._v)
        for car_part in self._img_side_contours:
            car_part = [i[0] for i in car_part.tolist()]
            car_part.append(car_part[0])
            new_plot(car_part)
        for i in range(len(self._estimate_center_match)):
            new_plot(self._bds_range[i][0][0], "ro")
            new_plot(self._bds_range[i][1][0], "ro")
            new_plot(self._bds_range[i][0][1], "ko")
            new_plot(self._bds_range[i][1][1], "ko")
            new_plot(self._v[self._bds[i][0]], "b^")
            new_plot(self._v[self._bds[i][1]], "b^")
            new_plot([self._estimate_center_match[i], 0], "co")
            new_plot(self._v[int(self._estimate_center_match[i] + self.pixel_wheel_diam_rang[0] * 0.5)], "yo")
            new_plot(self._v[max(int(self._estimate_center_match[i] - self.pixel_wheel_diam_rang[0] * 0.5), 0)], "yo")
            new_plot(self._v[int(self._estimate_center_match[i] - self.pixel_wheel_diam_rang[1] * 1.2)], "yo")
            new_plot(self._v[min(int(self._estimate_center_match[i] + self.pixel_wheel_diam_rang[1] * 1.2), len(self._v) - 1)], "yo")
            new_plot(build_cicle([self._fitting_wheel[i][0], self._fitting_wheel[i][1]], self._fitting_wheel[i][2]))

        pyplot.axis("equal")
        pyplot.show()


