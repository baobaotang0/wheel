from vtkmodules import all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from math_tools import xz_pixel, kernel_n, get_rotate_matrix
from plot_tools import *
from QuarterWheelFinder import QuarterWheelFinder

import time,numpy, cv2,math

def get_wheel_dis(front_list:list, behind_list:list):
    res = []
    for f in front_list:
        for b in behind_list:
            res.append([b-f, f, b])
    return res



class WholeWheelFinder:
    pixel_size = 0.02
    extention = 3
    frame_size = [10, 20]
    _p_min: list
    _p_max: list
    _p_mid: list
    _p_mid_piexl: list
    _p_max_piexl: list
    

    def __init__(self, car: list):

        self.car = car
        self._filtered_car = None
        self._car_contour = None

        self._img_car = []
        self._car_outline = []
        self.integrity = []
        self._straightened_outlin =[]


        self._statistical_filt()
        self._get_outline()
        self._exam_unworking_radar()
        if self.integrity == [0,1]:
            self._straighten_car()

            left_front_car, left_behind_car, right_front_car, right_behind_car = [], [], [], []
            for i in self._straightened_outline:
                if i[1] > self._p_mid_piexl[1]:
                    if i[0] < self._p_mid_piexl[0]:
                        left_front_car.append(i)
                    else:
                        left_behind_car.append(i)
                else:
                    if i[0] < self._p_mid_piexl[0]:
                        right_front_car.append(i)
                    else:
                        right_behind_car.append(i)

            self.left_front_car = QuarterWheelFinder(left_front_car,
                                                     p_min=[0, self._p_mid_piexl[1]],
                                                     p_max=[self._p_mid_piexl[0], self._p_max_piexl[1]],
                                                     is_left=True)
            self.left_behind_car = QuarterWheelFinder(left_behind_car,
                                                      p_min=self._p_mid_piexl,
                                                      p_max=self._p_max_piexl,
                                                      is_left=True)

            self.right_front_car = QuarterWheelFinder(right_front_car,
                                                      p_min=[0, 0],
                                                      p_max=self._p_mid_piexl,
                                                      is_left=False)
            self.right_behind_car = QuarterWheelFinder(right_behind_car,
                                                       p_min=[self._p_mid_piexl[0], 0],
                                                       p_max=[self._p_max_piexl[0],self._p_mid_piexl[1]],
                                                       is_left=False)



    def _statistical_filt(self):
        car = [p for p in self.car if p[1] < 0.95]  # 剪掉车的上半部分，省计算
        car_data_set = numpy_to_vtk(car)
        car_points = vtk.vtkPoints()
        car_points.SetData(car_data_set)
        car_poly = vtk.vtkPolyData()
        car_poly.SetPoints(car_points)

        removal_filter = vtk.vtkStatisticalOutlierRemoval()
        removal_filter.SetInputData(car_poly)
        removal_filter.SetStandardDeviationFactor(1.8)
        removal_filter.Update()
        out_poly = removal_filter.GetOutput()  # type:vtk.vtkPolyData
        points = out_poly.GetPoints()
        filtered_car = points.GetData()
        self._filtered_car = vtk_to_numpy(filtered_car).tolist()
        max_min = out_poly.GetBounds()
        self._p_min = [max_min[0], max_min[4]]
        self._p_max = [max_min[1], max_min[5]]
        self._p_mid = [(self._p_max[i] + self._p_min[i]) / 2 for i in range(2)]
        self._p_mid_piexl = [round((self._p_mid[i] - self._p_min[i]) / self.pixel_size * self.extention) for i in range(2)]
        self._p_max_piexl = [round((self._p_max[i] - self._p_min[i]) / self.pixel_size * self.extention) for i in range(2)]
        # vg_filter = vtk.vtkVertexGlyphFilter()
        # vg_filter.SetInputData(out_poly)
        #
        # vg_filter2 = vtk.vtkVertexGlyphFilter()
        # vg_filter2.SetInputData(car_poly)
        #
        # mapper = vtk.vtkPolyDataMapper()
        # mapper.SetInputConnection(vg_filter.GetOutputPort())
        #
        # mapper2 = vtk.vtkPolyDataMapper()
        # mapper2.SetInputConnection(vg_filter2.GetOutputPort())
        #
        # actor = vtk.vtkActor()
        # actor.SetMapper(mapper)
        # from vtktool import vtktool
        # actor2 = vtk.vtkActor()
        # actor2.SetMapper(mapper2)
        # vtktool.color_actor(actor2, (255, 0, 0))
        # vtktool.vtk_show(actor2,actor)

    def _get_outline(self):
        self._img_car = xz_pixel(self._filtered_car, self.pixel_size, self._p_min, self._p_max, self.extention)
        self._img_car = cv2.bitwise_not(self._img_car)
        dilate = cv2.dilate(self._img_car, kernel_n(5), iterations=1)
        self._img_car = cv2.erode(dilate, kernel_n(5), iterations=1)
        self.empty_img_c = numpy.array(
            [numpy.array([[0, 0, 0] for j in range(len(self._img_car[0]))], dtype=numpy.uint8)
             for i in range(len(self._img_car))], dtype=numpy.uint8)
        # plot_mosaic(self._img_car)
        contours, hierarchy = cv2.findContours(self._img_car, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
        for i in range(len(contours)):
            area = cv2.contourArea(contours[i])
            if self._car_contour is None or area > cv2.contourArea(self._car_contour):
                self._car_contour = contours[i]
        self._car_outline = [i[0] for i in self._car_contour.tolist()
                    if 30 < abs(i[0][1] - (self._p_mid_piexl[1])) < self._p_max_piexl[1] / 2 - self.frame_size[1] and \
                    abs(i[0][0] - (self._p_mid_piexl[0])) < self._p_max_piexl[0] / 2 - self.frame_size[0]]

    def _exam_unworking_radar(self):
        points_cnt = [0,0]
        for i in self._car_outline:
            if i[1]> self._p_mid_piexl[1]:
                points_cnt[1] +=1
            else:
                points_cnt[0] += 1
        print(points_cnt)
        # new_plot(self._car_outline)
        # pyplot.show()
        if points_cnt[0] == 0 or points_cnt[1]/ points_cnt[0] > 2:
            print("2号机没扫描")
            self.integrity = [0]
        elif points_cnt[1]/ points_cnt[0] < 1/2:
            print("1号机没扫描")
            self.integrity = [1]
        else:
            self.integrity = [0, 1]

    def _straighten_car(self):
        self._car_contour = numpy.array([[i] for i in self._car_outline])
        cv2.drawContours(self.empty_img_c, [self._car_contour], 0, (0, 0, 255), 2)
        ellipse = cv2.fitEllipse(self._car_contour)
        self._tilt_angle = 90 - ellipse[2] if abs(90 - ellipse[2]) < abs(ellipse[2]) else ellipse[2] # TODO:too short car
        print("tilt angle",self._tilt_angle)
        box = cv2.boxPoints(ellipse)
        box = numpy.int0(box)
        cv2.drawContours(self.empty_img_c, [box], 0, (0, 0, 255), 2)
        cv2.ellipse(self.empty_img_c, ellipse, (0, 255, 0), 2)

        rotate_matrix = get_rotate_matrix(self._tilt_angle)
        self._straightened_outline = [None for i in range(len(self._car_outline))]
        for i in range(len(self._car_outline)):
            self._straightened_outline[i] = numpy.dot(rotate_matrix, self._car_outline[i])
            




    def get_wheels(self):
        res = [[], []]
        for x in self.wheel_x[0]:
            res[0].append([x, 0.35+self._p_min[1], self._p_max[2]])
        for x in self.wheel_x[1]:
            res[1].append([x, 0.35+self._p_min[1], self._p_min[2]])
        return res

    def plot_wheels(self):
        from matplotlib import pyplot
        pyplot.figure(1)
        new_plot(self.left_front_car.car,".")
        new_plot(self.left_behind_car.car,".")
        new_plot([[i[0], 2*self._p_max[1]-i[1]] for i in self.right_front_car.car],".")
        new_plot([[i[0], 2*self._p_max[1]-i[1]] for i in self.right_behind_car.car],".")
        new_plot(build_cicle([self.wheel_x[0][0], 0.35+self._p_min[1]],0.35),"k")
        new_plot(build_cicle([self.wheel_x[0][1], 0.35 + self._p_min[1]], 0.35), "k")
        new_plot(build_cicle([self.wheel_x[1][0], 2*self._p_max[1]-(0.35 + self._p_min[1])], 0.35), "k")
        new_plot(build_cicle([self.wheel_x[1][1], 2*self._p_max[1]-(0.35 + self._p_min[1])], 0.35), "k")
        pyplot.axis("equal")
        pyplot.show()


