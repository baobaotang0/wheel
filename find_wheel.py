import cv2
import math
import numpy
from vtkmodules import all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy

from QuarterWheelFinder import QuarterWheelFinder, WheelConst
from math_tools import xz_pixel, kernel_n, get_rotate_matrix
from plot_tools import *


def get_wheel_dis(front_list: list, behind_list: list):
    res = []
    for f in front_list:
        for b in behind_list:
            res.append([b - f, f, b])
    return res


class WholeWheelFinder:
    attr=WheelConst()
    frame_size = [20, 35]
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
        self._straightened_outline = []
        self.wheels =[]

        self._statistical_filt()
        self._get_outline()
        self._straighten_and_separate_car()
        self._match_wheels()
        self.plot_wheels()




    def _statistical_filt(self):
        car_data_set = numpy_to_vtk(self.car )
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
        self._p_min = [max_min[0], max_min[4], max_min[2]]
        self._p_max = [max_min[1], max_min[5], max_min[3]]
        self._p_mid = [(self._p_max[i] + self._p_min[i]) / 2 for i in range(2)]
        self._p_mid_piexl = [round((self._p_mid[i] - self._p_min[i]) / self.attr.pixel_size * self.attr.extention) for i in
                             range(2)]
        self._p_max_piexl = [round((self._p_max[i] - self._p_min[i]) / self.attr.pixel_size * self.attr.extention) for i in
                             range(2)]
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
        self._img_car = xz_pixel(self._filtered_car, self.attr.pixel_size, self._p_min, self._p_max, self.attr.extention)
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
                             if 30 < abs(i[0][1] - (self._p_mid_piexl[1])) < self._p_max_piexl[1] / 2 - self.frame_size[
                                 1] and \
                             abs(i[0][0] - (self._p_mid_piexl[0])) < self._p_max_piexl[0] / 2 - self.frame_size[0]]


    def _straighten_and_separate_car(self):
        self._car_contour = numpy.array([[i] for i in self._car_outline])
        cv2.drawContours(self.empty_img_c, [self._car_contour], 0, (0, 0, 255), 2)
        ellipse = cv2.fitEllipse(self._car_contour)
        self._tilt_angle = 90 - ellipse[2] if abs(90 - ellipse[2]) < abs(ellipse[2]) else ellipse[2]
        # TODO:too short car
        print("tilt angle", self._tilt_angle)
        box = cv2.boxPoints(ellipse)
        box = numpy.int0(box)
        cv2.drawContours(self.empty_img_c, [box], 0, (0, 0, 255), 2)
        cv2.ellipse(self.empty_img_c, ellipse, (0, 255, 0), 2)

        left_front_car, left_behind_car, right_front_car, right_behind_car = [], [], [], []
        for i in self._car_outline:
            if i[1] > self._p_mid_piexl[1]:
                if i[0] < self._p_mid_piexl[0]:
                    left_front_car.append([i[0],i[1]+self._line_function(i[0])])
                else:
                    left_behind_car.append([i[0],i[1]+self._line_function(i[0])])
            else:
                if i[0] < self._p_mid_piexl[0]:
                    right_front_car.append([i[0],i[1]+self._line_function(i[0])])
                else:
                    right_behind_car.append([i[0],i[1]+self._line_function(i[0])])

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
                                                   p_max=[self._p_max_piexl[0], self._p_mid_piexl[1]],
                                                   is_left=False)

    def _match_wheels(self):
        new_plot([[i[0], i[2]] for i in self.car], ".")
        for i in [self.left_front_car,self.left_behind_car,self.right_front_car,self.right_behind_car]:
            local_wheel = [i.pixel_wheel_center[0]/ self.attr.extention * self.attr.pixel_size + self._p_min[0],
                               i.pixel_wheel_center[1]/ self.attr.extention * self.attr.pixel_size + self._p_min[1]]

            new_plot(local_wheel,"s")
            local_wheel = [local_wheel[0],local_wheel[1] - self._line_function(local_wheel[0])]

            self.wheels.append(local_wheel)

    def get_wheels(self):
        res = [[], []]
        res[0].append([self.wheels[0][0], 0.35 + self._p_min[2], self.wheels[0][1]])
        res[0].append([self.wheels[1][0], 0.35 + self._p_min[2], self.wheels[1][1]])
        res[1].append([self.wheels[2][0], 0.35 + self._p_min[2], self.wheels[2][1]])
        res[1].append([self.wheels[3][0], 0.35 + self._p_min[2], self.wheels[3][1]])
        return res

    def _line_function(self,x):
        if self._tilt_angle != 90:
            tilt_k = math.tan(math.radians(self._tilt_angle))
            return tilt_k*(x-self._p_min[0])
        else:
            return 0

    def plot_wheels(self):
        from matplotlib import pyplot
        # pyplot.figure(1)

        new_plot(self.wheels, "*")
        # for i, txt in enumerate(numpy.arange(len(self.wheels_x))):
        #     new_plot([[self.wheels_x[i],self._p_min[1]],[self.wheels_x[i],self._p_max[1]]], "-")
        #     pyplot.annotate(txt, [self.wheels_x[i],self._p_max[1]])



        # new_plot(build_cicle([self.wheel_x[0][0], 0.35 + self._p_min[1]], 0.35), "k")
        # new_plot(build_cicle([self.wheel_x[0][1], 0.35 + self._p_min[1]], 0.35), "k")
        # new_plot(build_cicle([self.wheel_x[1][0], 2 * self._p_max[1] - (0.35 + self._p_min[1])], 0.35), "k")
        # new_plot(build_cicle([self.wheel_x[1][1], 2 * self._p_max[1] - (0.35 + self._p_min[1])], 0.35), "k")
        pyplot.axis("equal")
        pyplot.show()
