from vtkmodules import all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from math_tools import get_min_max,cut_cloud
from QuarterWheelFinder import QuarterWheelFinder


class WholeWheelFinder:
    def __init__(self, car: list):
        self.car = car.copy()
        car = cut_cloud(car, boundary={1: [None, 0.95]})  # 剪掉车的上半部分，省计算

        car_data_set = numpy_to_vtk(car)
        car_points = vtk.vtkPoints()
        car_points.SetData(car_data_set)
        car_poly = vtk.vtkPolyData()
        car_poly.SetPoints(car_points)

        removal_filter = vtk.vtkStatisticalOutlierRemoval() # TODO: adjust para
        removal_filter.SetInputData(car_poly)
        removal_filter.Update()
        out_poly = removal_filter.GetOutput()  # type:vtk.vtkPolyData
        points = out_poly.GetPoints()
        filtered_car = points.GetData()
        filtered_car = vtk_to_numpy(filtered_car).tolist()


        self.p_min, self.p_max = get_min_max(filtered_car)
        left_car, right_car = cut_cloud(filtered_car,
                                        boundary={2: [(self.p_min[2] + self.p_max[2]) / 2, None]},
                                        need_rest=True)
        if len(left_car) / len(right_car) > 1 / 2:
            print("1号机扫描了")
            left_p_min, left_p_max = get_min_max(left_car)
            left_front_car, left_behind_car = cut_cloud(left_car,
                                                        boundary={0: [None, (left_p_min[0] + left_p_max[0]) / 2]},
                                                        need_rest=True)  # TODO: input max min
            self.left_front_car = QuarterWheelFinder(left_front_car, True)
            self.left_behind_car = QuarterWheelFinder(left_behind_car, True)
        if len(left_car) / len(right_car) < 2:
            print("2号机扫描了")
            right_p_min, right_p_max = get_min_max(right_car)
            right_front_car, right_behind_car = cut_cloud(right_car,
                                                          boundary={0: [None, (right_p_min[0] + right_p_max[0]) / 2]},
                                                          need_rest=True)
            self.right_front_car = QuarterWheelFinder(right_front_car,False)
            self.right_behind_car = QuarterWheelFinder(right_behind_car,False)




        # from vtktool import vtktool
        # vtktool.vtk_show(car)


