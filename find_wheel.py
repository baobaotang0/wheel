from vtkmodules import all as vtk
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from math_tools import get_min_max,cut_cloud, new_plot, build_cicle
from QuarterWheelFinder import QuarterWheelFinder



def get_wheel_dis(front_list:list, behind_list:list):
    res = []
    for f in front_list:
        for b in behind_list:
            res.append([b-f, f, b])
    return res



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
        removal_filter.SetStandardDeviationFactor(1.8)

        removal_filter.Update()
        out_poly = removal_filter.GetOutput()  # type:vtk.vtkPolyData

        points = out_poly.GetPoints()
        filtered_car = points.GetData()
        filtered_car = vtk_to_numpy(filtered_car).tolist()


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
        #
        # vtktool.vtk_show(actor2, actor)


        self.p_min, self.p_max = get_min_max(filtered_car)
        self.left_car, self.right_car = cut_cloud(filtered_car,
                                        boundary={2: [(self.p_min[2] + self.p_max[2]) / 2, None]},
                                        need_rest=True)

        left_p_min, left_p_max = get_min_max(self.left_car)
        left_front_car, left_behind_car = cut_cloud(self.left_car,
                                                    boundary={0: [None, (left_p_min[0] + left_p_max[0]) / 2]},
                                                    need_rest=True)
        self.left_front_car = QuarterWheelFinder(left_front_car, True)
        self.left_behind_car = QuarterWheelFinder(left_behind_car, True)
        self.left_wheel_dis = get_wheel_dis(self.left_front_car.get_wheel(), self.left_behind_car.get_wheel())



        right_p_min, right_p_max = get_min_max(self.right_car)
        right_front_car, right_behind_car = cut_cloud(self.right_car,
                                                      boundary={0: [None, (right_p_min[0] + right_p_max[0]) / 2]},
                                                      need_rest=True)
        self.right_front_car = QuarterWheelFinder(right_front_car,False)
        self.right_behind_car = QuarterWheelFinder(right_behind_car,False)
        self.right_wheel_dis = get_wheel_dis(self.right_front_car.get_wheel(), self.right_behind_car.get_wheel())

        wheel_dis_difference = None
        self.wheel_x = None
        for l in self.left_wheel_dis:
            for r in self.right_wheel_dis:
                if abs(l[1]-r[1]) < 0.2:
                    if wheel_dis_difference is None or wheel_dis_difference < abs(l[0]-r[0]):
                        self.wheel_x = [l[1:], r[1:]]

        if self.wheel_x is None:
            print("!!!!!!!!!!!!!no match")
            self.wheel_x = [[self.left_front_car.get_wheel()[0], self.left_behind_car.get_wheel()[0]],
                              [self.right_front_car.get_wheel()[0], self.right_behind_car.get_wheel()[0]]]
        print(self.wheel_x)
        # self.plot_wheels()

    def get_wheels(self):
        res = [[], []]
        for x in self.wheel_x[0]:
            res[0].append([x, 0.35+self.p_min[1], self.p_max[2]])
        for x in self.wheel_x[1]:
            res[1].append([x, 0.35+self.p_min[1], self.p_min[2]])
        return res

    def plot_wheels(self):
        from matplotlib import pyplot
        pyplot.figure(1)
        new_plot(self.left_car,".")
        new_plot([[i[0], 2*self.p_max[1]-i[1]] for i in self.right_car],".")
        new_plot(build_cicle([self.wheel_x[0][0], 0.35+self.p_min[1]],0.35),"k")
        new_plot(build_cicle([self.wheel_x[0][1], 0.35 + self.p_min[1]], 0.35), "k")
        new_plot(build_cicle([self.wheel_x[1][0], 2*self.p_max[1]-(0.35 + self.p_min[1])], 0.35), "k")
        new_plot(build_cicle([self.wheel_x[1][1], 2*self.p_max[1]-(0.35 + self.p_min[1])], 0.35), "k")
        pyplot.axis("equal")
        pyplot.show()


