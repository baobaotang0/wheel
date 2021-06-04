import math
import numpy
from matplotlib import pyplot
from scipy.optimize import minimize
from scipy.signal import savgol_filter
from math_tools import get_min_max,  interpolate_by_pixel, interpolate_by_stepLen,  calculate_dis, \
sum_error
from plot_tools import *
import time

class QuarterWheelFinder:
    pixel_size = 0.02
    extention = 3
    wheel_diam_range = [0.55 / 2, 0.90 / 2]
    tire_thickness = 0.05
    pixel_tire_thickness =tire_thickness/ pixel_size * extention
    print("tt",pixel_tire_thickness)
    pixel_wheel_diam_rang = [int(wheel_diam_range[0] / pixel_size * extention),
                             math.ceil(wheel_diam_range[1] / pixel_size * extention)]
    area_filter = 1000  # 不能小于2个像素点，这样边界会只有3个点而无法进行圆拟合
    circle_area_filter = pixel_wheel_diam_rang[0] ** 2 * math.pi / 2


    def __init__(self, car: list, p_min, p_max, is_left: bool):
        self._car = car
        self._p_min, self._p_max, = p_min, p_max

        self._is_left = is_left
        if is_left:
            self._car = [[i[0],-i[1]] for i in self._car]
            self._p_min[1], self._p_max[1], = -p_max[1], -p_min[1]

        print("max,min", p_min, p_max)

        self._edge = [None for j in range(self._p_max[0]-self._p_min[0]+1)]
        for p in self._car:
            x = round(p[0]) - p_min[0]
            if self._edge[x] is None or p[1] < self._edge[x]:
                self._edge[x] = p[1]

        self._edge_continuous = [[i, self._edge[i]] for i in range(len(self._edge)) if self._edge[i] is not None]
        self._edge_continuous = interpolate_by_pixel(self._edge_continuous, False)


        savgol = savgol_filter([i[1] for i in self._edge_continuous], int(len(self._edge_continuous)/20)*2 +5, 3)
        # TODO: >5 judge
        self._edge_continuous =[[self._edge_continuous[i][0], savgol[i]]for i in range(len(self._edge_continuous))]


        lowest = None
        for i in self._edge:
            if i is not None:
                if lowest is None or i < lowest:
                    lowest = i

        low_points = []
        for i in self._edge_continuous:
            if i[1] < lowest + self.pixel_tire_thickness:
                low_points.append(i)

        self._estimate_center = [sum([i[0] for i in low_points])/len(low_points), lowest]



        self._v = [[self._edge_continuous[i][0],
                    +10*(self._edge_continuous[i+1][1]-self._edge_continuous[i][1])/ \
                    (self._edge_continuous[i+1][0]-self._edge_continuous[i][0])]
                   for i in range(len(self._edge_continuous)-1)]



        para_estimate = numpy.array([0.1, -0.1*(low_points[0][0]+low_points[-1][0]), lowest])
        a = minimize(lambda para_list: sum_error(point_list=low_points,
                                                 a=para_list[0], b=para_list[1], c=para_list[2]),
                     x0=para_estimate)

        print(-a.x[1]/a.x[0]/2)
        # for i, txt in enumerate(numpy.arange(len(self._car))):
        #     pyplot.annotate(txt, self._car[i])

        pyplot.plot(self._edge,"*--")
        new_plot(self._edge_continuous)
        new_plot([[i[0],a.x[0]*i[0]**2+a.x[1]*i[0]+a.x[2]]for i in low_points],"^-")
        new_plot([[i[0],i[1]+ lowest]for i in self._v],"g")
        new_plot([low_points[0],low_points[-1]],"o")
        new_plot(self._estimate_center,"r*")
        new_plot([-a.x[1]/a.x[0]/2, lowest],"bs")

        pyplot.axis("equal")
        pyplot.show()




        self._estimate_center = []



    def _match_wheel(self):
        def _pixel_to_real(l):
            for idx, i in enumerate(l):
                l[idx] = [j / self.extention * self.pixel_size for j in i]
                l[idx][0] = l[idx][0] + self._p_min[0]
                l[idx][1] = l[idx][1] + self._p_min[1]



        _pixel_to_real(self._fitting_wheel)





    def _estimate_wheel_center_by_side_view(self):
        lowest = []
        for i in range(len(self._edge)):
            if self._edge[i][1] < self:
                lowest.clear()
                lowest.append(self._edge[i])
        # pyplot.plot(self._edge, "sg-")
        # new_plot(lowest, "or")
        lowest = split_list(lowest)
        self._estimate_center_side = [(sum([j[0] for j in i]) / len(i)) for i in lowest]

        # new_plot([[i, self._edge[round(i)]] for i in res], "*y")
        # for i in self._estimate_center_ver:
        #     new_plot([[i[1],i[2]],[i[1]+i[3],i[2]]],"b^-")
        # pyplot.axis("equal")
        # pyplot.show()



