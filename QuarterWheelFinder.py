from scipy.signal import savgol_filter
from math_tools import interpolate_by_pixel
from plot_tools import *

class WheelConst:
    pixel_tire_thickness = 7.5
    pixel_size = 0.02
    extention = 3
    wheel_diam_range = (0.55 / 2, 0.90 / 2)
    pixel_wheel_diam_rang = (int(wheel_diam_range[0] / pixel_size * extention),
                             math.ceil(wheel_diam_range[1] / pixel_size * extention))


class QuarterWheelFinder:
    attr=WheelConst()
    pixel_wheel_center:list

    def __init__(self, car: list, p_min, p_max, is_left: bool):
        self._car = car

        self.p_min, self.p_max, = p_min, p_max
        self._is_left = is_left

        if len(self._car)>0:
            self._mirror()
            self._get_edge()
            self._get_local_center()
            self._get_wheel_center()
            self.plot_situation()
        else:
            self.pixel_wheel_center = [(self.p_min[0]+self.p_max[0])/2, (self.p_min[1]+self.p_max[1])/2]

    def _mirror(self):
        if self._is_left:
            self._car = [[i[0],-i[1]] for i in self._car]
            self.p_min[1], self.p_max[1], = -self.p_max[1], -self.p_min[1]

    def _get_edge(self):
        self._edge = [None for j in range(self.p_max[0]-self.p_min[0]+1)]
        for p in self._car:
            x = round(p[0]) - self.p_min[0]
            if self._edge[x] is None or p[1] < self._edge[x]:
                self._edge[x] = p[1]
        self._edge_continuous = [[i, self._edge[i]] for i in range(len(self._edge)) if self._edge[i] is not None]

        self._edge_continuous = interpolate_by_pixel(self._edge_continuous, False)
        if len(self._edge_continuous) > 5:
            savgol = savgol_filter([i[1] for i in self._edge_continuous], int(len(self._edge_continuous)/20)*2 +5, 3)
            self._edge_continuous =[[self._edge_continuous[i][0], savgol[i]]for i in range(len(self._edge_continuous))]
        
    def _get_local_center(self):
        self._lowest = self._edge_continuous[0] + [0]
        for i in range(1,len(self._edge_continuous)):
            if self._edge_continuous[i][1] < self._lowest[1]:
                self._lowest = self._edge_continuous[i] + [i]

        self._low_points = []
        searching_radius = 40
        start = max(self._lowest[2] - searching_radius, 0)
        end = min(self._lowest[2] + searching_radius, len(self._edge_continuous)-1)
        height = min(self._edge_continuous[start][1],
                     self._edge_continuous[end][1])
        for i in self._edge_continuous[start:end+1]:
            if i[1] < height:
                self._low_points.append(i)
        if self._low_points:
            self._local_center = [(self._low_points[0][0]+self._low_points[-1][0])/2,self._lowest[1]]
        else:
            self._local_center = [(self.p_max[0] - self.p_min[0]) / 2, self._lowest[1]]
            self._low_points.append(self._local_center)

    def _get_wheel_center(self):
        if self._is_left:
            self.pixel_wheel_center = [self._local_center[0] + self.p_min[0], - self._local_center[1]]
        else:
            self.pixel_wheel_center =[self._local_center[0] + self.p_min[0], self._local_center[1]]

    def plot_situation(self):
        pyplot.subplot(2,1,1)
        pyplot.plot(self._edge,"*--")
        new_plot(self._edge_continuous)
        new_plot([self._low_points[0],self._low_points[-1]],"o")
        new_plot(self._local_center,"r*")
        pyplot.axis("equal")
        pyplot.subplot(2, 1, 2)
        new_plot(self._car)
        new_plot([self._local_center[0] + self.p_min[0], self._lowest[1]],"*")
        pyplot.axis("equal")
        pyplot.show()












