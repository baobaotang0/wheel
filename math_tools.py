from matplotlib import pyplot
import numpy, math
from typing import List

def new_plot(l: list, style=None):
    if l is not []:
        if style:
            if isinstance(l[0],list):
                pyplot.plot([p[0] for p in l], [p[1] for p in l], style)
            else:
                pyplot.plot(l[0], l[1], style)
        else:
            pyplot.plot([p[0] for p in l], [p[1] for p in l])

def build_cicle(center: list, radius: float, lineNum=12):
    theta = numpy.linspace(0, math.pi * 2, lineNum)
    res = []
    for i in theta:
        res.append([center[0] + math.cos(i) * radius, center[1] + math.sin(i) * radius])
    return res

def get_vector(point1, point2):
    if isinstance(point1, list):
        return [point2[i] - point1[i] for i in range(len(point1))]
    else:
        return point2-point1

def is_positive(num: float,is_strict=True,loose_range=1e-8):
    if is_strict:
        if num > 0:
            return 1
        elif num == 0:
            return 0
        else:
            return -1
    else:
        if num > loose_range:
            return 1
        elif num < -loose_range:
            return -1
        else:
            return 0

def interpolate_by_x(outline):
    """只能给坐标为整数的点用"""
    res = []
    for i in range(len(outline)-1):
        if outline[i][0] == outline[i+1][0]:
            res.append(outline[i])
            continue
        else:
            vector = get_vector(outline[i], outline[i+1])
            for x in range(outline[i][0],outline[i+1][0],is_positive(outline[i+1][0]-outline[i][0])):
                percent = (x-outline[i][0])/(outline[i+1][0]-outline[i][0])
                res.append([x, round(outline[i][1] + percent * vector[1])])
    res.append(outline[-1])
    return res

