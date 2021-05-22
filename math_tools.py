import math

import numpy
from matplotlib import pyplot


def new_plot(l: list, style=None):
    if l is not []:
        if style:
            if isinstance(l[0], list):
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
        return point2 - point1


def is_positive(num: float, is_strict=True, loose_range=1e-8):
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


def calculate_dis(point1: list, point2: list) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def interpolate_by_pixel(outline, y_is_int=True):
    """只能给坐标为整数的点用"""
    res = []
    for i in range(len(outline) - 1):
        if outline[i][0] == outline[i + 1][0]:
            res.append(outline[i])
            continue
        else:
            vector = get_vector(outline[i], outline[i + 1])
            for x in range(outline[i][0], outline[i + 1][0], is_positive(outline[i + 1][0] - outline[i][0])):
                percent = (x - outline[i][0]) / (outline[i + 1][0] - outline[i][0])
                if y_is_int:
                    res.append([x, round(outline[i][1] + percent * vector[1])])
                else:
                    res.append([x, (outline[i][1] + percent * vector[1])])
    res.append(outline[-1])
    return res


def get_min_max(cloud):
    p_min = cloud[0].copy()
    p_max = cloud[0].copy()
    cnt = len(cloud[0])
    for p in cloud:
        for i in range(cnt):
            if p[i] < p_min[i]:
                p_min[i] = p[i]
            if p[i] > p_max[i]:
                p_max[i] = p[i]
    return p_min, p_max


def is_in_range(x: float, bounds: list):
    if bounds[0] is not None and bounds[1] is not None:
        flag = bounds[0] < x < bounds[1]
    elif bounds[1] is not None:
        flag = x < bounds[1]
    elif bounds[0] is not None:
        flag = bounds[0] < x
    else:
        flag = True
    return flag


def cut_cloud(cloud: list, boundary: dict, need_rest=False):
    """boundary = {0: [xmin,xmax], 1:[ymin,ymax], 2:[zmin,zmax]...}"""
    res = []
    rest = []
    cnt = len(cloud[0])
    if cnt < len(boundary.keys()):
        raise TypeError("")
    for p in cloud:
        is_in = 1
        for i, bds in boundary.items():
            is_in *= 1 if is_in_range(p[i], bds) else 0
        if is_in == 1:
            res.append([p[i] for i in range(cnt)])
        elif need_rest:
            rest.append([p[i] for i in range(cnt)])
    if need_rest:
        return res,rest
    else:
        return res


def pixel(cloud: list, pixel_size: float, p_min: list, p_max: list, darkest: float, extention=1, colored=False):
    resolution = [math.ceil((p_max[0] - p_min[0]) / pixel_size), math.ceil((p_max[1] - p_min[1]) / pixel_size)]
    res = [[0 for j in range(extention * (resolution[0] + 1))] for i in range(extention * (resolution[1] + 1))]
    for p in cloud:
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




def interpolate_by_stepLen(outline: list, dis_step: float):
    """需要提前检查重合点"""
    vector = []
    dis = [0]
    for i in range(1, len(outline)):
        vector.append(get_vector(outline[i - 1], outline[i]))
        dis.append(dis[i - 1] + calculate_dis(outline[i - 1], outline[i]))
    count = int(dis[-1] // dis_step)
    j = 0
    res = []
    for i in range(count + 1):
        dis_cur = dis_step * i
        while dis_cur - dis[j + 1] > 1e-6:
            j += 1
        percent = (dis_cur - dis[j]) / (dis[j + 1] - dis[j])
        res.append([outline[j][0] + percent * vector[j][0], outline[j][1] + percent * vector[j][1]])
    if calculate_dis(res[-1], outline[-1]) > dis_step / 2:
        res.append(outline[-1])
    else:
        res[-1] = outline[-1]
    return res
