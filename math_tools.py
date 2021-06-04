import math
import numpy


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

def is_clockwise(xyz_list: list):
    length = len(xyz_list)
    d = 0
    for i in range(length - 1):
        d += -0.5 * (xyz_list[i + 1][1] + xyz_list[i][1]) * (xyz_list[i + 1][0] - xyz_list[i][0])
    if d < 0:
        clockwise = True
    else:
        clockwise = False
    return clockwise




def calculate_dis(point1: list, point2: list) -> float:
    return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)


def interpolate_by_pixel(outline, y_is_int=True):
    """只能给x坐标为整数的点用"""
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


def pixel(cloud: list, pixel_size: float, p_min: list, p_max: list, darkest: float, ignore_num = 0, extention=1, colored=False):
    resolution = [math.ceil((p_max[0] - p_min[0]) / pixel_size), math.ceil((p_max[1] - p_min[1]) / pixel_size)]
    res = [[0 for j in range(extention * (resolution[0] + 1))] for i in range(extention * (resolution[1] + 1))]
    cnt = [[0 for j in range((resolution[0] + 1))] for i in range((resolution[1] + 1))]
    def one_unit(j,i,value):
        for k in range(extention):
            for l in range(extention):
                res[extention * j + k][extention * i + l] = value
    for p in cloud:
        i = int((p[0] - p_min[0]) / pixel_size)
        j = int((p[1] - p_min[1]) / pixel_size)
        if i<= resolution[0] and j <= resolution[1] :
            cnt[j][i] += 1
    for i in range(resolution[1]):
        for j in range(resolution[0]):
            if cnt[i][j] > darkest:
                one_unit(i,j, 255)
            elif cnt[i][j] <= ignore_num:
                one_unit(i,j, 0)
            else:
                one_unit(i,j,int(255 / darkest * res[extention * i][extention * j]))
    if colored:
        for i in range(len(res)):
            for j in range(len(res[0])):
                res[i][j] = [res[i][j]] * 3
    res = numpy.array(res, dtype=numpy.uint8)
    return res


def xz_pixel(cloud: list, pixel_size: float, p_min: list, p_max: list, extention=1):
    resolution = [math.ceil((p_max[0] - p_min[0]) / pixel_size), math.ceil((p_max[1] - p_min[1]) / pixel_size)]
    res = [[0 for j in range(extention * (resolution[0] + 1))] for i in range(extention * (resolution[1] + 1))]
    cnt = [[0 for j in range((resolution[0] + 1))] for i in range((resolution[1] + 1))]
    def one_unit(j,i,value):
        for k in range(extention):
            for l in range(extention):
                res[extention * j + k][extention * i + l] = value
    for p in cloud:
        i = int((p[0] - p_min[0]) / pixel_size)
        j = int((p[2] - p_min[1]) / pixel_size)
        if i <= resolution[0] and j <= resolution[1]:
            cnt[j][i] += 1
    for i in range(resolution[1]):
        for j in range(resolution[0]):
            if cnt[i][j] >= 1:
                one_unit(i, j, 255)
            else:
                one_unit(i, j, 0)
    res = numpy.array(res, dtype=numpy.uint8)
    return res


def kernel_n(n):
    return numpy.ones((n, n), dtype=numpy.uint8)

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


def sum_error(point_list, a: float, b: float, c: float):
    res = 0
    for p in point_list:
        one_err = (p[1] - (a*p[0]** 2 +b*p[0] + c)) ** 2
        res += one_err
    return res




def get_rotate_matrix(angle):
    angle = math.radians(angle)
    return numpy.array([[math.cos(angle), -math.sin(angle)],[math.sin(angle),math.cos(angle),]])