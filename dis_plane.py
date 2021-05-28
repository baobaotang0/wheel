import math
import numpy


def rotate(axix, angle_Deg):
    angle_Rad = math.radians(angle_Deg)
    res = numpy.eye(4, 4)
    if axix in ["x", 0]:
        res[1][1] = math.cos(angle_Rad)
        res[2][1] = math.sin(angle_Rad)
        res[1][2] = -math.sin(angle_Rad)
        res[2][2] = math.cos(angle_Rad)
    elif axix in ["y", 1]:
        res[0][0] = math.cos(angle_Rad)
        res[2][0] = -math.sin(angle_Rad)
        res[0][2] = math.sin(angle_Rad)
        res[2][2] = math.cos(angle_Rad)
    elif axix in ["z", 2]:
        res[0][0] = math.cos(angle_Rad)
        res[1][0] = math.sin(angle_Rad)
        res[0][1] = -math.sin(angle_Rad)
        res[1][1] = math.cos(angle_Rad)

    return res


stick_matrix = rotate(0, 45)

stick_matrix[0][-1] = 1
stick_matrix[1][-1] = 1
stick_matrix[2][-1] = 4
# print(stick_matrix)
point = [-1, -1, -1]
center = [0, 0, 0]
vector = numpy.array([point[i] - center[i] for i in range(3)] + [0])
n = numpy.dot(stick_matrix, numpy.array([0, -1, 0, 0]))
distance = numpy.dot(vector, n) / math.sqrt(n[0] ** 2 + n[1] ** 2 + n[2] ** 2)
