from matplotlib import pyplot
import numpy, math

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