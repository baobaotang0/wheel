from matplotlib import pyplot

def new_plot(l: list, style=None):
    if l is not []:
        if style:
            if isinstance(l[0],list):
                pyplot.plot([p[0] for p in l], [p[1] for p in l], style)
            else:
                pyplot.plot(l[0], l[1], style)
        else:
            pyplot.plot([p[0] for p in l], [p[1] for p in l])