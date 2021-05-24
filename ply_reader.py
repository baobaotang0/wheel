import os


def ply_loader(path):
    # print(path)
    data = []
    with open(path, 'r') as fp:
        for line in fp:
            if line == 'end_header\n':
                break
        for line in fp:
            p = line.split(' ')
            if len(p) != 3:
                break
            point = [float(i) for i in p]
            data.append(point)
    return data

def path_iter():
    ROOT_PATH = "platform_data/"
    for p in os.listdir(ROOT_PATH):
        # if p != "box_002.ply":
        #     continue
        if p.endswith('.ply'):
            print(p)
            yield os.path.join(ROOT_PATH, p)
