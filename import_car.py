import os,numpy

def car_importer_iter(folder_path = "cars/"):
    car_id = os.listdir(folder_path)
    for i in car_id:
        if i not in ["car_04.npy"]:
            continue
        if i.endswith("npy"):
            print(i)
            path2d = "cars/" + i
            with open(path2d, 'rb') as f_pos:
                yield list(numpy.load(f_pos))
