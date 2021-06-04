from import_car import car_importer_iter
from find_wheel import WholeWheelFinder
from ply_reader import *
import time
if __name__ == '__main__':
    # for car in car_importer_iter():
    for path in path_iter("platform_data/"):
        # if path != "box_000.ply":
        #     continue
        floor_shadow = ply_loader(os.path.join("platform_data/", path))
        # car = ply_loader(os.path.join("car_data/", path))
        time1 = time.time()
        a = WholeWheelFinder(floor_shadow)
        print(time.time()-time1)
