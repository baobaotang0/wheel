from import_car import car_importer_iter
from find_wheel import WholeWheelFinder
from ply_reader import *

if __name__ == '__main__':
    for car in car_importer_iter():
    # for path in path_iter("car_data/"):
    #     # if path != "car_data/box_001.ply":
    #     #     continue
    #     car = ply_loader(path)

        a = WholeWheelFinder(car)
