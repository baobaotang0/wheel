from import_car import car_importer_iter
from find_wheel import WholeWheelFinder

if __name__ == '__main__':
    for car in car_importer_iter():
        a = WholeWheelFinder(car)
        # print(a.get_wheel())
        # a.left_car.plot_opencv()
        # a.right_car.plot_opencv()