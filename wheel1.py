import os,numpy,vtktool,math
from matplotlib import pyplot
from math_tools import new_plot
import cv2
def get_min_max_3d(car):
    p_min = car[0].copy()
    p_max = car[0].copy()
    for p in car:
        for i in range(3):
            if p[i] < p_min[i]:
                p_min[i] = p[i]
            if p[i] > p_max[i]:
                p_max[i] = p[i]
    return p_min, p_max

def cut_2dcar(car:list,idx:int,limit:list):
    res = []
    for p in car:
        if limit[0] <= p[idx] <= limit[1] :
            res.append([p[0],p[1]])
    return res

def pixel(car:list, pixel_size, p_min:list, p_max:list, darkest:float,extention=1, colored=False):
    resolution = [math.ceil((p_max[0]-p_min[0])/pixel_size), math.ceil((p_max[1]-p_min[1])/pixel_size)]
    res = [[0 for j in range(extention*(resolution[0]+1))] for i in range(extention*(resolution[1]+1))]
    for p in car:
        i = int((p[0] - p_min[0]) / pixel_size)
        j = int((p[1]-p_min[1])/pixel_size)
        for k in range(extention):
            for l in range(extention):
                res[extention*j+k][extention*i+l] += 1
    for i in range(resolution[1]):
        for j in range(resolution[0]):
            if res[extention * i][extention * j] > darkest:
                for k in range(extention):
                    for l in range(extention):
                        res[extention * i + k][extention * j + l] = 255
            else:
                for k in range(extention):
                    for l in range(extention):
                        res[extention * i + k][extention * j + l] = int(255/darkest*res[extention * i][extention * j])
    if colored:
        for i in range(len(res)):
            for j in range(len(res[0])):
                res[i][j] = [res[i][j]]*3
    res = numpy.array(res, dtype=numpy.uint8)
    return res


if __name__ == '__main__':

    folder_path = "cars/"
    car_id = os.listdir(folder_path)
    for i in car_id:
        # if i not in ["car_12.npy"]:
        #     continue
        if i.endswith("npy"):
            print(i)
            path2d = "cars/" + i
            with open(path2d, 'rb') as f_pos:
                car = list(numpy.load(f_pos))
                p_min, p_max = get_min_max_3d(car)
                mid = (p_min[2]+p_max[2])/2
                half_car = cut_2dcar(car,idx=2,limit=[mid,p_max[2]])
                p_max[1] = 0.95
                p_min[1] = 0
                half_car = cut_2dcar(half_car,idx=1,limit=[p_min[1],p_max[1]])
                # pyplot.figure(figsize=(20,10))
                # new_plot(half_car,".")
                # pyplot.axis("equal")
                # pyplot.show()
                # vtktool.vtk_show(car)
                pixel_size = 0.015
                # pyplot.figure(figsize=(20,5))
                mosaic_matrix = pixel(half_car,pixel_size,p_min, p_max, darkest=1, extention=2)
                # c=pyplot.pcolormesh(mosaic_matrix, cmap ='magma')
                # pyplot.colorbar(c)
                # pyplot.axis("equal")
                # pyplot.show()

                mosaic_matrix2 = pixel(half_car, pixel_size, p_min, p_max, darkest=1, extention=2,colored=True)
                img = mosaic_matrix
                empyt_img_line = numpy.array([numpy.array([[0] for j in range(len(mosaic_matrix[0]))], dtype=numpy.uint8)
                                            for i in range(len(mosaic_matrix))], dtype=numpy.uint8)
                empyt_img_bw = numpy.array([numpy.array([[0] for j in range(len(mosaic_matrix[0]))], dtype = numpy.uint8)
                       for i in range(len(mosaic_matrix))], dtype = numpy.uint8)
                empyt_img_c = numpy.array([numpy.array([[0,0,0] for j in range(len(mosaic_matrix[0]))], dtype=numpy.uint8)
                                            for i in range(len(mosaic_matrix))], dtype=numpy.uint8)
                kernel_2 = numpy.ones((2, 2), dtype=numpy.uint8)
                kernel_3 = numpy.ones((3, 3), dtype=numpy.uint8)
                kernel_4 = numpy.ones((4, 4), dtype=numpy.uint8)# 卷积核变为4*
                kernel_5 = numpy.ones((5, 5), dtype=numpy.uint8)
                dilate = cv2.dilate(img, kernel_2, iterations=1)
                erosion = cv2.erode(dilate, kernel_3, iterations=1)
                ss = numpy.hstack((img, erosion))
                # cv2.imshow('cleaner', ss)
                # cv2.waitKey(0)
                # image = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                contours, hierarchy = cv2.findContours(erosion, cv2.RETR_EXTERNAL,
                                                       cv2.CHAIN_APPROX_TC89_L1)  # find contours with simple approximation
                hierarchy = numpy.squeeze(hierarchy)

                print(len(contours))
                count = 0
                drop_wheel = []
                for i in range(len(contours)):
                    # 存在父轮廓，说明是里层

                    if cv2.contourArea(contours[i]) >1000:
                        count += 1
                        hull = cv2.convexHull(contours[i])
                        cv2.polylines(img, [hull, ], True, (0, 0, 255), 2)  # red
                        shadow_line = cv2.drawContours(empyt_img_bw, contours, i, color=255, thickness=2)
                        shadow_bw = cv2.drawContours(empyt_img_bw, contours, i, color=255, thickness=-1)
                        shadow_c = cv2.drawContours(empyt_img_c, contours, i, color=255, thickness=-1)
                        if cv2.contourArea(contours[i]) <10000:
                            (x, y), radius = cv2.minEnclosingCircle(contours[i])
                            drop_wheel.append([x, y, radius])
                            (x, y, radius) = numpy.int0((x, y, radius))
                            cv2.circle(empyt_img_c, (x, y), radius, (0, 0, 255), 2)
                            cv2.circle(empyt_img_c, (x, y), 2, (0, 0, 255), 3)

                    # if cv2.contourArea(contours[i]) >10000:
                    #
                    #     approx = cv2.approxPolyDP(contours[i], 12, True)
                    #     # print(approx)
                    #     # 3.画出多边形
                    #     cv2.polylines(mosaic_matrix2, [approx], True, (0, 255, 0), 3)
                    #
                    #     cv2.imshow('cleaner', mosaic_matrix2)
                    #     cv2.waitKey(0)
                print(count)
                if len(drop_wheel)>2:
                    print("*******")

                # cv2.drawContours(image, contours, 0, (0, 255, 75), 1)
                # image = cv2.cvtColor(image, closing)
                # cv2.imshow('cleaner', img)  # Figure 3
                # cv2.waitKey(0)



                # img = empyt_img
                # # print(int((p_max[0]-p_min[0])*6),int((p_max[0]-p_min[0])*8.5))
                #
                # img = cv2.medianBlur(img, 3)
                # cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
                hough_wheel = cv2.HoughCircles(empyt_img_bw,cv2.HOUGH_GRADIENT,dp=1,minDist = 300,
                                            param1=50,param2=5,minRadius=30,maxRadius=60)
                print(hough_wheel)


                if hough_wheel is not None:
                    hough_wheel = numpy.uint16(numpy.around(hough_wheel))
                    for i in hough_wheel[0,:]:
                        # draw the outer circle
                        cv2.circle(empyt_img_c,(i[0],i[1]),i[2],(0,255,0),2)
                        # draw the center of the circle
                        cv2.circle(empyt_img_c,(i[0],i[1]),2,(0,255,0),3)
                cv2.imshow('detected hough_wheel',empyt_img_c)
                cv2.waitKey(0)
                # cv2.destroyAllWindows()


