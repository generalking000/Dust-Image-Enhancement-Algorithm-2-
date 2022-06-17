import cv2 as cv
import numpy as np

def pixel_operation(image_path: str):
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    cv.imshow('input', img)

    # 注意：python中的print函数默认换行，可以用end=''或者接任意字符
    # 像素均值、方差
    means, dev = cv.meanStdDev(img)
    print('means: {}, \n dev: {}'.format(means, dev))
    # 像素最大值和最小值
    # min_pixel = np.min(img[:, :, 0])
    # max_pixel = np.max(img[:, :, -1])
    # print('min: {}, max: {}'.format(min_pixel, max_pixel))

    # 若是一个空白图像
    # blank = np.zeros((300, 300, 3), dtype=np.uint8)
    # 像素均值、方差
    # blank[:, :] = (255, 0, 255)
    # means, dev = cv.meanStdDev(blank)
    # print('means: {}, \n dev: {}'.format(means, dev))

    cv.waitKey(0)
    cv.destroyAllWindows()


if __name__ == '__main__':
    path = './images/result/lighten.jpg'
    pixel_operation(path)
