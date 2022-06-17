import cv2
import numpy as np
import AWB


def gamma(orgimg , m):
    disimg = np.power(orgimg/float(np.max(orgimg)), m)
    return disimg

if __name__ == '__main__':
    orgImg = cv2.imread("./images/yanshi/Labbai.jpg")
    AWB.whiteBalance(orgImg)
    cv2.imshow('src', AWB.whiteBalance(orgImg))
    m = 1.5
    img = gamma(AWB.whiteBalance(orgImg), m)
    cv2.imshow('gamma='+'%.2f'%m, img)
    cv2.imwrite("./images/yanshi/gamma1.5.jpg", img*255)  # 将图片保存
    cv2.waitKey(0)
