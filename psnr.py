import math
from skimage import io
import random
import numpy as np
import cv2
#输入原图
img1 =cv2.imread('./images/103.png')
cv2.imshow('img1',img1)
#SR图像
def gauss_noise(image):
    img = image.astype(np.int16)  # 此步是为了避免像素点小于0，大于255的情况
    mu = 0
    sigma = 10
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            for k in range(img.shape[2]):
                img[i, j, k] = img[i, j, k] + random.gauss(mu=mu, sigma=sigma)
    img[img > 255] = 255
    img[img < 0] = 0
    img = img.astype(np.uint8)
    return img

if __name__ == '__main__':
    img = cv2.imread("./images/452.jpg")
    # img2 = gauss_noise(img)
    img2 = cv2.imread("./images/104.png")
    cv2.imshow("img2",img2)
cv2.waitKey(0)
print('img1的图像shape:',img1.shape)
print('img2的图像shape:',img2.shape)
# 计算PSNR的值
def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
print(psnr(img1,img2))
