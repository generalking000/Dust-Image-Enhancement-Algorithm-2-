import cv2
import numpy as np
import sys
from PIL import Image
import matplotlib.pyplot as plt

'''
def clahe(img):
    # 拆分通道
    B,G,R = cv2.split(img)
    # 创建CLAHE对象
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8, 8))
    # 将每个帧转换为灰度或将其应用于每个通道(转换为灰度)
    # gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe_B = clahe.apply(B)
    clahe_G = clahe.apply(G)
    clahe_R = clahe.apply(R)
    dst = cv2.merge([clahe_B, clahe_G, clahe_R])
    # 限制对比度的自适应顾值均衡化
    # temp = clahe.apply(gray_image)
    # 转回彩色图像
    # dst = cv2.cvtColor(temp, cv2.COLOR_GRAY2BGR)

    return dst
'''

def clahe(img):
    img = np.uint8(img)

    imgr = img[:, :, 0]
    imgg = img[:, :, 1]
    imgb = img[:, :, 2]

    claher = cv2.createCLAHE(clipLimit=3, tileGridSize=(10, 18))
    claheg = cv2.createCLAHE(clipLimit=2, tileGridSize=(10, 18))
    claheb = cv2.createCLAHE(clipLimit=1, tileGridSize=(10, 18))
    cllr = claher.apply(imgr)
    cllg = claheg.apply(imgg)
    cllb = claheb.apply(imgb)

    rgb_img = np.dstack((cllr, cllg, cllb))

    return rgb_img

if __name__ == '__main__':
    img = Image.open('./images/yanshi/dark.jpg').convert('RGB')
    '''
    img = cv2.imread("./images/01.jpg")
    cv2.imshow("原图", img)
    cv2.imshow("直方图均衡化", clahe(img))
    cv2.waitKey(0)
    cv2.destroyAllWindows
    '''

    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    plt.subplot(1, 2, 1), plt.imshow(img)
    plt.title('原图'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(clahe(img))
    plt.title('Clahe'), plt.axis('off')
    plt.show()
    cv2.imwrite("./images/yanshi/clahed.jpg", cv2.cvtColor(clahe(img), cv2.COLOR_RGB2BGR))  # 将图片保存