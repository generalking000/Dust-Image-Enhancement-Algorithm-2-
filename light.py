import cv2
import numpy as np
import matplotlib.pyplot as plt


def light(img):

    lighten = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    for x in range(lighten.shape[0]):
        for y in range(lighten.shape[1]):
            l, a, b = lighten[x, y, :]
            # fix for CV correction
            lighten[x, y, 0] = l * 1.1
            lighten[x, y, 1] = a
            lighten[x, y, 2] = b

    lighten = cv2.cvtColor(lighten, cv2.COLOR_LAB2BGR)

    return lighten


if __name__ == '__main__':
    img = cv2.imread("./images/yanshi/dark.jpg")
    orgImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    light(img)
    m = cv2.cvtColor(light(img), cv2.COLOR_BGR2RGB)
    cv2.imwrite("./images/yanshi/lighten.jpg", light(img))  # 将图片保存
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    plt.subplot(121), plt.title("原图"), plt.axis('off')
    plt.imshow(orgImg)  # matplotlib 显示彩色图像(RGB格式)
    plt.subplot(122), plt.title("Lab亮度补偿图"), plt.axis('off')
    plt.imshow(m)
    plt.show()