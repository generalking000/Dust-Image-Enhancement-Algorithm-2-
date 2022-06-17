import cv2
import numpy as np
import matplotlib.pyplot as plt



def whiteBalance(img):

    # rows = img.shape[0]
    # cols = img.shape[1]

    final = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    avg_a = np.average(final[:, :, 1])
    avg_b = np.average(final[:, :, 2])

    for x in range(final.shape[0]):
        for y in range(final.shape[1]):
            l, a, b = final[x, y, :]
            # fix for CV correction
            l *= 100 / 255.0
            final[x, y, 1] = a - ((avg_a - 128) * 1.1)
            final[x, y, 2] = b - ((avg_b - 128) * 1.1)
            #final[x, y, 1] = a - ((avg_a - 128) * (1 / 100.0) * 1.1)
            #final[x, y, 2] = b - ((avg_b - 128) * (1 / 100.0) * 1.1)

    final = cv2.cvtColor(final, cv2.COLOR_LAB2BGR)

    return final

if __name__ == '__main__':
    img = cv2.imread("./images/yanshi/shate.jpg")
    orgImg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    whiteBalance(img)
    m=cv2.cvtColor(whiteBalance(img), cv2.COLOR_BGR2RGB)
    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    plt.subplot(121), plt.title("原图"), plt.axis('off')
    plt.imshow(orgImg)  # matplotlib 显示彩色图像(RGB格式)
    plt.subplot(122), plt.title("Lab白平衡图"), plt.axis('off')
    plt.imshow(m)
    plt.show()
    cv2.imwrite("./images/yanshi/Labbai.jpg", whiteBalance(img))  # 将图片保存



'''
def GW(orgImg):
    B, G, R = np.double(orgImg[:, :, 0]), np.double(orgImg[:, :, 1]), np.double(orgImg[:, :, 2])
    B_ave, G_ave, R_ave = np.mean(B), np.mean(G), np.mean(R)
    K = (B_ave + G_ave + R_ave) / 3
    Kb, Kg, Kr = K / B_ave, K / G_ave, K / R_ave
    Ba = (B * Kb)
    Ga = (G * Kg)
    Ra = (R * Kr)
    print(np.mean(Ba), np.mean(Ga), np.mean(Ra))
    dst_img = np.uint8(np.zeros_like(orgImg))
    dst_img[:, :, 0] = Ba
    dst_img[:, :, 1] = Ga
    dst_img[:, :, 2] = Ra
    dst_img = cv2.cvtColor(dst_img, cv2.COLOR_BGR2RGB)
    return dst_img
    #cv2.imshow("dstimg", dst_img)
    #cv2.waitKey(0)

if __name__ == '__main__':
    orgImg = cv2.imread("beiyingsha.png")
    #cv2.imshow("orgImg", orgImg)
    #cv2.waitKey(0)
    GW(orgImg)
    orgImg = cv2.cvtColor(orgImg, cv2.COLOR_BGR2RGB)

    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    plt.subplot(121), plt.title("原图"), plt.axis('off')
    plt.imshow(orgImg)  # matplotlib 显示彩色图像(RGB格式)
    plt.subplot(122), plt.title("白平衡图"), plt.axis('off')
    plt.imshow(GW(orgImg))
    plt.show()
'''