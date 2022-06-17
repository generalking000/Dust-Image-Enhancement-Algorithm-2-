import numpy as np
import cv2
from skimage import io

# 导入图片
img = cv2.imread('./images/25.jpg', cv2.IMREAD_COLOR)

# 显示原图
cv2.imshow('GrayWorld_before', img)

# 定义BGR通道
r = img[:, :, 2]
b = img[:, :, 0]
g = img[:, :, 1]

# 计算BGR均值
averB = np.mean(b)
averG = np.mean(g)
averR = np.mean(r)

# 计算灰度均值
grayValue = (averR + averB + averG) / 3

# 计算增益
kb = grayValue / averB
kg = grayValue / averG
kr = grayValue / averR

# 补偿通道增益
r[:] = r[:] * kr
g[:] = g[:] * kg
b[:] = b[:] * kb

# 显示图像
cv2.imshow('GrayWorld_after', img)
cv2.imwrite("./images/result/RGBbai.jpg", img)  # 将图片保存

# 响应键盘事件
while True:
    k = cv2.waitKey(1) & 0xFF
    # 如果输入ESC，就结束显示。
    if k == 27:
        break

# 清理窗口
cv2.destroyAllWindows()