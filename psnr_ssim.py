from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import numpy as np
import math
import cv2


def calc_ssim(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    ssim_score : numpy.float64
        结构相似性指数（structural similarity index，SSIM）.

    References
    -------
    https://scikit-image.org/docs/dev/auto_examples/transform/plot_ssim.html

    '''
    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处因为转换为灰度值之后的图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    ssim_score = ssim(img1, img2, data_range=255)
    return ssim_score


def calc_psnr(img1_path, img2_path):
    '''
    Parameters
    ----------
    img1_path : str
        图像1的路径.
    img2_path : str
        图像2的路径.

    Returns
    -------
    psnr_score : numpy.float64
        峰值信噪比(Peak Signal to Noise Ratio, PSNR).

    References
    -------
    https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    '''
    img1 = Image.open(img1_path)
    img2 = Image.open(img2_path)
    img2 = img2.resize(img1.size)
    img1, img2 = np.array(img1), np.array(img2)
    # 此处的第一张图片为真实图像，第二张图片为测试图片
    # 此处因为图像范围是0-255，所以data_range为255，如果转化为浮点数，且是0-1的范围，则data_range应为1
    psnr_score = psnr(img1, img2, data_range=255)
    return psnr_score

def psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2 )
    if mse == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

img1 =cv2.imread('./images/65.jpg')
img2 =cv2.imread('./images/result/none4.jpg')
print('SSIM', calc_ssim('./images/65.jpg', './images/result/none4.jpg'), 'PSNR',psnr(img1, img2))

'''
img1 = np.array(Image.open('./images/452.jpg')).astype(np.float64)
img2 = np.array(Image.open('./images/result/lighten.jpg')).astype(np.float64)


def psnr(origimal_img, filtered_img):
    
    计算图像的PSNR（Peak Signal to Noise Ratio）

    参数：
    -------
    origimal_img: 2-D array
                 图像的二维数组数据
    filtered_img: 2-D array
                图像的二维数组数据

    返回值：float，单位是dB
    ------
    返回psnr值
    
    if origimal_img.shape != filtered_img.shape:
        raise ValueError("the shapes of img1 and img2 are not same.")
    img1 = (origimal_img + 255) % 255  # 转到0-255
    img2 = (filtered_img + 255) % 255  # 转到0-255
    m = img1.shape[0]
    n = img1.shape[1]
    mse = np.sum((img1 - img2) ** 2) / (m * n)  # 计算MSE
    return 20 * np.log10(np.max(img1) / np.sqrt(mse))  # 计算PSNR


if __name__ == "__main__":
    print(psnr(img1, img2))
'''