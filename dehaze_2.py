import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import AWB
from gamma import gamma


def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b);
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz);
    imvec = im.reshape(imsz,3);

    indices = darkvec.argsort();
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx;
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.8;
    im3 = np.empty(im.shape,im.dtype);

    for ind in range(0,3):
        im3[:,:,ind] = im[:,:,ind]/A[0,ind]

    transmission = 1 - omega*DarkChannel(im3,sz);
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
    cov_Ip = mean_Ip - mean_I*mean_p;

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
    var_I   = mean_II - mean_I*mean_I;

    a = cov_Ip/(var_I + eps);
    b = mean_p - a*mean_I;

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));

    q = mean_a*im + mean_b;
    return q;

def TransmissionRefine(im,et):
    im = im.astype(np.uint8)
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
    gray = np.float64(gray)/255;
    r = 60;
    eps = 0.0001;
    t = Guidedfilter(gray,et,r,eps);

    return t;

def Recover(im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype);
    t = cv2.max(t,tx);

    for ind in range(0,3):
        res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]

    return res

if __name__ == '__main__':
    import sys
    try:
        fn = sys.argv[1]
    except:
        # fn = './images/65.jpg'
        fn = './images/yanshi/gamma1.5.jpg'


    src = cv2.imread(fn);
    # src = AWB.whiteBalance(src)
    # src = gamma(AWB.whiteBalance(src), 1.5)*255

    I = src.astype('float64')/255;
 
    dark = DarkChannel(I,15);
    A = AtmLight(I,dark);
    te = TransmissionEstimate(I,A,15);
    t = TransmissionRefine(src,te);
    J = Recover(I,t,A,0.1);
    cv2.imwrite("./images/yanshi/dark.jpg", J * 255);
    #cv2.imwrite("./images/result/none4.jpg", J * 255);

    fig = plt.figure()

    plt.rcParams['font.sans-serif'] = ['FangSong']  # 支持中文标签
    ax1 = fig.add_subplot(211)
    plt.imshow(I[:, :, ::-1])
    plt.title('Dust Image')
    plt.xticks([])
    plt.yticks([])

    ax3 = fig.add_subplot(212)
    plt.imshow(J[:, :, ::-1])
    plt.title('Defogging Image($\omega$=0.95)')
    plt.xticks([])
    plt.yticks([])
    plt.show()
'''
    ax5 = fig.add_subplot(223)
    plt.imshow(dark[:, ::1])
    plt.title('Dark Channel')
    plt.xticks([])
    plt.yticks([])

    ax6 = fig.add_subplot(224)
    plt.imshow(t[:, ::1])
    plt.title('Coarse Transmittance Diagram')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    '''

'''
    cv2.imshow("dark",dark);
    cv2.imshow("t",t);
    cv2.imshow('I',src/255);
    cv2.imshow('J',J);
    cv2.imwrite("./images/result/dark_1.5gamma.jpg",J*255);
    cv2.waitKey();
    '''
