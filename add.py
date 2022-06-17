import cv2

a=cv2.imread("./images/yanshi/lighten.jpg")
b=cv2.imread("./images/yanshi/clahed.jpg")



p = 0.01
temp = 100
c = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
d = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)

for i in range(100):
    result2 = cv2.addWeighted(c, p , d, (1-p), 0)
    means, dev = cv2.meanStdDev(result2)
    print('means: {},  dev: {},  p: {}'.format(means, dev, p))
    if dev > temp:
        temp = dev
        p += 0.01
    else:
        break

print('means: {}, \n dev: {}, \n p: {}'.format(means, dev, p))
result = cv2.addWeighted(a, p, b, (1-p), 0)


# cv2.imshow("lighten",a)
# cv2.imshow("clahed",b)
cv2.imshow("result",cv2.addWeighted(a, 0.65, b, (0.35), 0))
cv2.imwrite("./images/result/add0.65result6.jpg", cv2.addWeighted(a, 0.65, b, (0.35), 0))  # 将图片保存
cv2.waitKey()
cv2.destroyAllWindows()
