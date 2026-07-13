import cv2  # 读取的格式是BGR
import numpy as np

# # 1、图像腐蚀，函数为：
# # cv2.erode(src, kernel, dst, anchor, iterations, borderType, borderValue)
# # src:输入的图像
# # kernel:用于腐蚀的结构元件，他的形状和大小直接影响腐蚀的效果
# # dst:它是与src相同大小和类型的输出图像。
# # iterations:腐蚀操作的迭代次数，默认为1。次数越多，腐蚀操作执行的次数越多，腐蚀效果越明显

sun = cv2.imread('sun.png')
cv2.imshow('src', sun)
cv2.waitKey(0)
kernel = np.ones((3, 3), np.uint8)
erosion_1 = cv2.erode(sun, kernel, iterations=2)
cv2.imshow('erosion_1', erosion_1)
cv2.waitKey(0)

# # 2、图像膨胀，函数为：
# # cv2.dilate(img, kernel, iterations)
# # 参数含义：
# # img-目标图片
# # kernel- 进行操作的内核，默认为3x3的矩阵
# # iterations- 膨胀次数，默认为1
wenzi = cv2.imread('wenzi.png')
cv2.imshow('src1', wenzi)
cv2.waitKey(0)
kernel = np.ones((3, 3), np.uint8)
wenzi_new = cv2.dilate(wenzi, kernel, iterations=2)
cv2.imshow('wenzi_new', wenzi_new)
cv2.waitKey(0)

# # 5、顶帽和黑帽
# # 顶帽= 原始图像- 开运算结果(先腐蚀后膨胀)    用于提取比周围区域亮的细节。
# # 黑帽= 闭运算(先膨胀后腐蚀) - 原始图像        用于提取比周围区域暗的细节。
sun = cv2.imread('sun.png')
cv2.imshow('sun_yuantu', sun)
cv2.waitKey(0)
kernel = np.ones((2, 2), np.uint8)
# 开运算
open_sun = cv2.morphologyEx(sun, cv2.MORPH_OPEN, kernel)
cv2.imshow('open_sun', open_sun)
cv2.waitKey(0)
# 顶帽
tophat = cv2.morphologyEx(sun, cv2.MORPH_TOPHAT, kernel)
cv2.imshow('TOPHAT', tophat)
cv2.waitKey(0)
# 闭运算
close_sun = cv2.morphologyEx(sun, cv2.MORPH_CLOSE, kernel)
cv2.imshow('close_sun', close_sun)
cv2.waitKey(0)
# 黑帽
blackhat = cv2.morphologyEx(sun, cv2.MORPH_BLACKHAT, kernel)
cv2.imshow('BLACKHAT', blackhat)
cv2.waitKey(0)

# ''' 边缘检测 '''
# # sobel算子
# # cv2.Sobel(src, ddepth, dx, dy[, ksize[, scale[, delta[, borderType]]]])
# #参数:
# # src:输入图像
# # ddepth:输出图像的深度(可以理解为数据类型), -1表示与原图像相同的深度
# # dx,dy:当组合为dx=1,dy=0时x方向的一阶导数，当组合为dx=0,dy=1时求y方向的一阶导数(如果同时为，通常效果不佳
# # ksize: (可选参数)Sobel算子的大小，必须是1,3,5或者7，默认为3。
yuan = cv2.imread('yuan.png')
cv2.imshow('yuan', yuan)
cv2.waitKey(0)

# # x方向上的边缘
yuan_x = cv2.Sobel(yuan, -1, dx=1, dy=0)
cv2.imshow('yuan_x', yuan_x)
cv2.waitKey(0)
# x方向上的边缘，包括负数信息，但显示不出来，范围(0~255)
# 二值黑白对比极强，左右梯度绝对值大小接近，视觉叠加成单条圆弧，
yuan_x_64 = cv2.Sobel(yuan, cv2.CV_64F, dx=1, dy=0)
cv2.imshow('yuan_x_64', yuan_x_64)
cv2.waitKey(0)
# x方向上的边缘，包括负数信息，进行取绝对值的操作，右端的负值信息就可以显示出来了
yuan_x_full = cv2.convertScaleAbs(yuan_x_64)
cv2.imshow('yuan_x_full', yuan_x_full)
cv2.waitKey(0)

# y方向上的边缘
yuan_y = cv2.Sobel(yuan, -1, dx=0, dy=1)
cv2.imshow('yuan_y', yuan_y)
cv2.waitKey(0)
# y方向上的边缘，包括负数信息，但显示不出来，因为范围是(0~255)
yuan_y_64 = cv2.Sobel(yuan, cv2.CV_64F, dx=0, dy=1)
yuan_y_full = cv2.convertScaleAbs(yuan_y_64)
cv2.imshow('yuan_y_full', yuan_y_full)
cv2.waitKey(0)

# 如果同时使用x，y方向的结果如何呢?(不建议使用!!)只能检测 45° 斜线条，横竖轮廓全部丢失，边缘残缺；
yuan_xy = cv2.Sobel(yuan, -1, dx=1, dy=1)
cv2.imshow('yuan_xy', yuan_xy)
cv2.waitKey(0)

# 使用图像加权运算组合x和y方向的2个边缘。
yuan_xy_full = cv2.addWeighted(yuan_x_full, 1, yuan_y_full, 1, 0)
cv2.imshow('yuan_xy_full', yuan_xy_full)
cv2.waitKey(0)


import cv2

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. 转灰度图 + Canny边缘检测
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    canny_img = cv2.Canny(gray, 40, 160)

    # 2. Sobel边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_all = cv2.addWeighted(sobel_x, 0.5, sobel_y, 0.5, 0)

    # Sobel反向二值化：阈值40，低于40=255，高于40=0
    ret, sobel_bin_inv = cv2.threshold(sobel_all, thresh=40, maxval=255, type=cv2.THRESH_BINARY_INV)

    # 3. 三个窗口同时显示
    cv2.imshow("Canny边缘", canny_img)
    cv2.imshow("Sobel原图", sobel_all)
    cv2.imshow("Sobel反向二值化", sobel_bin_inv)

    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2

# 1. 图片灰度、二值化、轮廓提取
hua = cv2.imread('hua.png')
hua_gray = cv2.cvtColor(hua, cv2.COLOR_BGR2GRAY)
cv2.imshow('hua_b', hua_gray)
cv2.waitKey(0)

# 反向二值化
ret, hua_binary = cv2.threshold(hua_gray, thresh=240, maxval=255, cv2.THRESH_BINARY_INV)
cv2.imshow('hua_binary', hua_binary)
cv2.waitKey(0)

# 查找轮廓
_, contours, hierarchy = cv2.findContours(hua_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
print(f"轮廓总数：{len(contours)}")

# 绘制最大面积轮廓
image_copy = hua.copy()
# 按面积从大到小排序
sortcnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
cv2.drawContours(image_copy, [sortcnt], contourIdx=-1, color=(0, 255, 0), thickness=3)
cv2.imshow('image_contours', image_copy)
cv2.waitKey(0)

# 轮廓近似多边形
epsilon = 0.005 * cv2.arcLength(sortcnt, closed=True)
approx = cv2.approxPolyDP(sortcnt, epsilon, closed=True)
print("原轮廓点数", sortcnt.shape)
print("近似后点数", approx.shape)

# 绘制近似轮廓
cv2.drawContours(image_copy, [approx], contourIdx=-1, color=(0, 255, 0), thickness=3)
cv2.imshow('image_contours', image_copy)
cv2.waitKey(0)

# 轮廓基础属性：面积、周长、外接矩形、外接圆
# 面积
area = cv2.contourArea(sortcnt)
print("轮廓面积：", area)
# 周长
length = cv2.arcLength(sortcnt, closed=True)
print("轮廓周长：", length)
# 外接矩形
x, y, w, h = cv2.boundingRect(sortcnt)
cv2.rectangle(image_copy, (x, y), (x+w, y+h), (0, 0, 255), 2)
# 外接圆
(x_c, y_c), r = cv2.minEnclosingCircle(sortcnt)
cv2.circle(image_copy, (int(x_c), int(y_c)), int(r), (255, 0, 0), 2)

cv2.destroyAllWindows()

import cv2

zl = cv2.imread('img1.jpeg', cv2.IMREAD_GRAYSCALE)
cv2.imshow('zl', zl)
cv2.waitKey(0)

# 1. Scharr算子
zl_x_64 = cv2.Scharr(zl, cv2.CV_64F, dx=1, dy=0)
zl_x_full = cv2.convertScaleAbs(zl_x_64)
zl_y_64 = cv2.Scharr(zl, cv2.CV_64F, dx=0, dy=1)
zl_y_full = cv2.convertScaleAbs(zl_y_64)
zl_xy_Scharr_full = cv2.addWeighted(zl_x_full, 1, zl_y_full, 1, 0)
cv2.imshow('zl_xy_Scharr_full', zl_xy_Scharr_full)
cv2.waitKey(0)

# 2. Laplacian拉普拉斯算子
zl_lap = cv2.Laplacian(zl, cv2.CV_64F, ksize=3)
zl_lap_full = cv2.convertScaleAbs(zl_lap)
cv2.imshow('zl_lap_full', zl_lap_full)
cv2.waitKey(0)

# 3. Canny边缘检测
zl_canny = cv2.Canny(zl, 100, 150)
cv2.imshow('zl_canny', zl_canny)
cv2.waitKey(0)

cv2.destroyAllWindows()

import cv2

kele = cv2.imread('kele.png')
template = cv2.imread('template.png')
cv2.imshow('kele', kele)
cv2.imshow('template', template)
cv2.waitKey(0)

h, w = template.shape[:2]
# 模板匹配
res = cv2.matchTemplate(kele, template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
top_left = max_loc
bottom_right = (top_left[0] + w, top_left[1] + h)
# 绘制匹配框
cv2.rectangle(kele, top_left, bottom_right, (0, 255, 0), 2)
cv2.imshow('kele_template', kele)
cv2.waitKey(0)
cv2.destroyAllWindows()

import argparse

# 创建参数解析器
parser = argparse.ArgumentParser()
# 添加各类参数
parser.add_argument("--SERIAL_PORT1", type=str, default='COM5', help='第一个报警器的串口号')
parser.add_argument("--area_thred", type=int, default=1500, help='物体面积的阈值')
parser.add_argument("--confid_level", type=float, default=0.8, help='识别的置信度')
parser.add_argument("--aaa", type=int, default=100)
parser.add_argument('-b', "--bbb", type=int, default=10)

# 解析参数
opt = parser.parse_args()
a = opt.aaa
b = opt.bbb
print(a + b)

import cv2

def sort_contours(cnts, method="left-to-right"):
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][0]))
    return cnts, boundingBoxes

import numpy as np
import argparse
import cv2
import myutils

# 命令行参数
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
ap.add_argument("-t", "--template", required=True, help="path to template OCR-A image")
args = vars(ap.parse_args())

# 银行卡类型映射
FIRST_NUMBER = {
    "3": "American Express",
    "4": "Visa",
    "5": "MasterCard",
    "6": "Discover Card"
}

# 绘图封装函数
def cv_show(name, img):
    cv2.imshow(name, img)
    cv2.waitKey(0)

# ----------------------模板数字预处理----------------------
img = cv2.imread(args["template"])
cv_show('img', img)
ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv_show('ref', ref)
# 反向二值化 黑底白字
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
cv_show('ref', ref)

# 模板轮廓提取
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cv2.drawContours(img, refCnts, -1, (0, 255, 0), 3)
cv_show('refCnts', img)
# 轮廓从左到右排序
refCnts, _ = myutils.sort_contours(refCnts, method="left-to-right")
digits = {}

# 保存0-9数字模板
for (i, c) in enumerate(refCnts):
    x, y, w, h = cv2.boundingRect(c)
    roi = ref[y:y + h, x:x + w]
    roi = cv2.resize(roi, (57, 88))
    digits[i] = roi

# ----------------------信用卡原图处理----------------------
image = cv2.imread(args["image"])
cv_show('image', image)
image = myutils.resize(image, width=300)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv_show('gray', gray)

# 形态学顶帽运算
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
cv_show('tophat', tophat)

# 闭运算连接数字
closeX = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, rectKernel)
cv_show('closeX', closeX)
# 自适应二值化
thresh = cv2.threshold(closeX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
cv_show('thresh', thresh)
# 再次闭运算降噪
thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
cv_show('close2', thresh)

# 查找数字区域轮廓
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnts_img = image.copy()
cv2.drawContours(cnts_img, cnts, -1, (0, 0, 255), 3)
cv_show('cnts_img', cnts_img)

# 筛选符合比例的数字块
locs = []
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    ar = w / float(h)
    if 2.5 < ar < 4.0:
        if (40 < w < 55) and (10 < h < 20):
            locs.append((x, y, w, h))
# 从左到右排序
locs = sorted(locs, key=lambda x: x[0])

# ----------------------模板匹配识别数字----------------------
output = []
for (gX, gY, gW, gH) in locs:
    group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
    cv_show('group', group)
    group = cv2.threshold(group, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('group', group)
    digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    digitCnts, _ = myutils.sort_contours(digitCnts, method="left-to-right")
    groupOutput = []
    for c in digitCnts:
        x, y, w, h = cv2.boundingRect(c)
        roi = group[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        cv_show('roi', roi)
        # 模板匹配打分
        scores = []
        for (digit, digitROI) in digits.items():
            result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
            _, score, _, _ = cv2.minMaxLoc(result)
            scores.append(score)
        # 取最高分数字
        groupOutput.append(str(np.argmax(scores)))
    # 绘制框+文字
    cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
    cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    output.extend(groupOutput)

# 输出识别结果
print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
print("Credit Card #: {}".format("".join(output)))
cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()