import numpy as np
import cv2
from xml.dom import minidom
import os.path
import numpy as np

cap = cv2.VideoCapture('4.mp4')
retval, frame = cap.read()
beforeFrame = frame
retval, frame = cap.read()
nowFrame = frame
kernel = np.ones((3, 3), np.uint8)
kernel2 = np.ones((5, 5), np.uint8)
i = 2
precisionList = []
if not os.path.exists("./mine"):
    os.mkdir("./mine")
if os.path.exists("./mine/IOU.txt"):
    os.remove("./mine/IOU.txt")
if os.path.exists("./mine/precision.txt"):
    os.remove("./mine/precision.txt")


def isCrossed(a, b):
    """
    :param a:
    :param b:
    :return:
    """
    if a[0] + a[2] < b[0] or a[1] + a[3] < b[1] or b[0] + b[2] < a[0] or b[1] + b[3] < a[1]:
        return False
    else:
        return True


def getDiff(frame1, frame2):
    """
    :param frame1:
    :param frame2:
    :return: binDiff
    """
    diffFrame = cv2.absdiff(frame1, frame2)
    grayDiff = cv2.cvtColor(diffFrame, cv2.COLOR_BGR2GRAY)
    retval, binDiff = cv2.threshold(grayDiff, 0, 255, cv2.THRESH_OTSU)
    binDiff = cv2.erode(binDiff, kernel, iterations=1)
    binDiff = cv2.dilate(binDiff, kernel2, iterations=5)
    binDiff = cv2.morphologyEx(binDiff, cv2.MORPH_CLOSE, kernel, iterations=9)
    # cv2.imshow("video", frame1)
    # cv2.imshow("gray",grayDiff)
    # cv2.waitKey(500)
    # cv2.imshow("bin",binDiff)
    # cv2.waitKey(500)
    return binDiff


def output_IOU(tech, mine):
    """
    :param tech:
    :param mine:
    :return:
    """
    resultList = []
    tempList = []
    # print(tech)
    # print(mine)
    # print(f"******比较帧{i}******")
    """
    遍历进行矩形的IOU计算
    """
    for item1 in mine:
        # print(item1)
        for item2 in tech:
            # print(item2)
            if not isCrossed(item1, item2):
                continue
            intersection_w = item1[2] + item2[2] - (
                    max(item1[0] + item1[2], item2[0] + item2[2]) - min(item1[0], item2[0]))
            intersection_h = item1[3] + item2[3] - (
                    max(item1[1] + item1[3], item2[1] + item2[3]) - min(item1[1], item2[1]))
            intersection = intersection_h * intersection_w
            union = item1[2] * item1[3] + item2[2] * item2[3] - intersection
            IOU = round(intersection / union, 2)
            # print(IOU)
            tempList.append(IOU)
        if not tempList:
            continue
        resultList.append(max(tempList))
        precisionList.append(max(tempList))
        tempList = []
    """
    IOU结果写入txt文件
    """
    with open("./mine/IOU.txt", mode="a") as f:
        f.write(f"帧{i}：")
        for item in resultList:
            f.write(f"{item}\t")
        f.write("\n")
        f.close


def getPrecison():
    """
    :return:
    """
    positive = 0
    negative = 0
    for item in precisionList:
        if item >= 0.5:
            positive += 1
        else:
            negative += 1
    precision = positive / (positive + negative)
    with open("./mine/precision.txt", mode="w") as f:
        f.write(f"{positive}\t{negative}\t{precision}")
        f.close()


while True:
    print(f"\r######{i}######", end="")
    mineList = []
    techList = []
    ret, frame = cap.read()
    if not ret:
        break
    nextFrame = frame
    noteFrame = nowFrame

    """
    预处理图像
    """
    diff1 = getDiff(beforeFrame, nowFrame)
    diff2 = getDiff(nowFrame, nextFrame)
    diff = cv2.bitwise_and(diff1, diff2)

    """
    车辆查找与画出矩形
    """
    contours, hierarchy = cv2.findContours(
        diff, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if x > 1500 or y < 400:
            continue
        if w * h > 10000:
            cv2.rectangle(noteFrame, (x, y), (x + w, y + h), (255, 255, 0), 2)
            mineList.append([x, y, w, h])
    """
    处理xml文件
    """
    if not os.path.exists("./Annotations/{}.xml".format(i)):
        i = i + 1
        beforeFrame = nowFrame
        nowFrame = nextFrame
        continue
    with open("./Annotations/{}.xml".format(i)) as f:
        dom = minidom.parse(f)
        xmins = dom.getElementsByTagName("xmin")
        a = []
        for xmin in xmins:
            a.append(int(xmin.childNodes[0].data))
        ymins = dom.getElementsByTagName("ymin")
        b = []
        for ymin in ymins:
            b.append(int(ymin.childNodes[0].data))
        xmaxs = dom.getElementsByTagName("xmax")
        c = []
        for xmax in xmaxs:
            c.append(int(xmax.childNodes[0].data))
        ymaxs = dom.getElementsByTagName("ymax")
        d = []
        for ymax in ymaxs:
            d.append(int(ymax.childNodes[0].data))
        for j in range(0, len(d)):
            if b[j] < 400 or a[j] > 1500:
                continue
            cv2.rectangle(noteFrame, (a[j], b[j]),
                          (a[j] + c[j], b[j] + d[j]), (0, 0, 255), 2)  # red color
            techList.append([a[j], b[j], c[j], d[j]])
        f.close()

    """
    处理IOU计算结果
    """
    output_IOU(techList, mineList)
    # cv2.imshow("diff", diff)
    cv2.imshow('video', noteFrame)
    cv2.waitKey(10)
    beforeFrame = nowFrame
    nowFrame = nextFrame
    i = i + 1

getPrecison()
cap.release()
cv2.destroyAllWindows()
