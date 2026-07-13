import numpy as np
import cv2
import myutils
import sys

def cv_show(name, img):
    if img is not None:
        cv2.imshow(name, img)
        cv2.waitKey(0)

if __name__ == "__main__":
    # 现在图片和代码在同一个文件夹，使用相对路径
    image_path = "card1.png"
    template_path = "kahao.jpg"

    FIRST_NUMBER = {
        "3": "American Express",
        "4": "Visa",
        "5": "MasterCard",
        "6": "Discover Card"
    }

    img = cv2.imread(template_path)
    if img is None:
        print("读取失败：card文件夹里找不到 kahao.png")
        sys.exit()
    cv_show('img', img)
    ref = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv_show('ref', ref)
    ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
    cv_show('ref', ref)

    refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cv2.drawContours(img, refCnts, -1, (0, 255, 0), 3)
    cv_show('refCnts', img)
    refCnts, _ = myutils.sort_contours(refCnts, method="left-to-right")
    digits = {}

    for (i, c) in enumerate(refCnts):
        x, y, w, h = cv2.boundingRect(c)
        roi = ref[y:y + h, x:x + w]
        roi = cv2.resize(roi, (57, 88))
        digits[i] = roi

    image = cv2.imread(image_path)
    if image is None:
        print("读取失败：card文件夹里找不到 card1.png")
        sys.exit()
    cv_show('image', image)
    image = myutils.resize(image, width=300)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv_show('gray', gray)

    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, rectKernel)
    cv_show('tophat', tophat)

    closeX = cv2.morphologyEx(tophat, cv2.MORPH_CLOSE, rectKernel)
    cv_show('closeX', closeX)
    thresh = cv2.threshold(closeX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    cv_show('thresh', thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    cv_show('close2', thresh)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts_img = image.copy()
    cv2.drawContours(cnts_img, cnts, -1, (0, 0, 255), 3)
    cv_show('cnts_img', cnts_img)

    locs = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / float(h)
        if 2.5 < ar < 4.0:
            if (40 < w < 55) and (10 < h < 20):
                locs.append((x, y, w, h))
    locs = sorted(locs, key=lambda x: x[0])

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
            scores = []
            for (digit, digitROI) in digits.items():
                result = cv2.matchTemplate(roi, digitROI, cv2.TM_CCOEFF)
                _, score, _, _ = cv2.minMaxLoc(result)
                scores.append(score)
            groupOutput.append(str(np.argmax(scores)))
        cv2.rectangle(image, (gX - 5, gY - 5), (gX + gW + 5, gY + gH + 5), (0, 0, 255), 1)
        cv2.putText(image, "".join(groupOutput), (gX, gY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        output.extend(groupOutput)

    print("Credit Card Type: {}".format(FIRST_NUMBER[output[0]]))
    print("Credit Card #: {}".format("".join(output)))
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()