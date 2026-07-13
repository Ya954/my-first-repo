import cv2

if __name__ == "__main__":
    # 1 hua.png轮廓处理
    hua = cv2.imread('hua.png')
    hua_gray = cv2.cvtColor(hua, cv2.COLOR_BGR2GRAY)
    cv2.imshow('hua_b', hua_gray)
    cv2.waitKey(0)
    ret, hua_binary = cv2.threshold(hua_gray, 240, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow('hua_binary', hua_binary)
    cv2.waitKey(0)
    _, contours, hierarchy = cv2.findContours(hua_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(f"轮廓总数:{len(contours)}")
    image_copy = hua.copy()
    sortcnt = sorted(contours, key=cv2.contourArea, reverse=True)[0]
    cv2.drawContours(image_copy, [sortcnt], -1, (0, 255, 0), 3)
    cv2.imshow('image_contours', image_copy)
    cv2.waitKey(0)
    # 轮廓近似
    epsilon = 0.005 * cv2.arcLength(sortcnt, True)
    approx = cv2.approxPolyDP(sortcnt, epsilon, True)
    print("原轮廓点数", sortcnt.shape[0])
    print("近似之后点数", approx.shape[0])
    cv2.drawContours(image_copy, [approx], -1, (0, 255, 0),3)
    cv2.imshow('image_contours', image_copy)
    cv2.waitKey(0)
    # 轮廓面积、周长、外接矩形、外接圆
    area = cv2.contourArea(sortcnt)
    length = cv2.arcLength(sortcnt, True)
    print(f"轮廓面积:{area},轮廓周长:{length}")
    x, y, w, h = cv2.boundingRect(sortcnt)
    cv2.rectangle(image_copy, (x, y), (x + w, y + h), (0, 0, 255), 2)
    (cx, cy), r = cv2.minEnclosingCircle(sortcnt)
    cv2.circle(image_copy, (int(cx), int(cy)), int(r), (255, 0, 0), 2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()