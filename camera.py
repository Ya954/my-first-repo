import cv2

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Canny边缘检测
    canny_img = cv2.Canny(gray, 40, 160)
    # Sobel边缘检测
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, dx=1, dy=0)
    sobel_x = cv2.convertScaleAbs(sobel_x)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, dx=0, dy=1)
    sobel_y = cv2.convertScaleAbs(sobel_y)
    sobel_all = cv2.addWeighted(sobel_x,0.5,sobel_y,0.5,0)
    # 反向二值化：小于40置255，其余置0
    ret, sobel_bin_inv = cv2.threshold(sobel_all,40,255,cv2.THRESH_BINARY_INV)
    # 窗口显示
    cv2.imshow("Canny边缘", canny_img)
    cv2.imshow("Sobel原图", sobel_all)
    cv2.imshow("Sobel反向二值化", sobel_bin_inv)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()