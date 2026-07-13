import cv2

if __name__ == "__main__":
    zl = cv2.imread("img1.jpg", cv2.IMREAD_GRAYSCALE)
    cv2.imshow("zl", zl)
    cv2.waitKey(0)
    # Scharr算子
    scharr_x = cv2.Scharr(zl, cv2.CV_64F, dx=1, dy=0)
    scharr_x = cv2.convertScaleAbs(scharr_x)
    scharr_y = cv2.Scharr(zl, cv2.CV_64F, dx=0, dy=1)
    scharr_y = cv2.convertScaleAbs(scharr_y)
    scharr_total = cv2.addWeighted(scharr_x,1,scharr_y,1,0)
    cv2.imshow("scharr result", scharr_total)
    cv2.waitKey(0)
    # Laplacian算子
    lap = cv2.Laplacian(zl, cv2.CV_64F, ksize=3)
    lap = cv2.convertScaleAbs(lap)
    cv2.imshow("laplacian", lap)
    cv2.waitKey(0)
    # Canny算子
    canny_out = cv2.Canny(zl, 100, 150)
    cv2.imshow("canny", canny_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()