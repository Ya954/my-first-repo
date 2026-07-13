import cv2

if __name__ == "__main__":
    kele = cv2.imread("kele.png")
    template = cv2.imread("template.png")
    cv2.imshow("kele", kele)
    cv2.imshow("template", template)
    cv2.waitKey(0)
    h, w = template.shape[:2]
    res = cv2.matchTemplate(kele, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    cv2.rectangle(kele, top_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow("match_result", kele)
    cv2.waitKey(0)
    cv2.destroyAllWindows()