import numpy as np
import imutils
import cv2

# TODO: Detect screen corners, automatically crop to isolate LCD output

def binarize(img, saves=False):
    # Resize image
    out = imutils.resize(img, height=300)

    # Convert to grayscale
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    # Removes glare and increases contrast
    out = cv2.createCLAHE().apply(out)

    kernel = np.ones((2,2), np.uint8)
    out = cv2.dilate(out, kernel)
    out = cv2.erode(out, kernel)
    
    # min_val, max_val, _, _ = cv2.minMaxLoc(out)
    # out = cv2.convertScaleAbs(out, alpha=255.0/(max_val-min_val), beta=-255.0*min_val/(max_val-min_val))
    
    # Reduce noise
    out = cv2.GaussianBlur(out, (5,5), 0)
    out = cv2.bilateralFilter(out, 11, 17, 17)

    # Squash values (binarization)
    # out = cv2.adaptiveThreshold(out, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, out = cv2.threshold(out, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    if saves:
        cv2.imwrite('binarized.jpg', out)

    return out

def edge_detect(img, saves=False):
    kernel = np.ones((3,3), np.uint8)
    big_kernel = np.ones((6,6), np.uint8)
    out = cv2.dilate(img, big_kernel)
    out = cv2.erode(out, big_kernel)
    out = cv2.dilate(out, kernel)
    out = cv2.erode(out, kernel)
    # Canny edge detection
    out = cv2.Canny(out, 50, 200)
    contours, _ = cv2.findContours(out, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    # out = cv2.HoughLines(out, 1, np.pi/180, 150, None, 0, 0)

    if saves:
        out = cv2.drawContours(out, contours, -1, 255, 2)
        out = 255 - out
        cv2.imwrite('contours.jpg', out)

def prune(img, contours, saves=False):
    res = np.zeros_like(img)
    unconnected = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 123 and area < 760000:
            cv2.drawContours(res [contour], 0, (255,255,255), cv2.FILLED)
            unconnected.append(contour)
    

def find_contours(img, saves=False):
    cnts = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    displayCnt = None

    for c in cnts:
        # Perform contour approximation
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            displayCnt = approx
            break
    
    if displayCnt is not None:
        warped = four_point_transform(img, np.reshape(displayCnt, (4, 2)))

        if saves:
            cv2.imwrite("warped.jpg", warped)

if __name__ == "__main__":
    img = cv2.imread('sample/sample_1.jpeg')
    binarized = binarize(img, saves=True)
    edge_detect(binarized, saves=True)
    # find_contours(binarized, saves=True)

    # reader = easyocr.Reader(['en'])
    # print(reader.readtext(binarized))
