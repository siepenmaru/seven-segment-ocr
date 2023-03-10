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
    # Canny edge detection
    out = cv2.Canny(img, 30, 200)

    if saves:
        cv2.imwrite('contours.jpg', out)

if __name__ == "__main__":
    img = cv2.imread('sample/sample_1.jpeg')
    binarize(img, saves=True)
    edge_detect(img, saves=True)
