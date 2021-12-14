import cv2
import numpy as np

def read_img(filename):
    img = cv2.imread(filename)
    return img

def edge_detection(img, line_wdt, blur):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayBlur = cv2.medianBlur(gray, blur)
    # edges of the image
    edges = cv2.adaptiveThreshold(grayBlur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_wdt, blur)
    return edges

# cartoon painting the image and k - means clustering
# Clustering we take two colors one is blue and green colors on image here the clustering will perform divided same colors of image one side and green color will other side.
def color_quantization(img, k):
    # data = matrix of the image
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    result = center[label.flatten()]
    result = result.reshape(img.shape)
    return result


img = read_img('C:\\Users\\HP\\Cartoon_Effect_On_Image-Project\\Flower_Water_lily.jpg')
line_wdt = 9
blur_value = 7
totalColors = 6

edgeImg = edge_detection(img, line_wdt, blur_value)
img = color_quantization(img, totalColors)
blurred = cv2.bilateralFilter(img, d=7, sigmaColor=200, sigmaSpace=200)
cartoon = cv2.bitwise_and(blurred, blurred, mask=edgeImg)
cv2.imwrite('cartoon1.jpg', cartoon)





