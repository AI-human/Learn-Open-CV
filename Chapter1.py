import cv2
import numpy as np 
import matplotlib.pyplot as plt

img = cv2.imread("Resources/lina.jpg")
img = cv2.resize(img,(640,480))


kernel = np.ones((5,5),np.uint8)

imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)# rgb to gray
imgBlur = cv2.GaussianBlur(imgGray,(7,7),0)# img to blur
imgCanny = cv2.Canny(img,100,100)
imgDilation = cv2.dilate(imgCanny,kernel,iterations=1)
imgEroded = cv2.erode(imgDilation,kernel,iterations=1)

Titles =["Original","imgGray","imgBlur","imgCanny","imgDilation","imgEroded"]
images = [img,imgGray,imgBlur,imgCanny,imgDilation,imgEroded]

# for i in range(len(images)):
#     plt.subplot(3,3,i+1)
#     plt.title(Titles[i])
#     plt.imshow(images[i])

# plt.show()

# # cv2.imshow("image",imgGray)
# cv2.imshow("1 image",imgBlur)
cv2.imshow("2 image",imgEroded)
# cv2.imshow("3 ")
cv2.waitKey(0)






# vid = cv2.VideoCapture(1) # 1 for webcam and dir path for video
# vid.set(3,640)# for webcam setup
# vid.set(4,480)# same
# vid.set(10,100)# for brightness setup
# while True:
#     success, img = vid.read()
#     cv2.imshow("video",img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break



# opening image
# img = cv2.imread("G:\DataScience\OpenCV\Resources\lina.jpg")
# cv2.imshow("output",img)
# cv2.waitKey(0)   