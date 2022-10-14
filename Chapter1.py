from queue import Empty
from re import S
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt




























"""Color Detection"""

"""
cv.namedWindow("TrackBars")#creating a gui window
cv.resizeWindow("TrackBars",640,240)
cv.createTrackbar("Hue Min","TrackBars",0,179,Empty)# creating kind of button of range 
cv.createTrackbar("Hue Max","TrackBars",19,179,Empty)
cv.createTrackbar("Sat Min","TrackBars",110,255,Empty)
cv.createTrackbar("Sat Max","TrackBars",240,255,Empty)
cv.createTrackbar("Val Min","TrackBars",153,255,Empty)
cv.createTrackbar("Val Max","TrackBars",255,255,Empty)

while True: # for color detection its need to continous so used loop 
   img = cv.imread("G:\DataScience\OpenCV\Learn-OpenCV-in-3-hours\Resources\lambo.PNG")
   imgHSV = cv.cvtColor(img,cv.COLOR_BGR2HSV) # Hsv must need 
   h_min = cv.getTrackbarPos("Hue Min","TrackBars") # connected button with mask 
   h_max = cv.getTrackbarPos("Hue Max","TrackBars")
   s_min = cv.getTrackbarPos("Sat Min","TrackBars")
   s_max = cv.getTrackbarPos("Sat Max","TrackBars")
   v_min = cv.getTrackbarPos("Val Min","TrackBars")
   v_max = cv.getTrackbarPos("Val Max","TrackBars")
   print(h_min,h_max,s_min,s_max,v_min,v_max)
   lower = np.array([h_min,s_min,v_min])
   upper = np.array([h_max,s_max,v_max])
   mask = cv.inRange(imgHSV,lower,upper)
   images= [img,imgHSV]
   np_horizontal = np.concatenate((img,imgHSV),axis=1) 
   imgResult = cv.bitwise_and(img,img,mask=mask) # 


   cv.imshow("img",np_horizontal)
   cv.imshow("Result",imgResult)
   cv.waitKey(1)"""


"""Wrap Prespective"""
# height,width = 718,408
# pts1 = np.float32([[702,223],[624,626],[1428,269],[1282,870]])
# pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
# matrix = cv.getPerspectiveTransform(pts1,pts2)
# img1 = cv.warpPerspective(img,matrix,(width,height))

# # cv.imshow("image", img)
# cv.imshow("2",img1)
# cv.waitKey(0)
# cv.destroyAllWindows()




"""1.OpenCV takes argument as (Width,Height)
   2.cv takes color parameter as (R,G,B) if 255 pixel image """

# for text in image
# cv.putText(img," Cat ",(250,390),cv.FONT_HERSHEY_COMPLEX,1,(250,255,43),4)

# for make line, rectangle, circle,
# Make cv.line(image, start_point, end_point, color, thickness)
# cv.line(img,(0,0),(480,640),(255,255,0))
# cv.rectangle(image, start_point, end_point, color, thickness)
# cv.rectangle(img,(5,5),(450,620),(0,255,255),2)
# cv.circle(image, center_coordinates, radius, color, thickness)
# cv.imshow("Image",img)
# cv.waitKey(0)


# img = cv.resize(img, (480, 640))  # resize(imgvariable,(width _ ,height | ))

# print(img.shape) # prints (Heigth,width,number of channel) here number of channel is BGR
# imgcrop = img[0:200,50:250] # [height,width] # cv mainly inverse i.e. (width,hieght)
# cv.imshow("cropped",imgcrop)


# # for gen new image and color
# #img = np.zeros((512,512,3),uint8) #gen new image
# #img[:] =500,500,500 # color


# # img = cv.imread("Resources/lina.jpg")
# # img = cv.resize(img,(640,480))


# # kernel = np.ones((5,5),np.uint8)

# # imgGray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)# rgb to gray
# # imgBlur = cv.GaussianBlur(imgGray,(7,7),0)# img to blur
# # imgCanny = cv.Canny(img,100,100)
# # imgDilation = cv.dilate(imgCanny,kernel,iterations=1)
# # imgEroded = cv.erode(imgDilation,kernel,iterations=1)

# # Titles =["Original","imgGray","imgBlur","imgCanny","imgDilation","imgEroded"]
# # images = [img,imgGray,imgBlur,imgCanny,imgDilation,imgEroded]

# # for i in range(len(images)):
# #     plt.subplot(3,3,i+1)
# #     plt.title(Titles[i])
# #     plt.imshow(images[i])

# # plt.show()

# # # cv.imshow("image",imgGray)
# # cv.imshow("1 image",imgBlur)
# # cv.imshow("2 image",imgEroded)
# # cv.imshow("3 ")
# # cv.waitKey(0)


# # vid = cv.VideoCapture(1) # 1 for webcam and dir path for video
# # vid.set(3,640)# for webcam setup
# # vid.set(4,480)# same
# # vid.set(10,100)# for brightness setup
# # while True:
# #     success, img = vid.read()
# #     cv.imshow("video",img)
# #     if cv.waitKey(1) & 0xFF == ord('q'):
# #         break


# # opening image
# # img = cv.imread("G:\DataScience\OpenCV\Resources\lina.jpg")
# # cv.imshow("output",img)
# # cv.waitKey(0)
