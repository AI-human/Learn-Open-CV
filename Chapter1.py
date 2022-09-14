import cv2

img = cv2.imread("Resources/lina.jpg")

imgGray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
cv2.imshow("image",imgGray)
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