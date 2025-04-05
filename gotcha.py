import cv2


face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

video_capture = cv2.VideoCapture(0)

def makeRectangle(frame):
    grey_scale_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey_scale_img, 1.1, 6, minSize=(50, 50))
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 8)

while True:
    result, video_frame = video_capture.read()
    if result is False:
        print("Error in Reading.")
        break

    faces = makeRectangle(video_frame)
    cv2.imshow("Real Time Cheating Detection", video_frame)
    if cv2.waitKey(1) & 0xFF == ord("a"):
        break

video_capture.release()
cv2.destroyAllWindows()
