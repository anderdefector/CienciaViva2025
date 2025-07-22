import cv2
import mediapipe as mp
from cv2 import aruco
import numpy as np

class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=1, detectionCon=0.5, trackCon=0.5):
        self.__mode__   =  mode
        self.__maxHands__   =  maxHands
        self.__detectionCon__   =   detectionCon
        self.__trackCon__   =   trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw= mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)  
        if self.results.multi_hand_landmarks: 
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms,self.handsMp.HAND_CONNECTIONS)

        return frame
 
    def findPosition( self, frame, handNo=0, draw=True):
        xList =[]
        yList =[]
        bbox = []
        self.lmsList=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])
                if draw:
                    cv2.circle(frame,  (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            #print( "Hands Keypoint")
            #print(bbox)
            
            #centro = calcularCentro(ymin, ymax, xmin, xmax)
            #print(centro)
            #cv2.circle(frame,(xmin,ymin),5,(255,0,0),cv2.FILLED)
            #cv2.circle(frame,(xmax,ymax),5,(255,0,0),cv2.FILLED)
            #cv2.circle(frame,(centro[0],centro[1]),5,(255,0,0),cv2.FILLED)
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20),(xmax + 20, ymax + 20),
                               (0, 255 , 0) , 2)

        return self.lmsList, bbox
    
    def findFingerUp(self):
         fingers=[]

         if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0]-1][1]:
              fingers.append(1)
         else:
              fingers.append(0)

         for id in range(1, 5):            
              if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id]-2][2]:
                   fingers.append(1)
              else:
                   fingers.append(0)
        
         return fingers

    def findDistance(self, p1, p2, frame, draw= True, r=15, t=3):
         
        x1 , y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx , cy = (x1+x2)//2 , (y1 + y2)//2

        if draw:
              cv2.line(frame,(x1, y1),(x2,y2) ,(255,0,255), t)
              cv2.circle(frame,(x1,y1),r,(255,0,255),cv2.FILLED)
              cv2.circle(frame,(x2,y2),r, (255,0,0),cv2.FILLED)
              cv2.circle(frame,(cx,cy), r,(0,0.255),cv2.FILLED)
        len= math.hypot(x2-x1,y2-y1)

        return len, frame , [x1, y1, x2, y2, cx, cy]

def main():
        
        ctime=0
        ptime=0
        cap = cv2.VideoCapture(0)
        detector = HandTrackingDynamic()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        while True:
            ret, frame = cap.read()
            ar = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

             # detector parameters can be set here (List of detection parameters[3])
            parameters = aruco.DetectorParameters()
            parameters.adaptiveThreshConstant = 10

            # lists of ids and the corners belonging to each id
            corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

            
            #M, mask = cv2.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
            frame = detector.findFingers(frame)
            lmsList = detector.findPosition(frame)
            if(lmsList[1] != []):
                dedos = detector.findFingerUp()

                if(dedos == [0,1,0,0,0]):
                    ruta = "archivos/illit.jpeg"
                    
                elif(dedos == [0,1,1,0,0]):
                    ruta = "archivos/muse.jpeg"
                
                elif(dedos == [0,1,1,1,0]):
                    ruta = "archivos/twice.jpg"
                
                elif(dedos == [0,1,1,1,1]):
                    ruta = "archivos/paramore.jpg"
                elif(dedos == [1,1,1,1,1]):
                    ruta = "archivos/nayeon.jpeg"
                else:
                    ruta ="archivos/happy.jpg"

                image = cv2.imread(ruta, cv2.IMREAD_COLOR)
                resized_image = cv2.resize(image, (480, 480))
                vertices_origen = np.array([[0, 0],[480,0], [480, 480], [0,480]])
                #print(vertices_origen.shape)
                if len(corners) == 1:
                    print(corners[0])
                    vertices = corners[0][0]
                    print(vertices.shape)
                    print(type(vertices))

                    M, _ = cv2.findHomography(vertices_origen, vertices)

                    warped_img = cv2.warpPerspective(resized_image, M, (640,480))
                    img2gray = cv2.cvtColor(warped_img,cv2.COLOR_BGR2GRAY)
                    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
                    mask_inv = cv2.bitwise_not(mask)

                    #img1_bg = cv.bitwise_and(,roi,mask = mask_inv)
                    img1_fg = cv2.bitwise_and(ar,ar,mask = mask_inv)
                    img2_fg = cv2.bitwise_and(warped_img,warped_img,mask = mask)
                    dst = cv2.add(img1_fg,img2_fg)
                    #cv2.imshow("Prueba",resized_image) 
                    #cv2.imshow("Transformada", warped_img)s

                    #cv2.imshow("Transformada2", mask)
                    cv2.imshow("Salida", dst)

            else:
                print("No hay ninguna mano.")
                cv2.destroyWindow('Prueba')
            frame = cv2.flip(frame,1)
            cv2.imshow('Ventana', frame)
            cv2.waitKey(1)


                
if __name__ == "__main__":
            main()