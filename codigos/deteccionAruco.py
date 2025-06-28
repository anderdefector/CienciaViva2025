import numpy as np
import cv2, PIL
from cv2 import aruco




cap = cv2.VideoCapture(0)

#im_src = cv2.imread("Img/muse.jpg")
#im_src = cv2.flip(im_src, 0)

#width, height, channels = im_src.shape




gray=0

#Camara PS3

cameraMatrix = np.array([[5.109663638795043994e+02, 0, 3.268095902480743007e+02], 
                        [ 0, 5.103928144841747780e+02, 2.631678319887266753e+02], 
                        [0.000000, 0.000000, 1.000000]])
dist = np.array([[-1.504925480024328077e-01], 
                [2.893519143083206902e-01], 
                [2.807256039291659497e-03], 
                [4.306493998655061203e-03], 
                [-1.784404386540009158e-01]])





while (True):
    ret, frame = cap.read()

    frame = cv2.undistort(frame, cameraMatrix, dist)

    # operations on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # set dictionary size depending on the aruco marker selected
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_50)

    # detector parameters can be set here (List of detection parameters[3])
    parameters = aruco.DetectorParameters()
    parameters.adaptiveThreshConstant = 10

    # lists of ids and the corners belonging to each id
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    pts = np.empty([4, 2], dtype=int)

    # font for displaying text (below)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # check if the ids list is not empty
    # if no check is added the code will crash
    if np.all(ids != None):

        for x in range(0, len(corners)):
            pts = np.delete(pts,x, axis=0)
            pts = np.insert(pts,x,corners[x][[0],[0]],axis = 0)
        
        #im_out = cv2.add(warped_image, frame)
        # estimate pose of e    ach marker and return the values
        # rvet and tvec-different from camera coefficients
        rvec, tvec ,_ = aruco.estimatePoseSingleMarkers(corners, 0.10, cameraMatrix, dist)
        #print(rvec)
        print(tvec)
        #(rvec-tvec).any() # get rid of that nasty numpy value array error

        for i in range(0, ids.size):
            # draw axis for the aruco markers
            cv2.drawFrameAxes(frame, cameraMatrix, dist, rvec[i], tvec[i], 0.1)

        # draw a square around the markers
        aruco.drawDetectedMarkers(frame, corners)


        # code to show ids of the marker found
        strg = ''
        for i in range(0, ids.size):
            strg += str(ids[i][0])+', '


        traslaciones = "X : {:.1f} Y : {:.1f} Z : {:.1f} ".format(tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, "Id: " + strg, (0,64), font, 0.5, (0,255,0),2,cv2.LINE_AA)
        cv2.putText(frame, traslaciones, (0,72), font, 0.5, (0,255,0),2,cv2.LINE_AA)

    else:
        # code to show 'No Ids' when no markers are found

        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,255,0),2,cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
