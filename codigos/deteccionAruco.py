import numpy as np
import cv2, PIL
from cv2 import aruco
from scipy.spatial.transform import Rotation as R

cap = cv2.VideoCapture(0)

gray=0

#Camara PS3

cameraMatrix = np.loadtxt("FotosCalibracion/cameraMatrix.txt", delimiter=",")
dist = np.loadtxt("FotosCalibracion/cameraDistortion.txt", delimiter=",")


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
        #print(tvec.shape)
        print("Posición X: "+str(tvec[0][0][0])+ " Y: "+str(tvec[0][0][1]) + " Z: "+str(tvec[0][0][2]))
        rotation_matrix, _ = cv2.Rodrigues(rvec[0])
        rotation = R.from_matrix(rotation_matrix)

        # Get Euler angles in degrees
        # 'xyz' means:
        # euler_x (roll) = rotation around X-axis
        # euler_y (pitch) = rotation around Y-axis
        # euler_z (yaw) = rotation around Z-axis
        euler_angles_degrees = rotation.as_euler('xyz', degrees=True)

        print(f"\nEuler Angles (XYZ convention, degrees):")
        print(f"  Rotation around X-axis (Roll): {euler_angles_degrees[0]:.2f} degrees")
        print(f"  Rotation around Y-axis (Pitch): {euler_angles_degrees[1]:.2f} degrees")
        print(f"  Rotation around Z-axis (Yaw): {euler_angles_degrees[2]:.2f} degrees")
        #print("Rotación X: "+str(rvec[0][0][0])+ " Y: "+str(rvec[0][0][1]) + " Z: "+str(rvec[0][0][2]))
        #print(tvec[0])
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


        #traslaciones = "X : {:.1f} Y : {:.1f} Z : {:.1f} ".format(tvec[0], tvec[1], tvec[2])
        cv2.putText(frame, "Id: " + strg, (0,64), font, 0.5, (0,0,255),2,cv2.LINE_AA)
        #cv2.putText(frame, traslaciones, (0,72), font, 0.5, (0,255,0),2,cv2.LINE_AA)

    else:
        # code to show 'No Ids' when no markers are found

        cv2.putText(frame, "No Ids", (0,64), font, 1, (0,0,255),2,cv2.LINE_AA)

    # display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
