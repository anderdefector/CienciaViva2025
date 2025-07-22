import cv2

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

num = 0

while(True):
    ret, frame = cap.read()
    nombre = "FotosCalibracion/Foto"+str(num)+".png"
    cv2.imshow("Vista", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite(nombre, frame)
        print("Foto "+str(num))
        num = num + 1

cap.release()
cv2.destroyAllWindows()
print("Adios..")
