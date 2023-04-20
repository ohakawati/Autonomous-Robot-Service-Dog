import math

import cv2
import pickle
import cv2.aruco as aruco

#droid cam
url=""
#ipwebcam
url2=""
def find_markers(img, markersize=6, totalmarkers=250,draw=True):
    grayscale=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    key=getattr(aruco,f"DICT_{markersize}X{markersize}_{totalmarkers}")
    aruco_dict=aruco.Dictionary_get(key)
    arucoParamater=aruco.DetectorParameters_create()
    corners,ids,rejected=aruco.detectMarkers(img,aruco_dict,parameters=arucoParamater)
    if draw:
        aruco.drawDetectedMarkers(img,corners=corners,ids=ids)


    return [corners,ids]






def distance_approx(corners, markerlength):
    with open("calibration.pckl", "rb") as file:
        cameraMatrix, distCoeffs, rvecs, tvecs=pickle.load(file)
    rv,tv,objpoints=aruco.estimatePoseSingleMarkers(corners=corners, markerLength=markerlength,cameraMatrix=cameraMatrix,distCoeffs=distCoeffs)
    #tv[0][0][0] return value in meters
    x_m=tv[0][0][0]/10
    y_m=tv[0][0][1]/10
    z_m=tv[0][0][2]/10
    total_dis=math.sqrt(x_m**2+y_m**2+z_m**2)
    total_dis_in=total_dis*39.3701
    return [total_dis,total_dis_in]


def main():
    cap=cv2.VideoCapture(url2)
    #fourc=cv2.VideoWriter_fourcc("M", "J", "P", "G")
    #width=1920
    height=1080
    #cap.set(6,fourc)
    #cap.set(3,width)
    #cap.set(4,height)
    while True:

        _,frame=cap.read()
        found=find_markers(frame,6,totalmarkers=1000,draw=True)
        #if found[1] is not None:
            #dist=distance_approx(found[0], markerlength=0.025)
            #print(cap.get(3))
            #print(cap.get(4))
            #print(f"distance in meters = {dist[0]*2.25}\n")
            #print(f"distance in inches = {dist[1]*2.25}")

        cv2.imshow("window",frame)
        if cv2.waitKey(1)==32:
            break
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()