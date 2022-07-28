import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import datetime

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    start = time.time()
    image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_h, img_w, img_c = image.shape

    face_3d = []
    face_2d = []

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idX, lm in enumerate(face_landmarks.landmark):
                if idX == 33 or idX == 263 or idX == 1 or idX == 61 or idX == 291 or idX == 199:
                    if idX == 1:
                        nose_2d = (lm.x*img_w, lm.y*img_h)
                        nose_3d = (lm.x*img_w, lm.y*img_h, lm.z*3000)
                    x, y = int(lm.x*img_w), int(lm.y*img_h)
                    face_2d.append([x, y])
                    face_3d.append([x, y, lm.z])

            face_2d = np.array(face_2d, dtype=np.float64)
            face_3d = np.array(face_3d, dtype=np.float64)

            focal_length = 1 * img_w

            cam_matrix = np.array([[focal_length, 0, img_h/2],
                                   [0, focal_length, img_w/2],
                                   [0, 0, 1]])

            dist_matrix = np.zeros((4, 1), dtype=np.float64)

            success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

            rmat, jac = cv2.Rodrigues(rot_vec)

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

            x = angles[0] * 360
            y = angles[1] * 360
            z = angles[2] * 360

            if y < -10:
                text = "looking left"
            elif y > 10:
                text = "looking right"
            elif x < -10:
                text = "looking down"
            elif x > 10:
                text = "looking up"
            else:
                text = "forward"
                
            datetime_object=datetime.datetime.now()

            dict_1={"x":None,"y":None,"z":None,"text":None,"date_time":None}

            dict_1["x"]=x
            dict_1["y"]=y
            dict_1["z"]=z
            dict_1["text"]=text
            dict_1["date_time"]=datetime_object

            dict_new=[]
            dict_new.append(dict_1)

            print("dict_1",dict_1)

            headers=["x","y","z","text","date_time"]

            with open('Sample.csv','w',newline="") as writeobj:
                write=csv.DictWriter(writeobj,fieldnames=headers)
                write.writeheader()
                while cap.isOpened():
                    # for data in dict_new:
                    #     write.writerow(data)
                    write.writerows(dict_new)

            writeobj.close()

            nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

            cv2.line(image, p1, p2, (255, 0, 0), 3)

            cv2.putText(image, text, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
            cv2.putText(image, 'x: ' + str(np.round(x, 2)), (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, 'y: ' + str(np.round(y, 2)), (500, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(image, 'z: ' + str(np.round(z, 2)), (500, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        end = time.time()
        totaltime = end - start

        fps = 1 / totaltime
        print("FPS: ", fps)

        cv2.putText(image, f'FPS: {int(fps)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=drawing_spec,
            connection_drawing_spec=drawing_spec)

    cv2.imshow("Image", image)

    if cv2.waitKey(5) and 0xFF == 'k':
        break
