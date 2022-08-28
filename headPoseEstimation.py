import argparse
import math
from datetime import datetime, timedelta
import cv2
import mediapipe as mp
import numpy as np
import csv

header = ["datetime", "x", "y", "z", "mar", "activity"]
log = {}
level_text = ''

right_eye = [[33, 133], [160, 144], [159, 145], [158, 153]]  # right eye landmark positions
left_eye = [[263, 362], [387, 373], [386, 374], [385, 380]]  # left eye landmark positions
mouth = [[61, 291], [39, 181], [0, 17], [269, 405]]  # mouth landmark coordinates
states = ['alert', 'drowsy']


def euclideanDistance(point, point1):
    x, y = point
    x1, y1 = point1
    distance = math.sqrt((x1 - x) ** 2 + (y1 - y) ** 2)
    return distance


def blinkRatio(landmarks, right_indices, left_indices):
    # Right eyes
    # horizontal line
    rh_right = landmarks[right_indices[0]]
    rh_left = landmarks[right_indices[8]]
    # vertical line
    rv_top = landmarks[right_indices[12]]
    rv_bottom = landmarks[right_indices[4]]

    # LEFT_EYE
    # horizontal line
    lh_right = landmarks[left_indices[0]]
    lh_left = landmarks[left_indices[8]]
    # vertical line
    lv_top = landmarks[left_indices[12]]
    lv_bottom = landmarks[left_indices[4]]

    rhDistance = euclideanDistance(rh_right, rh_left)
    rvDistance = euclideanDistance(rv_top, rv_bottom)

    lvDistance = euclideanDistance(lv_top, lv_bottom)
    lhDistance = euclideanDistance(lh_right, lh_left)

    reRatio = rhDistance / rvDistance
    leRatio = lhDistance / lvDistance

    ratio = (reRatio + leRatio) / 2
    return ratio


def eye_feature(landmarks):
    """ Calculate the eye feature as the average of the eye aspect ratio for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Eye feature value
    """
    return (eye_aspect_ratio(landmarks, left_eye) + eye_aspect_ratio(landmarks, right_eye)) / 2


def distance(p1, p2):
    """ Calculate distance between two points
    :param p1: First Point
    :param p2: Second Point
    :return: Euclidean distance between the points. (Using only the x and y coordinates).
    """
    return (((p1[:2] - p2[:2]) ** 2).sum()) ** 0.5


def eye_aspect_ratio(landmarks, eye):
    """ Calculate the ratio of the eye length to eye width.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Eye aspect ratio value
    """
    N1 = distance(landmarks[eye[1][0]], landmarks[eye[1][1]])
    N2 = distance(landmarks[eye[2][0]], landmarks[eye[2][1]])
    N3 = distance(landmarks[eye[3][0]], landmarks[eye[3][1]])
    D = distance(landmarks[eye[0][0]], landmarks[eye[0][1]])
    return (N1 + N2 + N3) / (3 * D)


def mouth_feature(landmarks):
    """ Calculate mouth feature as the ratio of the mouth length to mouth width
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Mouth feature value
    """
    N1 = distance(landmarks[mouth[1][0]], landmarks[mouth[1][1]])
    N2 = distance(landmarks[mouth[2][0]], landmarks[mouth[2][1]])
    N3 = distance(landmarks[mouth[3][0]], landmarks[mouth[3][1]])
    D = distance(landmarks[mouth[0][0]], landmarks[mouth[0][1]])
    return (N1 + N2 + N3) / (3 * D)


def pupil_circularity(landmarks, eye):
    """ Calculate pupil circularity feature.
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :param eye: List containing positions which correspond to the eye
    :return: Pupil circularity for the eye coordinates
    """
    perimeter = distance(landmarks[eye[0][0]], landmarks[eye[1][0]]) + \
                distance(landmarks[eye[1][0]], landmarks[eye[2][0]]) + \
                distance(landmarks[eye[2][0]], landmarks[eye[3][0]]) + \
                distance(landmarks[eye[3][0]], landmarks[eye[0][1]]) + \
                distance(landmarks[eye[0][1]], landmarks[eye[3][1]]) + \
                distance(landmarks[eye[3][1]], landmarks[eye[2][1]]) + \
                distance(landmarks[eye[2][1]], landmarks[eye[1][1]]) + \
                distance(landmarks[eye[1][1]], landmarks[eye[0][0]])
    area = math.pi * ((distance(landmarks[eye[1][0]], landmarks[eye[3][1]]) * 0.5) ** 2)
    return (4 * math.pi * area) / (perimeter ** 2)


def pupil_feature(landmarks):
    """ Calculate the pupil feature as the average of the pupil circularity for the two eyes
    :param landmarks: Face Landmarks returned from FaceMesh MediaPipe model
    :return: Pupil feature value
    """
    return (pupil_circularity(landmarks, left_eye) + pupil_circularity(landmarks, right_eye)) / 2


def run_face_mp(image, face_mesh):
    """ Get face landmarks using the FaceMesh MediaPipe model.
    Calculate facial features using the landmarks.
    :param face_mesh:
    :param image: Image for which to get the face landmarks
    :return: Feature 1 (Eye), Feature 2 (Mouth), Feature 3 (Pupil), \
        Feature 4 (Combined eye and mouth feature), image with mesh drawings
    """
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)

    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.multi_face_landmarks:
        landmarks_positions = []
        # assume that only face is present in the image
        for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
            landmarks_positions.append(
                [data_point.x, data_point.y, data_point.z])  # saving normalized landmark positions
        landmarks_positions = np.array(landmarks_positions)
        landmarks_positions[:, 0] *= image.shape[1]
        landmarks_positions[:, 1] *= image.shape[0]

        ear = eye_feature(landmarks_positions)
        mar = mouth_feature(landmarks_positions)
        puc = pupil_feature(landmarks_positions)
        moe = mar / ear
    else:
        ear = -1000
        mar = -1000
        puc = -1000
        moe = -1000

    return ear, mar, puc, moe, image


class VideoAnnotation:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5, thickness=1, circle_radius=1,
                 y_min=-5, y_max=8, x_min=-7, x_max=7, frame_rate_val=30, user='sugandh'):
        self.thickness = thickness
        self.min_tracking_confidence = min_tracking_confidence
        self.min_detection_confidence = min_detection_confidence
        self.circle_radius = circle_radius
        self.file_name = ''

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils

        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.y_min = y_min
        self.y_max = y_max
        self.x_min = x_min
        self.x_max = x_max
        self.frame_rate_val = frame_rate_val
        self.user = user
        self.calib_frame_count = 0
        self.ears = []
        self.mars = []
        self.pucs = []
        self.moes = []
        self.flag = 0
        self.ears_norm = 0
        self.mars_norm = 0
        self.pucs_norm = 0
        self.moes_norm = 0
        self.ear_main = 0
        self.mar_main = 0
        self.puc_main = 0
        self.moe_main = 0
        self.decay = 0.9  # use decay to smoothen the noise in feature values
        self.input_data = []
        self.frame_before_run = 0

    # Euclaidean distance

    def process(self, file_name):
        level_text = ''
        cap = cv2.VideoCapture(file_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps: ', fps)
        starttime = datetime.strptime(file_name.split('/')[-1].split('.')[0], "%Y%m%d_%H_%M_%S")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                self.calib_frame_count += 1
                frame_count += 1
                seconds = round(frame_count / fps)
                video_time = starttime + timedelta(seconds=seconds)
                if self.calib_frame_count <= fps:
                    ear, mar, puc, moe, image = run_face_mp(frame, self.face_mesh)
                    self.ears.append(ear)
                    self.mars.append(mar)
                    self.pucs.append(puc)
                    self.moes.append(moe)
                    continue

                if self.flag == 0:
                    self.ears = np.array(self.ears)
                    self.mars = np.array(self.mars)
                    self.pucs = np.array(self.pucs)
                    self.moes = np.array(self.moes)
                    self.ears_norm, self.mars_norm, self.pucs_norm, self.moes_norm = [self.ears.mean(), self.ears.std()], \
                                                                 [self.mars.mean(), self.mars.std()], \
                                                                 [self.pucs.mean(), self.pucs.std()], \
                                                                 [self.moes.mean(), self.moes.std()]
                    self.flag = 1

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_h, img_w, img_c = image.shape

                face_3d = []
                face_2d = []

                if results.multi_face_landmarks:
                    landmarks_positions = []
                    # assume that only face is present in the image
                    for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                        landmarks_positions.append(
                            [data_point.x, data_point.y, data_point.z])  # saving normalized landmark positions
                    landmarks_positions = np.array(landmarks_positions)
                    landmarks_positions[:, 0] *= image.shape[1]
                    landmarks_positions[:, 1] *= image.shape[0]

                    for face_landmarks in results.multi_face_landmarks:
                        for idX, lm in enumerate(face_landmarks.landmark):
                            if idX == 33 or idX == 263 or idX == 1 or idX == 61 or idX == 291 or idX == 199:
                                if idX == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])

                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        focal_length = 1 * img_w

                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])

                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        rmat, jac = cv2.Rodrigues(rot_vec)

                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

                        yaw_min = -10
                        yaw_max = 7
                        for_yaw_min = -4
                        for_yaw_max = 6
                        for_pitch_min = 0
                        for_pitch_max = 9
                        talk_min = 0.5
                        talk_max = 3
                        yawn_min = 4
                        up_pitch_min = 10
                        up_yaw_min = -4
                        up_yaw_max = 6

                        if y < yaw_min:
                            level_text = "looking right"
                        elif y > yaw_max:
                            level_text = "looking left"
                        elif for_yaw_min < y < for_yaw_max and for_pitch_min < x < for_pitch_max:
                            if talk_min < self.mar_main < talk_max:
                                level_text = "Talking"
                            elif self.mar_main > yawn_min:
                                level_text = "yawning"
                            else:
                                level_text = "looking forward"
                        elif x > up_pitch_min and up_yaw_min < y < up_yaw_max:
                            level_text = "looking up"
                        log = {"datetime": video_time, "activity": level_text}
                        print(log)
                        # datetime_object = datetime.datetime.now()

                        nose_3d_projection, jacobian = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix,
                                                                         dist_matrix)

                        p1 = (int(nose_2d[0]), int(nose_2d[1]))
                        p2 = (int(nose_2d[0] + y * 10), int(nose_2d[1] - x * 10))

                        cv2.line(image, p1, p2, (255, 0, 0), 3)

                        cv2.putText(image, level_text, (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                        cv2.putText(image, 'x: ' + str(np.round(x, 2)), (800, 250), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        cv2.putText(image, 'y: ' + str(np.round(y, 2)), (800, 300), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
                        cv2.putText(image, 'z: ' + str(np.round(z, 2)), (800, 350), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

                    cv2.putText(image, str(video_time), (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)

                    self.mp_drawing.draw_landmarks(
                        image=image,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=self.drawing_spec,
                        connection_drawing_spec=self.drawing_spec)

                    ear = eye_feature(landmarks_positions)
                    mar = mouth_feature(landmarks_positions)
                    puc = pupil_feature(landmarks_positions)
                    moe = mar / ear
                else:
                    ear = -1000
                    mar = -1000
                    puc = -1000
                    moe = -1000
                if ear != -1000:
                    ear = (ear - self.ears_norm[0]) / self.ears_norm[1]
                    mar = (mar - self.mars_norm[0]) / self.mars_norm[1]
                    puc = (puc - self.pucs_norm[0]) / self.pucs_norm[1]
                    moe = (moe - self.moes_norm[0]) / self.moes_norm[1]
                    if self.ear_main == -1000:
                        self.ear_main = ear
                        self.mar_main = mar
                        self.puc_main = puc
                        self.moe_main = moe
                    else:
                        self.ear_main = self.ear_main * self.decay + (1 - self.decay) * ear
                        self.mar_main = self.mar_main * self.decay + (1 - self.decay) * mar
                        self.puc_main = self.puc_main * self.decay + (1 - self.decay) * puc
                        self.moe_main = self.moe_main * self.decay + (1 - self.decay) * moe
                else:
                    self.ear_main = -1000
                    self.mar_main = -1000
                    self.puc_main = -1000
                    self.moe_main = -1000

                if len(self.input_data) == 20:
                    self.input_data.pop(0)
                self.input_data.append([self.ear_main, self.mar_main, self.puc_main, self.moe_main])

                self.frame_before_run += 1
                if self.frame_before_run >= 15 and len(self.input_data) == 20:
                    self.frame_before_run = 0

                print('Datetime, Ear, MAR, PUC, MOE ', (video_time, self.ear_main, self.mar_main, self.puc_main, self.moe_main))
                yawning = False
                if self.mar_main > 28:
                    yawning = True

                cv2.putText(image, 'MAR: ' + str(self.mar_main), (800, 400), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
                cv2.putText(image, f'Yawning: {yawning}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, 2)
                cv2.imshow("Image", image)
                if cv2.waitKey(self.frame_rate_val) & 0xFF == ord('s'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()

    def takelog(self, file_name):
        level_text = ''
        csv_file_name = file_name.split('.')[0] + '.csv'

        with open(csv_file_name, 'w') as f:
            csv.DictWriter(f, fieldnames=header).writeheader()
        cap = cv2.VideoCapture(file_name)
        fps = cap.get(cv2.CAP_PROP_FPS)
        print('fps: ', fps)
        starttime = datetime.strptime(file_name.split('/')[-1].split('.')[0], "%Y%m%d_%H_%M_%S")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if ret is True:
                self.calib_frame_count += 1
                frame_count += 1
                seconds = round(frame_count / fps)
                video_time = starttime + timedelta(seconds=seconds)
                if self.calib_frame_count <= fps:
                    ear, mar, puc, moe, image = run_face_mp(frame, self.face_mesh)
                    self.ears.append(ear)
                    self.mars.append(mar)
                    self.pucs.append(puc)
                    self.moes.append(moe)
                    continue

                if self.flag == 0:
                    self.ears = np.array(self.ears)
                    self.mars = np.array(self.mars)
                    self.pucs = np.array(self.pucs)
                    self.moes = np.array(self.moes)
                    self.ears_norm, self.mars_norm, self.pucs_norm, self.moes_norm = [self.ears.mean(),
                                                                                      self.ears.std()], \
                                                                                     [self.mars.mean(),
                                                                                      self.mars.std()], \
                                                                                     [self.pucs.mean(),
                                                                                      self.pucs.std()], \
                                                                                     [self.moes.mean(), self.moes.std()]
                    self.flag = 1

                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = self.face_mesh.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                img_h, img_w, img_c = image.shape

                face_3d = []
                face_2d = []

                if results.multi_face_landmarks:
                    landmarks_positions = []
                    # assume that only face is present in the image
                    for _, data_point in enumerate(results.multi_face_landmarks[0].landmark):
                        landmarks_positions.append(
                            [data_point.x, data_point.y, data_point.z])  # saving normalized landmark positions
                    landmarks_positions = np.array(landmarks_positions)
                    landmarks_positions[:, 0] *= image.shape[1]
                    landmarks_positions[:, 1] *= image.shape[0]

                    for face_landmarks in results.multi_face_landmarks:
                        for idX, lm in enumerate(face_landmarks.landmark):
                            if idX == 33 or idX == 263 or idX == 1 or idX == 61 or idX == 291 or idX == 199:
                                if idX == 1:
                                    nose_2d = (lm.x * img_w, lm.y * img_h)
                                    nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)
                                x, y = int(lm.x * img_w), int(lm.y * img_h)
                                face_2d.append([x, y])
                                face_3d.append([x, y, lm.z])

                        face_2d = np.array(face_2d, dtype=np.float64)
                        face_3d = np.array(face_3d, dtype=np.float64)

                        focal_length = 1 * img_w

                        cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                               [0, focal_length, img_w / 2],
                                               [0, 0, 1]])

                        dist_matrix = np.zeros((4, 1), dtype=np.float64)

                        success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                        rmat, jac = cv2.Rodrigues(rot_vec)

                        angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

                        x = angles[0] * 360
                        y = angles[1] * 360
                        z = angles[2] * 360

                        yaw_min = -10
                        yaw_max = 7
                        for_yaw_min = -4
                        for_yaw_max = 6
                        for_pitch_min = 0
                        for_pitch_max = 9
                        talk_min = 0.5
                        talk_max = 3
                        yawn_min = 4
                        up_pitch_min = 10
                        up_yaw_min = -4
                        up_yaw_max = 6

                        if y < yaw_min:
                            level_text = "looking right"
                        elif y > yaw_max:
                            level_text = "looking left"
                        elif for_yaw_min < y < for_yaw_max and for_pitch_min < x < for_pitch_max:
                            if talk_min < self.mar_main < talk_max:
                                level_text = "Talking"
                            elif self.mar_main > yawn_min:
                                level_text = "yawning"
                            else:
                                level_text = "looking forward"
                        elif x > up_pitch_min and up_yaw_min < y < up_yaw_max:
                            level_text = "looking up"
                        log = {"datetime": video_time, "x": x, "y": y, "z": z, "mar": self.mar_main, "activity": level_text}

                    ear = eye_feature(landmarks_positions)
                    mar = mouth_feature(landmarks_positions)
                    puc = pupil_feature(landmarks_positions)
                    moe = mar / ear
                else:
                    ear = -1000
                    mar = -1000
                    puc = -1000
                    moe = -1000
                if ear != -1000:
                    ear = (ear - self.ears_norm[0]) / self.ears_norm[1]
                    mar = (mar - self.mars_norm[0]) / self.mars_norm[1]
                    puc = (puc - self.pucs_norm[0]) / self.pucs_norm[1]
                    moe = (moe - self.moes_norm[0]) / self.moes_norm[1]
                    if self.ear_main == -1000:
                        self.ear_main = ear
                        self.mar_main = mar
                        self.puc_main = puc
                        self.moe_main = moe
                    else:
                        self.ear_main = self.ear_main * self.decay + (1 - self.decay) * ear
                        self.mar_main = self.mar_main * self.decay + (1 - self.decay) * mar
                        self.puc_main = self.puc_main * self.decay + (1 - self.decay) * puc
                        self.moe_main = self.moe_main * self.decay + (1 - self.decay) * moe
                else:
                    self.ear_main = -1000
                    self.mar_main = -1000
                    self.puc_main = -1000
                    self.moe_main = -1000

                if len(self.input_data) == 20:
                    self.input_data.pop(0)
                self.input_data.append([self.ear_main, self.mar_main, self.puc_main, self.moe_main])

                self.frame_before_run += 1
                if self.frame_before_run >= 15 and len(self.input_data) == 20:
                    self.frame_before_run = 0

                with open(csv_file_name, 'a') as f:
                    writer = csv.DictWriter(f, header)
                    writer.writerow(log)

                if cv2.waitKey(self.frame_rate_val) & 0xFF == ord('s'):
                    break
            else:
                break

        cap.release()
        cv2.destroyAllWindows()


def parseArg():
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('--user', help='Select user running the vehicle', default="anirban",
                        choices=["anirban", "sugandh"])
    parser.add_argument('--ext', help='Extension of the video file', default=".mp4",
                        choices=[".mp4", ".avi"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # args = parseArg()
    video_annotator = VideoAnnotation()
    # file_name = args.user + args.ext
    video_annotator.takelog(file_name='/home/argha/Documents/github/head_pose_estimation_mmWave/driving_dataset/dataset/20220828_17_15_04.mp4')
