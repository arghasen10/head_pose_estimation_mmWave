import os
from headPoseEstimation import VideoAnnotation

video_annotator = VideoAnnotation()
final_file_path = '/home/argha/Documents/nexardata/processed/final_processed/'
file_lists = os.listdir(final_file_path)

for each_file in file_lists:
    path = final_file_path+each_file
    video_annotator.process(path)

    # cap = cv2.VideoCapture(path)
    # fps = cap.get(cv2.CAP_PROP_FPS)
    # starttime = datetime.strptime(each_file.split('.')[0], "%Y%m%d_%H%M%S")
    # frame_count = 0
    # while cap.isOpened():
    #     ret, frame = cap.read()
    #     if ret:
    #         frame_count += 1
    #
    #         seconds = round(frame_count / fps)
    #         video_time = starttime + timedelta(seconds=seconds)
    #         cv2.putText(frame, str(video_time), (10, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, 2)
    #         cv2.imshow("Frame", frame)
    #
    #         if cv2.waitKey(25) & 0xFF == ord('s'):
    #             break
    #     else:
    #         break

# import os
# from datetime import datetime, timedelta
#
# all_files = os.listdir('/home/argha/Documents/nexardata/processed/sugandh/')
#
# for f in all_files: if 'B.mp4' in f: datetime_obj = datetime.strptime(f.split('.')[0].split('B')[0],
# "%Y%m%d_%H%M%S") + timedelta(hours=5, minutes=29, seconds=55) filepath =
# '/home/argha/Documents/nexardata/processed/sugandh/' # filepath += datetime_obj.strftime("%Y%m%d_%H%M%S") + ".mp4"
# os.rename(filepath+f, filepath) os.rename(filepath + f, filepath + "final_processed/" + datetime_obj.strftime(
# "%Y%m%d_%H%M%S") + ".mp4")
