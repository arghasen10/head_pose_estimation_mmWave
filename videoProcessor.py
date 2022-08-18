import os
from headPoseEstimation import VideoAnnotation


anirban_file_path = '/home/argha/Documents/nexardata/processed/anirban/final_processed/'
anirban_vids = os.listdir(anirban_file_path)

sugandh_file_path = '/home/argha/Documents/nexardata/processed/sugandh/final_processed/'
sugandh_vids = os.listdir(sugandh_file_path)

video_annotator = VideoAnnotation(y_min=-4, y_max=9, x_min=-6, x_max=6, frame_rate_val=1)
for each_file in anirban_vids:
    path = anirban_file_path+each_file
    video_annotator.takelog(path)

video_annotator = VideoAnnotation(y_min=-2, y_max=8, x_min=-4, x_max=4, frame_rate_val=1)
for each_file in sugandh_vids:
    path = sugandh_file_path + each_file
    video_annotator.takelog(path)
