import argparse

from moviepy.editor import *
from natsort import natsorted


def parseArg():
    parser = argparse.ArgumentParser(description='Run Experiment')
    parser.add_argument('--filepath', help='Directory where all files are present',
                        default="../../driver-head-pose/videodata/anirban/",
                        choices=["../../driver-head-pose/videodata/anirban/",
                                 "../../driver-head-pose/videodata/sugandh/"])

    parser.add_argument('--ext', help='Extension of the saved video file', default=".mp4",
                        choices=[".mp4", ".avi"])
    parser.add_argument('--user', help='Select user running the vehicle', default="anirban1",
                        choices=["anirban1", "sugandh1"])
    args = parser.parse_args()
    return args


def merg_videos(filepath, savingname, ext):
    L = []
    for root, dirs, files in os.walk(filepath):
        files = natsorted(files)
        for file in files:
            if ext == '.mp4':
                if 'B.mp4' in file:
                    filePath = os.path.join(root, file)
                    video = VideoFileClip(filePath)
                    L.append(video)
            elif '.avi' in file:
                filePath = os.path.join(root, file)
                video = VideoFileClip(filePath)
                L.append(video)

    sav_name = savingname
    final_clip = concatenate_videoclips(L)
    final_clip.to_videofile(sav_name, fps=30, remove_temp=False,  codec="libx264")


if __name__ == "__main__":
    args = parseArg()
    filepath = args.filepath
    sav_file_name = args.user + args.ext
    merg_videos(filepath, sav_file_name, args.ext)
