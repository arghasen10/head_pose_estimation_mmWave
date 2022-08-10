from moviepy.editor import *
from natsort import natsorted

L = []
for root, dirs, files in os.walk('../../nexardata/processed/anirban/'):
    files = natsorted(files)
    for file in files:
        if 'B.mp4' in file:
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("anirban.mp4", fps=30, remove_temp=False)


L = []
for root, dirs, files in os.walk('../../nexardata/processed/sugandh/'):
    files = natsorted(files)
    for file in files:
        if 'B.mp4' in file:
            filePath = os.path.join(root, file)
            video = VideoFileClip(filePath)
            L.append(video)

final_clip = concatenate_videoclips(L)
final_clip.to_videofile("sugandh.mp4", fps=30, remove_temp=False)