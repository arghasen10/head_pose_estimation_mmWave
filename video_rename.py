import os
from datetime import datetime, timedelta

filename = '/home/argha/Documents/nexardata/processed/anirban/final_processed/20220808_110407.mp4'
datetime_obj = datetime.strptime(filename.split('/')[-1].split('.')[0], "%Y%m%d_%H%M%S") \
               + timedelta(hours=5, minutes=29, seconds=55)

os.rename(filename, filename.split('/')[0] + datetime.strftime(datetime_obj, "%Y%m%d_%H%M%S") + ".mp4")

# datetime_obj = datetime.strptime(f.split('.')[0].split('B')[0],
# "%Y%m%d_%H%M%S") + timedelta(hours=5, minutes=29, seconds=55) filepath =
# '/home/argha/Documents/nexardata/processed/sugandh/' # filepath += datetime_obj.strftime("%Y%m%d_%H%M%S") + ".mp4"
# os.rename(filepath+f, filepath) os.rename(filepath + f, filepath + "final_processed/" + datetime_obj.strftime(
# "%Y%m%d_%H%M%S") + ".mp4")
# print(datetime_obj)
