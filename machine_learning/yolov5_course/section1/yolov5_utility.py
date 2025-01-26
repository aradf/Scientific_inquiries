import os
import torch
from IPython.display import Image

print("Libraries are successfully loaded \U0001F44C")

if( torch.cuda.is_available() ):
    print( torch.cuda.is_available() )
    print( torch.cuda.device_count() )
    print( torch.cuda.current_device() )
    print( torch.cuda.get_device_name() )
else:
    print("There is no Nvidia GPU available ...")    

print( "Current active directory is: ")
print( os.getcwd() )

d = os.path.join( os.getcwd(), "yolov5")
print(" The directory to be activatd is: ")
print(d)

os.chdir(d)
print( "Current active directory is: ")
print( os.getcwd() )

os.system("python detect.py --help")

os.system("python detect.py --source data\images\image_to_test_section1.jpg --conf-thres 0.5 --save-txt --line-thickness 4")

os.system("python detect.py --source data\videos\video_to_test_section1.mp4 --conf-thres 0.5 --save-txt --line-thickness 4")

os.system("python detect.py --source 0 --conf-thres 0.5 --save-txt --line-thickness 4")


Image(filename="runs/detect/exp2/image_to_test_section1.jpg")

print("Got this far ...")

      

