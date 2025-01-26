# 
# https://roboflow.ai
# 
# https://wandb.ai/fa-arad-terrapower


!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="S59AYDEbt3HUewaEprFa")
project = rf.workspace("faramarz-arad").project("bccd-yolov5-uodf5")
version = project.version(1)
dataset = version.download("yolov5")



!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="S59AYDEbt3HUewaEprFa")
project = rf.workspace("faramarz-arad").project("orange_bicyclehelmet_yolov5")
version = project.version(1)
dataset = version.download("yolov5")
                
!pip install roboflow

from roboflow import Roboflow
rf = Roboflow(api_key="S59AYDEbt3HUewaEprFa")
project = rf.workspace("faramarz-arad").project("trafficsign_yolov5-ryga4")
version = project.version(1)
dataset = version.download("yolov5")
print('Done with Download Dataset')
                                
# train yolov5 models locally
# activate the yolo5_venv, change directory to where yolo5 is installed.
# run python train.py command 
source yolo5_venv/bin/activate
cd /home/montecarlo/Desktop/scientific_computing/yolov5_course/section1/yolov5
python train.py --data ../../section3/custom_dataset/BCCD-YOLOv5/data.yaml --weights yolov5s.pt

python detect.py --help
python detect.py --source data/images/image_to_test_section1.jpg --conf-thres 0.5 --save-txt --line-thickness 4
python detect.py --source data/videos/video_to_test_section1.mp4 --conf-thres 0.5 --save-txt --line-thickness 4
python detect.py --source 0 --conf-thres 0.5 --save-txt --line-thickness 4
python detect.py --source data/video/video_to_test_section1.mp4 --conf-thres 0.5 --save-txt --line-thickness 4 --device 0
python detect.py --source ../../section3/custom_dataset/BCCD-YOLOv5/test/images/ --conf-thres 0.5 --save-txt --line-thickness 4 --device 0
python detect.py --source ../../section6/elephants.mp4 --weight ../../section6/weights/custom_dataset/best.pt --conf-thre 0.5 --save-txt --line-thickness 4 --device 0
python detect.py --source ../../section3/custom_dataset/video_forest_yolov5/forest-road.mp4 --weight runs/train/exp2/weights/best.pt --conf-thre 0.5 --line-thickness 4 --device 0
python detect.py --source ../../section3/custom_dataset/video_forest_yolov5/forest-road.mp4 --weight runs/train/exp2/weights/best.onnx --conf-thre 0.5 --line-thickness 4 --device cpu

python train.py --data ../../section3/custom_dataset/BCCD-YOLOv5/data.yaml --weights yolov5s.pt --device 1 --epochs 100 --batch-size 2 --workers 0
python train.py --data ../../section3/custom_dataset/BCCD-YOLOv5/data.yaml --weights yolov5s.pt --device cpu --epochs 100 --batch-size 2 --workers 0
python train.py --data ../../section3/custom_dataset/BCCD-YOLOv5/data.yaml --weights yolov5s.pt --device cpu --epochs 100 --batch-size 2 --workers 0 --cash ram
python train.py --data ../../section3/custom_dataset/BCCD-YOLOv5/data.yaml --weights yolov5s.pt --device cpu --epochs 100 --batch-size 2 --workers 0 --cash disk
python train.py --data ../../section3/custom_dataset/video_forest_yolov5/data.yaml --weights yolov5s.pt --device 0 --epochs 100 --batch-size 2 --workers 0

python val.py --data ../../section3/custom_dataset/BCCD-YOLOv5/data.yaml --weight runs/train/exp/weights/best.pt --batch-size 2 --workers 0 --device 0 --task val
python val.py --data ../../section3/custom_dataset/BCCD-YOLOv5/data.yaml --weight runs/train/exp/weights/best.pt --batch-size 2 --workers 0 --device 0 --task test


# convert best.pt to onnx format
python export.py --weights runs/train/exp3/weights/best.pt --include onnx


# install jupyter notebook
pip3 install jupyter
jupyter notebook

# Train YOLO v5 in Colaboratory Link to Google Drive:
https://drive.google.com/


from google.colab import drive
drive.mount('/content/drive')

!zip -r my_directory.zip /content/my_directory

from google.colab import files
files.download('my_directory.zip')


# nvidia information.
nvidia-smi





