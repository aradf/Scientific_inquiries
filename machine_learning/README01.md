# Install virtual envirnoment
apt install python3.12-venv
python3 -m venv ml-venv

# How to activate the ml-env
source ml-venv/bin/activate

# How to launch labelImg for YOLO labelng:
label-studio start

# Install
apt-get -y install python3-pip
pip3 install --upgrade pip
pip3 install numpy
pip3 install pandas
pip3 install scikit-learn
pip3 install statsmodels
pip3 install matplotlib
pip3 install tensorflow
pip3 install tf2onnx
pip3 install onnxruntime
pip3 install opencv-python

# label sutdio installation
pip3 install -U label-studio
label-studio start

# label ffmpeg
sudo apt-get install ffmpeg
ffmpeg -version
ffmpeg -i <filename> -vf fps=4 image-%d.jpeg

# install Open Images Dataset v4 (OIDv4 )
https://storage.googleapis.com/openimages/web/visualizer/index.html
git clone https://github.com/EscVM/OIDv4_ToolKit.git
pip3 install pandas
pip3 install numpy
pip3 install awscli
pip3 install urllib3
pip3 install tqdm
pip3 install opencv-python

python3 main.py downloader --classes Car Bicycle_wheel Bus --type_csv train --multiclasses 1 --limit 9
python3 main.py visualizer

# install darknet
cd ~
git clone https://github.com/AlexeyAB/darknet.git
cd darknet
make 
./darknet 
# cd /home/montecarlo/darknet
./darknet detector test cfg/coco.data cfg/yolov3.cfg weights/yolov3.weights data/test-image.jpg


# install yolo5_env
apt install python3.12-venv
python3.9 -m venv yolo5_venv

# How to activate the yolo5_venv
source yolo5_venv/bin/activate

# install yolov5
# extra training for yolov5: https://www.youtube.com/watch?v=mRhQmRm_egc
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip3 install -r requirements.txt
python detect.py --help

pip3 install ipython

#install fifty one env nad Jupyter.
python3.9 -m venv fo-venv
source fo-venv/bin/activate

pip3 install --upgrade pip setuptools wheel build
pip3 install fiftyone
fiftyone --help

pip3 install jupyter
jupyter notebook

# install Weight and Biased (wandb )
source yolo5_venv/bin/activate
pip3 install wandb
wandb login
# use the www.wandb.ai to identify the api number
wandb login

# disable the wandb
wandb disabled

# install tensorboard
pip3 install tensorboard
tensorboard --logdir runs/train/exp












