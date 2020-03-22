
# ID Card Detection Kit
This repo lets you easily perform instance segmentation for id cards.

## Installation
- Manually install Miniconda (Python3) for your OS:
https://docs.conda.io/en/latest/miniconda.html

Or install Miniconda (Python3) by bash script on Linux:
```console
sudo apt update --yes
sudo apt upgrade --yes
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p ~/miniconda 
rm ~/miniconda.sh
```

- Inside the base project directory, open up a terminal/anaconda promt window and create environment:
```console
conda env create -f environment.yml
```

- After environment setup, activate environment and run the tests to see if everything is ready:
```console
conda activate cardetkit
python -m unittest
```

## Usage
- In the base project directory, open up a terminal/anaconda promt window, and activate environment:
```console
conda activate cardetkit
```

- Perform prediction for image "tests/data/idcard2.jpg" using pretrained Mask-RCNN with resnet50 backbone:
```console
python predcit.py tests/data/idcard2.jpg maskrcnn_resnet50
```
Results will be saved to "output" folder.

