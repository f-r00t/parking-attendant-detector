# Parking attendant detector

This repository contains an AI system that can detect people and parking attendants from live video data, video recordings and still images.

It uses UltraLytics YOLOv11 as the model, that has been trained on a dataset contained within the data/ folder in this repository.


# Usage

## Requirements
- python3

This software suite requires no special hardware, other than those required by the dependancies found in requirements.txt, but a cuda capable GPU is recommended.

## Installation
First edit requirements.txt to reflect a torch version suitable for your system, see link for more details: https://pytorch.org/get-started/locally/
Then edit the torch row of the requirements.txt file with whatever version fits best for your system and then run the following in your terminal:
```
pip install -r requirements.txt
```
After installation is complete, start Jupyter notebook by running:
```
jupyter notebook
```
### Optionally
Go to notebooks and open **1-train.ipynb** and run the file to train the model. However, a pretrained model is provided in **models/yolov11n_attendant.pt** but running the first code block of the notebook may be required in order to download dependancies.

## Running the software

The program is configured to use the first video source on your system as an input device for detection, if this is adequate for your usage simply run:
```
python3 src/video_stream.py
```
To start detecting in real time. The system also allows running on video and image data, by changing this line in **src/video_stream.py**:
```
---
70    run_video_stream(source=0)
+++
70    run_video_stream(source="video.mp4")
```
It's also possible to save the output to a new video file, see the file for details.
## Testing on a single image

There is also a designated script for running the system on a single image. It's configured to run on the provided image in **data/sample.jpg**.

To test it, simply run:
```
python3 src/test_on_image.py
```
