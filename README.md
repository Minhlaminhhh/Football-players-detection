# About this Project

This project aims to develop an AI Pipeline that can detect players and their jersey number within a video. Particularly, There are two main componenents within the pipeline: An object detection model for detecting players in the field and an image clasisfication model for classifying the jersey number.

## Object Detection Model
This is the first model in the pipeline. It takes a single frame within a video as the input and detecting the position of the players. Following is the result of the object detector results

*I also attempted to train the model on other objects that appear in the scene, such as the ball and boundary markers, but the model was only able to reliably detect players, with an accuracy of around 95%.*
<img width="2400" height="1200" alt="results" src="https://github.com/user-attachments/assets/d32f6e51-d6c0-4e7f-ac0b-9ee7b308cf89" />

## Classification Model
After detecting the position of each players, the second model, which is a classifier, will predict the jersey number of the corresponding players. It takes the crop image of the players as the input.

In general, the first model outperformed with 85% accuracy

*I made the model focus more on the jersey numbers by cropping out 20% from the bottom and 5% from the top of each extracted frame when separating the labels from the video for training. In the full pipeline, I also applied the same cropping ratio before sending the image to the model for prediction.*
Results of the first model checkpoint using transfer learning setup with convnext_tiny

Val Accuracy 
<img width="1838" height="870" alt="val-acc" src="https://github.com/user-attachments/assets/be727960-a4fb-43c8-8229-495c625e7453" />
After that, I further fine-tuned the model using a dataset that also included partial-visible samples. 

*Between reducing samples (adjusting the data distribution) and reducing weights (adjusting the loss function or gradient impact), I chose to reduce the partial samples by setting a 3–7 sampling ratio so the model would place less emphasis on partial cases. I also used a smaller learning rate to let the model gradually adapt to the more challenging patterns. However, the results were : the VIS accuracy did not even reach 80%, and the PAR accuracy was only around 40%.*

Val Accuracy 
<img width="1852" height="891" alt="val-acc finetune" src="https://github.com/user-attachments/assets/ac738c3e-386c-4063-8f98-0067b6b4a654" />

# How to use
I created a simple interface with a top button for running live screen detection and a second button below it for running detection on a selected video.
<img width="393" height="208" alt="image" src="https://github.com/user-attachments/assets/7ba8a624-a94f-48ea-8393-da745711e035" />

## An example of a frame in the output videos:
<img width="2624" height="927" alt="test" src="https://github.com/user-attachments/assets/ee9798d6-fa40-4be3-a049-2bd9e330dd18" />
# Project Structure

```text
football_Prj
├── app
│   ├── app_detect_and_cls
│   │   ├── app_detect_and_cls.exe
│   │   └── _internal (app data)
│   ├── models
│   │   ├── best_cls.pt
│   │   └── best_yolo.pt
│   └── app_detect_and_cls.py
├── cls (image_classification)
│   ├── dataset_for_cls
│   │   ├── partial
│   │   │   ├── train
│   │   │   └── val
│   │   ├── visible
│   │   │   ├── train
│   │   │   └── val
│   ├── Log
│   │   ├── Log_finetune_withpartial.txt
│   │   └── Log_train_visible.txt
│   ├── cls_kaggle_finetune.ipynb
│   ├── cls_kaggle_train.ipynb
│   ├── detect_label_for_cls command.txt
│   └── detect_label_for_cls.py
├── yolo (object_detection)
│   ├── dataset_for_yolo
│   │   ├── images
│   │   │   ├── train
│   │   │   └── val
│   │   ├── labels
│   │   │   ├── train
│   │   │   └── val
│   │   ├── classes.txt
│   │   └── data.yaml
│   ├── detect_label_for_yolo.py
│   └── train_yolo.py
└── source video
```
# Link
Link source video and detected dataset

Link models (model weight) https://drive.google.com/drive/folders/1-CRu_91iQS1QRhykcRoPqrOTm1XTnjQw?usp=drive_link

Link app (has been packed) https://drive.google.com/drive/folders/1-CRu_91iQS1QRhykcRoPqrOTm1XTnjQw?usp=drive_link
