# 🎱 Computer Vision Project: 8-Ball Pool Table Analysis
This repository contains the implementation for Task 1 of the Computer Vision course project. The primary goal is to develop an image processing pipeline using OpenCV to detect and analyze balls on an 8-ball pool table.

-----
# 🚀 Project Overview
The system processes images of 8-ball pool tables to extract spatial and categorical data. The pipeline transforms a standard perspective view into a processed top-down view while identifying every ball present.

Key Functionalities
* Ball Detection: Locates all balls on the table and provides bounding boxes.
* Ball Identification: Determines the number of each ball based on its color.
  * Note: The white cue ball is considered number 0.
* Perspective Transformation: Generates a top-view image of the table.
* Automated Counting: Outputs the total number of balls detected.
------

# 🛠️ Technical Stack
* Language: Python 
* Core Libraries:
  * OpenCV (Main computer vision tasks) 
  * NumPy (Numerical processing) 
  * Matplotlib (Visualization)



