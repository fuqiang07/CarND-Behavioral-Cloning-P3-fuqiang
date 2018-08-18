# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[model]: ./output_images/model.png "Model"
[architecture]: ./output_images/architecture.png "architecture"
[distribution]: ./output_images/distribution.png "distribution"
[center]: ./output_images/center.jpg "center"
[recover1]: ./output_images/recover1.jpg "recover1"
[recover2]: ./output_images/recover2.jpg "recover2"
[recover3]: ./output_images/recover3.jpg "recover3"
[recover4]: ./output_images/recover4.jpg "recover4"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_fuqiang.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I implemented the [NVIDIA Model](https://devblogs.nvidia.com/), presented in the lecture (model.py lines 181).
The reason that I use this model is that it is effective to deal with Behavior Cloning task.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 212). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 310 - 314). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 310-314).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and counter-clockwise driving.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture leads to the file ``model.py``.

Based on the experience of the previous projects, I tried my best to collect as many data as I can (center lane driving, recovering from the left and right sides of the road, counter-clockwise driving).

Then I took the idea from [Jeremy Shannon's Github](https://github.com/jeremy-shannon/CarND-Behavioral-Cloning-Project) to uniform the data, that is, try to shrink the data with almost zero steering angle. As shown below, with this idea, the distribution in red looks more unifrom after some data cut off (in blue).
![alt text][distribution]

Afterwards, I flip each image to double the size of the training and test data, which is helpful to overcome overfitting problems.

It is lucky that the vehicle is able to drive autonomously within the lane with only one trial of my approach.

#### 2. Final Model Architecture

The final model architecture (model.py lines 181-225) consisted of a convolution neural network with the following layers and layer sizes
![alt text][model]

Here is a visualization of the architecture
![alt text][architecture]


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded 5 laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center by itself. These images show what a recovery looks like starting from the left side of the road:

![alt text][recover1]
![alt text][recover2]
![alt text][recover3]
![alt text][recover4]

Then I repeated this process on on counter-clockwise in order to get more data points.

To make the data more uniform, I cut off some data with zero steering angles. This is helpful to make the data more uniform and make the result more robust.

To augment the data sat, I also flipped images and angles thinking that this would supress overfitting. 

After the collection process, I had 22943 number of data points. I then preprocessed this data by normalization.
```python
# Normalize
model.add(Lambda(lambda x: x/255.0-0.5, input_shape=(160,320,3)))
```

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 6 by balancing the computation and the accuracy. I used an adam optimizer so that manually training the learning rate wasn't necessary.

The final result can be found as in the [video](./video_1.mp4).