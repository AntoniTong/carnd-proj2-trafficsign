# **Traffic Sign Recognition** 

### Project Summary

**In this traffic sign recognition project, I first partitioned a large set of German traffic sign data into three blocks: training, testing, and validation. The training data is used to perform backpropagation for model optimization, the testing data is used to check model accuracy during training, and the validation data is set aside to test on the trained model. Through the help of course materials and the LeNet code, I implemented a multi-layer convolutional neural network with max pulling. Model optimization result indicated 94.7% estimation accuracy and**


[//]: # (Image References)

[image1]: ./write_up_img/Training_Data.png "Data Distribution"
[image2]: ./write_up_img/Raw_Img.png "Raw Images"
[image3]: ./write_up_img/Gray_Img.png "After Processing"
[image4]: ./write_up_img/New_Img.png 
[image5]: ./write_up_img/New_Img_Test.png 
[image6]: ./write_up_img/Indepth_1.png 
[image7]: ./write_up_img/Indepth_2.png 
[image8]: ./write_up_img/New_Img_Raw.jpg
---



### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,1)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is a bar chart showing the number count foreach identifying classes in the training dataset.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I implemented two preprocess steps for the raw 32x32 images: normalization and gray scale. 

For normalization, the formulais the following 
	(images.astype('float32') / 255.) - 0.5 

For gray scale images,instead of CV2function, I used a simple blending formula to convert from 'RGB' to 'Gray' via the following

images = np.matmul(images, [0.2989 , 0.5870 , 0.1140 ] )

It is worth mentioning that after the matrix multiplication, the output data size became (32,32). It is convenient to change the data dimension to (32,32,1) using the following code:

np.reshape(images, images.shape + (1, ) )

The following figure shows the original images

![alt text][image2]

And the following figure shows the images after preprocessing

![alt text][image3]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Gray Image  							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x16 	|
| Activation RELU					|												|
| Max Pulling     	| 2x2 stride, valid padding, outputs 14x14x16 	|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x32 	|
| Activation RELU					|												|
| Max Pulling     	| 2x2 stride, valid padding, outputs 5x5x32 	|
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 5x5x64 	|
| Activation RELU					|												|
| Max Pulling     	| 1x1 stride, valid padding, outputs 5x5x64 	|
| Flatten    | outputs 800     									|
| Fully connected		| outputs 100        									|
| Activation RELU					|												|
| Fully connected		| outputs 43      									|
|						|												|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

| Variables        		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Optimizer        		| Adam  							| 
| Epochs     	| 20	|
| Learning Rate					|		0.005										|
| Batch Size     	| 128 	|
|  	|  	|
| 				|												|


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and wherein the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well-known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 94.6%
* validationset accuracy of 93.4%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
I implemented two layer CNN with output layer of 28x28x2 and 10x10x4. 

* What were some problems with the initial architecture?
The model has a problem of low accuracy after training. 


* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

I adjusted the model to be deeper (from 2 convolution layer to 3 convolution layer) and also enlarged output size to 28x28x16 and 10x10x32. And the model trains with bettery accuracy without overfitting. 

I also tried other adjustments including feeding both the Conv Layer 2 and the Conv Layer 3 into the flatten layer, as well as implementing dropouts. The improvements were limited. 

​​* Which parameters were tuned? How were they adjusted and why?
I adjusted both batch size and learning rate. 
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here an example of German traffic signs that I found on the internet:

 ![alt text][image8] 

Because the image is colored, in higher solution, and different dimension, first I resized the figure into a standard 32x32 image, as shown in the following figure. 

![alt text][image4]

The resulting figure is distorted, which could partially be the reason of low identification accuracy. A better way to preprocess the image, perhaps is to train another model to detect the region of the picture where the traffic sign is located, and cast it to a 32x32 image.
​
The following figure shows the result of traffic sign identification using the trained model. 

 ![alt text][image5]

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

For the 9 images, I found online, 7 images were predicted with 100% certainty (although, 1 of them was identified incorrectly). For the rest two images (see figures below), the softmax outputs were split between a>95% confidencefor the primary guess, and <5% confidencefor the secondaryguess.


![alt text][image6]

![alt text][image7]
