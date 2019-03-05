## Project: DeepLearning - Follow Me
### Project 4 submission by Lewis Siempelkamp for Udacity's Robotics NanoDegree Term 1 
### March 4th 2019

---
[//]: # (Image References)

[image1]: ./images/1.jpg
[image2]: ./images/diagram.jpg
[image3]: ./images/training_curves.jpg
[image4]: ./images/Predictions.JPG


![Project][image1]


# Requirements for a Passing Submission:
1. Provide a write-up / README document including all rubric items addressed in a clear and concise manner. The document can be submitted either in either Markdown or a PDF format.
2. Update model_training.ipynb to create a Neural Network that generates a model to perform pixelwise segmentation from a generated training dataset using tuned Hyper Parameters  
3. Submit the trained model in the correct format that achieves a score of at least 40%

## [Rubric](https://review.udacity.com/#!/rubrics/1155/view) Points
 
---

This project is an exercise in training a Fully Convolutional Deep Neural Network (FCN) to perform pixelwise segmentation on each frame of a video stream provided by a Quad Copter to identify, locate, and track a distinct individual amongst a crowd of people in a virtual city.

---

An FCN is a type of Deep Neural Network. 

A Neural Network (NN) a system of linear equations and non-linear activation functions (f_activation()) that take a tensor of inputs (X) and return a tensor of predicted outputs (Y_hat) dependent on a tensor of weights and biases within the system.

predicted_output Y_hat = f_activation(W*X + B)

where each element of the inputs and outputs (X[i], Y_hat[j]) can be thought of as a node in the system with each input node connected to each output node by that edge's corresponding weight[Wij]. 
Since each node from the input layer is connected to each node in the output layer - this system of nodes can be thought of as Fully Connected.
 
If the outputs are known (Y) for a given input(X), then error between the predicted output from the NN can be compared against the actual output (the ground truth) and this error which is a function of (X,W,B) can then be backpropogated through the system to adjust each of the weights and biases (W[ij], B[j] ). 
By iterating through this process many times the weight tensor will eventually be tuned to the point that the error between the calculated output and the ground truth output becomes small.
With an adequately tuned weight tensor the NN will be able to return useful outputs from entirely new inputs.

A Deep Neural Network (DNN) is a NN with one or more hidden layers (l_n) with one or more nodes between the input tensor (X) and the output tensor (Y_hat) with each layer's output feeding into the next layer's input.

Y_hat = f_activation(L_n), 
where L_n = f_activation(W_n*L_(n-1) + B_n) and,
Hidden Layer1 L_1 = f_activation(W_1*X + B_1)

By introducing additional layers the NN is capable of performing intermdediate processing steps that may be necessary to return more meaningful outputs than a single layer NN.
The added benefits of multi-layer DNNs come at the cost of increasing their complexity and the total number of tunable parameters.

A Convolutional Neural Network (CNN, convnet) deviates from the Fully Connected architecture seen in NNs/DNNs by instead of connecting each node between an input and output layer a smaller NN is creaated that only looks at a patch of the input layer at a time but is swept across the entire Input layer forming an output tensor with new dimensions. 
The example application for CNNs explored in this project and the preceding lessons was that of a network that takes as its input tensor a colour image with R,G, and B channels such that the input X has dimensions HxWxRGB.
A CNN would take this input and sweep a filter of size (KxKxF) over the HxW of the image and output a new tensor with dimensions (K<H=<H,K<=W'<W,F=>RGB), smaller in H' and W' than the original input but with a greater depth, F, than the 3 RGB channels, where F represents the number of filter channels created by this layer.
Because of how this filter sweeps over the image rather than taking it all in at once the output is able to view small features within the image input that can exist anywhere on the HXW plane.
The depth F of the filter represents the number of new feature maps generated from the input with each new filter featuremap potentially captyuring specific feature information such as lines or curves within a patch of the image as it sweeps over.
By combining additional convolutional layers to the first, with each new layer sweeping over the output filter of the previous, the network can potentially extract complex higher level features from within an image for example the 2nd convolutional layer might combine the lines and curves extracted by the first layer and combine them into shapes, and the next layer combine those shapes into objects and so forth.

The CNN is completed when the last convulutional layer is Fully Connected to the output which when tuned is capable of classifying a large input dataset based on internal potentially repeating features within that data into a smaller range of potential outputs.

The CNN can be iteratively tuned in the same way that the NN/DNN can be tuned by systematically changing the weight tensors based on the output error.

Great! So with a CNN one can identify what is in a dataset. In the case of an RGB image it can classify what object lies in that image.

A Fully Convolutional Neural Network (FCN) goes one step further an allows us to determine where an object lies within an image.
It does this by taking the output of the CNN and performing an equal number of Deconvolution Steps (Transposed Convolutions) such that the final output is the same shape as the input with the spational information from the input preserved.
In addition to taking only the inputs from the final output of the CNN the FCN can implement Skip Connections where information from previous layers is passed directly up to subsequent deconvolution layers, skipping intermediate layers entirely, allowing for the processing of higher resolution information that would otherwise be lost without these connections.



An FCN can be deoncstructed into two major blocks: The Encoder and the Decoder. 
The Encoder block is comprised of the Convolutional layers and its goal is to extract features from the data.
The Decoder block is comprised of the Deconvolutional layers, upsampling these features back to the original image size allowing for classification at the pixel level.

In a CNN the Encoder block is output to a fully connected output layer. The output layer becomes flattened, however, and spatial information is lost. 
To preserve spatial information from the Encoder block, and to allow the FCN be compatible with any size image, the FCN replaces the fully connected layer with a 1x1 Convolution Layer. A 1x1 convolution sweeps across the output of the final Encoder Layer 
Within the FCN model there exist several configurable parameters. 
    The number of layers in the Encoder Block and Decoder block can be increased or decreased allowing for an increased/decreased hierarchy of feature extraction with increasing complexity the more layers are added.
    Each convolution layer has a depth associated with it, its number of Filters/Kernels. This depth loosely correlates to the number of categories that can be identified by each layer, with each subsequent layer in the Encoder having increasing depth and each in the Decoder block subsequently decreasing in depth.
    The non-linear activation function can be any of several functions including ReLu, Sigmoid, Step and others.
    The Bilinear upsampling rate (the rate at which the Decoder blocks upsample from previous layers) can be adjusted.
    
For the purpose of this project an FCN was conctructed to analyze an incoming stream of RGB images to perform pixel-wise semantic-segmenation of the image to classify what objects it is seeing and where those objects lie in the image.

The FCN constructed is comprised of the following layers and connections:

Input Layer [3 Featuremaps] -> Encoder Layer 1 [32 Filters] -> Encoder Layer 2 [64 Filters] -> 1X1 Convolution Layer [128 Filters] -> Decoder Layer 1 [64 Filters] -> Decoder Layer 2 [32 Filters] -> Output Layer [3 Featuremaps]
    
With Skip Connections from: 
    Encoder Layer 1 -> Decoder Layer 1
    Input Layer -> Decoder Layer 2

![Diagram][image2]
    
The FCN network model, Encoder Layers, Decoder Layers, and Fully Connected Layers can be seen implemented in code in [model_training.ipynm](/code/).


To train the model, images were colleceted from the Unity Quad-Simulation Environment by recording the target actor and background actors in a number of different setups and environments in an attempt to populate the batch of training data with a large variety of images to work with.
~6000 images were collected and the ground truth masks corresponding to the Target Actor, the Background Actors, and Background Environment of those images were generated using [preprocess_ims.py](/code/preprocess_ims.py).

The model was then trained using the raw RGB images as inputs with the ground truth masks to determine the error/Loss from the predicted output.     

Several Tunable Hyperparameters exist that will impact how effectively and how quickly the model can be trained from a given data set:

Learning Rate: 0.01
    Learning Rate is the coefficient that scales the delta weight change for each weight element during the backpropogation phase of a run. Higher learning rates result in initially faster learning but cause longer training required overall due to larger misteps in the wrong direction.
Batch Size: 128
    Batch Size is the number of training samples that get passed through the network in each pass. The larger the batch size, the larger the memory allocation for each run with the upper limit being determined by your hardware. 
Number of Epochs: ~40
    The number of times the entire training set is passed through the trainer. It is common to see hugh gains in early epochs with later epochs providing only marginal gains. (depending on the learning rate)
Number of Steps per Epoch: ~Total Sample Count/Batch Size
    Since hardware limits prevent an entire trianing be passed through the network at once, the trainer must step through the samples in increments of the batch size.

The below figure shows how the loss initially reduces quickly in the first 20 epochs but only small reductions in loss are seen afterwards:
![training curves][image3]

Although once could conceivably have a perfect dataset and tuned hyper parameters that can train a model in one pass, for this project the model was trained using several passes with different hyper parameters and the dataset being fed updates regularly to help fill in the learning deficincies during trianing.    

Using the above descriped FCN, Hyper Parameters, and training set, a [model](/data/weights/model_weightsA8) was trained until it reached the target IOU accuracy of 40%.
The model successfuly differentiates between the target actor, background actors, and background environment with a final score of 40.1%

The model can be run in real time by running the follower.py <model-weights script in conjunction with the Unity Quad Sim.

Predictions from the model can be seen below demonstrating that the model succesfully differentiates between the target and background actors and background environment.

![Predictions][image4]
