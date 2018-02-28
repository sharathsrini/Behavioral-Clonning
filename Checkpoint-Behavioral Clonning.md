## Checkpoint : Behavioral Clonning

###### 23-02-2018
1. Online Classes : Keras and Transfer Learning.
2. Some Ressources :
	https://keras.io/layers/advanced-activations/
	https://devblogs.nvidia.com/deep-learning-self-driving-cars/
3.Read about Collabroratory Notebook
4.ELU - activation Function.


###### 26-02-2018
1. How do I choose an optimizer for my tensorflow model? https://www.quora.com/How-do-I-choose-an-optimizer-for-my-tensorflow-model

2. What are good initial weights in a neural network? https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network

3.Transfer Learning: How to choose a pre-trained model? https://www.analyticsvidhya.com/blog/2017/06/transfer-learning-the-art-of-fine-tuning-a-pre-trained-model/

Activation functions and it’s types-Which is better? https://towardsdatascience.com/activation-functions-and-its-types-which-is-better-a9a5310cc8f

###### 27-02-2018
1. Generator Class in the Pipeline
2. Keras - Checkpoints and Callbacks
3. Image Augmentation 
4. Matplotlib For Drawing Insights onj the Train and Validation Dataset.
5. Generated the model.h5

###### 28-02-2018
1. Localization + Classification.
2. Sliding window for detecting objects all over the image.(Size of the ppatch depends on the size of the image.).
3. Activation Map for detecting objects and classes.
4. YOLO  : Choose the ground truth of each classes properly.
5. Centre block or the grid cell that lies on centre of the object, is responsible for detecting the object.
6. YOLO v1 : 1000 classes.
7. It treats the image as a single regression problem.
8. Five outputs : X, Y, W, H, Confidence.
9. Confidence = Pr(Object) * IOU.
10.IOU : Intersection over Union.
11.IOU tends to be within 0 and 1. It depends ont he overlapping of the images.
12.Non-max supression algorithm 
13.https://medium.com/diaryofawannapreneur/yolo-you-only-look-once-for-object-detection-explained-6f80ea7aaa1e
14.Region-based Convolutional Neural Networks
15.Mini-YOLO : https://github.com/xslittlegrass/CarND-Vehicle-Detection/blob/master/vehicle%20detection.ipynb
16.Objects with the Same center point on the image, Only object can be detected. Changing the size of the sliding window would help to certain level. 
