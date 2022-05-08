Self Driving Car in 3D game world using Neural Network
Group 4
Andrew Abbott
Jinal Patel
Kapil Sharma
Gopesh Thakur

Objective:
This project is an attempt to create a self driving vehicle through AI implemented using Googlenet iinception version 3 model architecture in a 3D Car emulator game as world. This work is still in progress and currently self driving is not achieved completely and need more fine tuning of the model and collecting more training data.

**Use Latest Directory**

##To use this follow below instructions:
1. Create Dataset - This file "" has been used to create dataset. Please open GTA 5 in top left corner of window with 800*600 resolution.
2. Train  Model - File "" run this program to train model and set number of created dataset files befor executing this train model code.
3. Test Model - File "" will be used to test and send keystrokes for vehicle steering based on the learning completed by model in step 2. Please open GTA 5 in top left corner of window with 800*600 resolution.

##Description
Using opencv to grab image Preprocessing of image
Created white and yellow filter to display on images
Used canny images to show edges
Used gaussian blur to blur the noise
Draw hough lines
Training data has two parts -

preprocessed image
keyboard strokes performed during gameplay This preprocessed image is recorded with keyboard strokes in training data.
Train the model - We are using googlenet inception v3 architecure model and feeding the train data. Fit the model.

Testing the model -

Creating preprocessed image
Predicting the output
Use predicted output to provide keystroke to game.
