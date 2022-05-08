# PythonProjectSpring2022
### Team Name: Techletes

## Objective:

This project is an attempt to create a self driving vehicle through AI implemented using Googlenet iinception version 3 model architecture in a 3D Car emulator game as world. 
This work is still in progress and currently self driving is not achieved completely and need more fine tuning of the model and collecting more training data.

* Using opencv to grab image
Preprocessing of image
* Created white and yellow filter to display on images
* Used canny images to show edges
* Used gaussian blur to blur the noise
* Draw hough lines

Training data has two parts - 
1. preprocessed image
2. keyboard strokes performed during gameplay
This preprocessed image is recorded with keyboard strokes in training data.


Train the model - 
We are using googlenet inception v3 architecure model and feeding the train data.
Fit the model.

Testing the model - 
1. Creating preprocessed image
2. Predicting the output
3. Use predicted output to provide keystroke to game.
