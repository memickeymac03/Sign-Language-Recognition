<h1>Sign Language Recognition System (ASL)</h1>
This project aims to develop a Sign Language Recognition System using computer vision techniques. The system captures hand gestures through a camera and predicts the corresponding American Sign Language (ASL) letters. The predicted letters are then appended to form words or sentences, which are displayed on the screen in real-time. Additionally, the system includes a simple text-to-speech module for enabling two-way communication.

<h3>Files Included:</h3>

<h4>1. app.py</h4>
This Python script implements the sign language recognition system. It utilizes the Mediapipe library for hand tracking and keypoint extraction. The captured keypoints are processed and classified using pre-trained models for keypoint classification and point history classification.

<h4>2. keypoint_classification_EN.ipynb</h4>
This script is responsible for training the keypoint classification model. It loads the dataset containing keypoint coordinates and corresponding labels, splits the data into training and testing sets, and trains a neural network model using TensorFlow. After training, the model is evaluated, and the trained model is saved for inference.

<h3>Usage:</h3>
Ensure all required dependencies are installed (mediapipe, opencv, numpy, tensorflow, scikit-learn, pandas, seaborn, matplotlib).
Run webapp.py script with appropriate configurations and ensure a camera is connected.
Perform hand gestures in front of the camera, and the system will predict the corresponding ASL letters. These predictions will be displayed on the screen in real-time.
Optionally, utilize the text-to-speech module to enable two-way communication by enabling the relevant functionality in the code.

<h3>Features:</h3>
Real-time hand gesture recognition using a camera.
Predicts ASL letters based on hand gestures.
Appends predicted letters to form words or sentences.
Displays the recognized words/sentences on the screen.
Optional text-to-speech module for two-way communication.

<h3>Note:</h3>
This system currently supports recognition of American Sign Language (ASL) gestures.
The train.py script can be extended to retrain or fine-tune the models with additional data for improved performance.
Additional optimizations such as model quantization have been applied to improve inference efficiency, especially for deployment on resource-constrained platforms.
<br></br>
This project was inspired from https://github.com/kinivi/hand-gesture-recognition-mediapipe.
It has all the functionalities of the above mentioned project but I have trained extra gestures from A-Z and two more gestures to denote backspace and space to form sentences.
