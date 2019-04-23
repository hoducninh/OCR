# Build an OCR system using keras which is able to correctly convert this jpeg image to text. Reference: 20'000 leagues under the sea
### Hieu Nguyen
 
## Summary

Build an OCR system using keras which is able to correctly convert this jpeg image to text. Reference: 20'000 leagues under the sea

Due to the time limit and lack of data for this test, as well as my current experience, I used an existing project titled "Handwriting Detection and Recognition with Neural Networks" (Alexander Chiu, Andrew Quintanilla, Milan Ghori, Akshay Pawar, Chinmayee Tapaskar). The link will be provided below.

Using this application, the page will upload images to the server, then output the results of the server’s analysis of the image. 

Tesseract and OpenCV will be used to detect distinct blocks of text, and we can output the text for each block separately. 

Keras will be used to load the pre-trained model in order to classify words in the image. Keras will be used in dataload.py with extensive documentation.


## How to Use
Step 1: Install the prerequisite libraries and packages for Python 3.6:
* OpenCV 
* Numpy
* Pillow
* Tesserocr
* Tesseract
* Tensorflow
* Keras
* Flask

Step 2: Go to the project directory and run the command,
`python app.py`

Step 3: Run the above command will start the Flask web server

Step 4: Upload the image file 20000-leagues-006.jpg for translation.

## Classifier Pipeline
There are two steps for this problem: Word Detection and Word Recognition/Classification.

When a user uploads the image into the website, the backend will save that image into its own local storage. With that image, the detector will grayscale, binarize it, and create a temporary copy of the image for word detection. For every word detected, the detector will crop the word out and save it for classification. After the word detection is complete, the server will remove the temporary image. 

With the cropped word images, the classifier will create a list of word results. For each word image, the classifier will reshape the image and send it through its network for prediction. Each word classification will have a list of words and their resulting confidences. The classifier will extract the highest confidence and its respective word and append it to the word list. If the confidence level is below 50%, the system will return “(N/A)” as a the word since the authors know the word is not in the dictionary. When all of words have been classified, the system will return the list of words to the web front to be displayed as a sentence of classified words. 


## Detection and Segmentation
The detector is built with Tesseract and OpenCV. When the user uploads an image, the detector will preprocess the image to create a binarized image. The detector performs binarization using OpenCV’s implementation of the Otsu method. This algorithm will search the entire image for a threshold value that minimizes the intra-variance of the image. The image will be a black and white image that Tesseract will use to find words. After finding the words, the detector crops out the detected word and saves it locally. After the detected words are cropped out, the detector removes the binarized image since it is not needed anymore. During the entire detection process, the original input image is preserved. 

## Neural Network Classification
The model consists of multiple neural network layers, with a classifier layer for output. It includes five CNN layers to extract features from the images and reduce dimensionality, which then feed into two dense layers. A final dense layer, with as many neurons as there are distinct words to classify from the authors' word dataset, is used to output the final classification. It is written in Python, using libraries such as numpy, Tensorflow, OpenCV, Tesseract and Keras.

The authors used multiple strategies for training the model. The paper the authors had based their model on described a 10,000 word classifier trained on 9 million images. Due to computational limitations, the authors could only train a 5,000 word classifier using a fraction of the images. The authors also had to train in stages, doing initial training on half of the whole dataset, then training the model again using the other half. To prioritize which words of the 10,000 to use, the author obtained a list of the 10,000 most common words according to Google, which came sorted, and used the first n, where n is how many words the author wished their model to classify.

## Python libraries
Numpy to access basic statistical functions and to use numpy arrays, and Pillow to convert the images to pixel intensity arrays. For the neural network, the authors used used Tensorflow, with Keras on top to streamline writing the code. The detector uses OpenCV and Tesseract to process images. Flask was used to write the frontend.


## References
* [AlexNet](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks)
* [A Beginner's Guide to Understanding Convolutional Neural Networks](https://adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)
* [Nvidia Digits](https://developer.nvidia.com/digits)
* [Synckey Tensorflow OCR](https://github.com/synckey/tensorflow_lstm_ctc_ocr)
* [VGG Synthetic Word Dataset](http://www.robots.ox.ac.uk/~vgg/data/text)
* [Tensorflow](https://www.tensorflow.org/)
* [OpenCV](https://opencv.org/)
* [Keras](https://keras.io/)
* [Tesseract](https://github.com/tesseract-ocr/tesseract)
* [Handwriting Detection and Recognition with Neural Networks](https://github.com/AlChiu/HandWriting-OCR-CNN-WebApp)
