# AI-Face-Mask-Detector
AI that can analyze face images and detect whether a person is wearing a face mask or not, as well as the type of mask that is being worn (No mask, Cloth mask, Surgical mask or N95 mask)

## FaceMaskCNN.py
FaceMaskCNN class that is used to create a FaceMaskCNN object. 

## CNN.py
Script that uses a FaceMaskCNN object to train and test a convolutional neural network. It also
generates a confusion matrix and other data such as accuracy, recall, precision, and F-score.  

## CNNSk.py
Script that uses a FaceMaskCNN object to train and test a convolutional neural network. It also
generates a confusion matrix and other data such as accuracy, recall, precision, and F-score. This one uses Skorch features. 

## MeanAndStd.py
Calculate the mean and standard variation that we use in our normalization pattern.

## SingleImage.py
This is the application mode of our CNN. It takes a single image and predicts the type of mask
that is found in it.

## References.txt
Text file containing the references that we built our dataset with. 

# INSTRUCTIONS
## a) Training
1) create conda environment 
2) install all necessary plugins (pytorch, torchvision, matplotlib, pandas, skorch, sklearn)
3) run following command in the command prompt: python CNNSk.py

## b) Application 
1) create conda environment 
2) install all necessary plugins (pytorch, torchvision, matplotlib, pandas, skorch, sklearn)
3) run following command in the command prompt: python SingleImage.py

