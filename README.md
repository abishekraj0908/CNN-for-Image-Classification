Multi Class Fish Image Classification
NAME OF THE CONTRIBUTOR: Abishek
BATCH: July 15
PROJECT NUMBER: 03
PROJECT SUMMARY:
GITHUB LINK:
{{github_link}}

PROBLEM STATEMENT:
To build a convolutional nueral network which can classify the images of the fishes that are being shown to them.
Using Transfer learning to use the power of pre-trained models
Deploying the best performing model in streamlit application to showcase the ability of the model in classifying the images.
DATA PREPROCESSING AND AUGMENTATION:
The dataset was loaded from a zip file in Google Drive and unzipped to the working directory.

Image data augmentation was applied to the training data using ImageDataGenerator from TensorFlow Keras. The augmentation techniques included rescaling, rotation, zoom, width and height shifts, and horizontal flipping. The validation and test data were only rescaled.

The data was loaded into train_generator, val_generator, and test_generator using flow_from_directory.

Found 6225 images belonging to 11 classes.
Found 1092 images belonging to 11 classes.
Found 3177 images belonging to 11 classes.
MODEL TRAINING AND EVALUATION:
Several convolutional neural network models were built and evaluated for the fish image classification task.

Custom CNN Model
A custom CNN model was built with convolutional layers, max pooling, flatten, dense, and dropout layers.

The model was compiled with categorical_crossentropy loss, adam optimizer, and accuracy metric.

The model was trained for 10 epochs.

Test Loss: 11.30
Test Accuracy: 88.70
Visualizations of the model's accuracy and loss during training are provided in the notebook.

Transfer Learning with VGG16
The VGG16 pre-trained model was loaded with weights from 'imagenet', excluding the top classification layer. Custom dense and dropout layers were added on top of the VGG16 base.

The base VGG16 model layers were initially frozen. The model was compiled with categorical_crossentropy loss, adam optimizer, and accuracy metric.

The model was trained for 10 epochs.

VGG16 Test Accuracy: 94.74 VGG16 Test Loss: 5.26

Visualizations of the VGG16 model's accuracy and loss during training are provided in the notebook.

Fine-tuning the VGG16 Model
The trained VGG16 model was loaded, and some of the last convolutional layers of the VGG16 base were unfrozen for fine-tuning.

The model was compiled with categorical_crossentropy loss and the Adam optimizer with a low learning rate (1e-5).

The fine-tuned model was trained for 5 epochs.

Fine-tuned VGG16 Test Accuracy: 94.93 Fine-tuned VGG16 Loss: 5.07

Building ResNet50 Model
The ResNet50 pre-trained model was loaded with weights from 'imagenet', excluding the top classification layer. Custom dense and dropout layers were added on top of the ResNet50 base.

The base ResNet50 model layers were frozen. The model was compiled with categorical_crossentropy loss and the Adam optimizer with a learning rate of 1e-4.

The model was trained for 10 epochs.

ResNet50 Test Accuracy: 18.22 ResNet50 Test loss: 81.78

Visualizations of the ResNet50 model's accuracy and loss during training are provided in the notebook.

Fine-Tuning the ResNet50 Model
The trained ResNet50 model was loaded, and all layers except the last 10 were frozen for fine-tuning.

The model was compiled with categorical_crossentropy loss and the Adam optimizer with a low learning rate (1e-5).

The fine-tuned model was trained for 5 epochs.

Fine-tuned ResNet50 Test Accuracy: 35..19

Building EfficientNetB0 Model
The EfficientNetB0 pre-trained model was loaded with weights from 'imagenet', excluding the top classification layer. Custom dense and dropout layers were added on top of the EfficientNetB0 base.

The base EfficientNetB0 model layers were frozen. The model was compiled with categorical_crossentropy loss and the Adam optimizer with a learning rate of 1e-4.

The model was trained for 10 epochs.

EfficientNetB0 Test Accuracy: 16.05 EfficientNetB0 Test loss: 83.95

Visualizations of the EfficientNetB0 model's accuracy and loss during training are provided in the notebook.

Fine-Tuning the EfficientNetB0 Model
The trained EfficientNetB0 model was loaded, and all layers except the last 20 were frozen for fine-tuning.

The model was compiled with categorical_crossentropy loss and the Adam optimizer with a low learning rate (1e-5).

The fine-tuned model was trained for 5 epochs.

Fine-tuned EfficientNetB0 Test Accuracy: 16.05 Fine-tuned EfficientNetB0 test Loss: 83.95

MODEL COMPARISON GUIDE
Based on the test set evaluation, here's a comparison of the models trained:
Observations:

Custom CNN: Achieved a decent accuracy of 88.70 % on the test set.
VGG16 (Transfer Learning): Showed significant improvement compared to the custom CNN with a test accuracy of 94.76 %. This indicates the effectiveness of using a pre-trained model.
Fine-tuned VGG16: Further fine-tuning the VGG16 model resulted in a slight improvement in test accuracy to 94.93 %, suggesting that adapting the pre-trained layers to the specific dataset can be beneficial.
ResNet50 (Transfer Learning): The initial ResNet50 transfer learning model performed poorly with a low test accuracy of 18.22 %. This might be due to the model being frozen and the added layers not being sufficient to learn the features of the fish dataset effectively.
Fine-tuned ResNet50: Fine-tuning the ResNet50 model did not lead to significant improvement, with a test accuracy of 35.19 % This could indicate that the base ResNet50 model, even when partially unfrozen, is not as suitable for this dataset compared to VGG16.
EfficientNetB0 (Transfer Learning): Similar to ResNet50, the initial EfficientNetB0 transfer learning model also showed poor performance with a test accuracy of 16.05 %
Fine-tuned EfficientNetB0: Fine-tuning EfficientNetB0 did not improve the performance, resulting in a test accuracy of 16.05 %.
CONCLUSION
Based on these results, the Fine-tuned VGG16 model achieved the highest test accuracy of 94.93%, making it the best-performing model among the ones evaluated for this multiclass fish image classification task. The transfer learning approach with VGG16 proved to be much more effective than training a custom CNN from scratch or using ResNet50 and EfficientNetB0 for this dataset.

