Pneumonia Detection using Deep Learning & Transfer LearningA deep learning project utilizing Transfer Learning with the VGG16 Convolutional Neural Network (CNN) architecture to classify chest X-ray images into two categories: Pneumonia and Normal.📌 Table of ContentsAbout The ProjectTools & TechnologiesDatasetStep-by-Step ImplementationModel Training & ResultsMaking Predictions📖 About The ProjectPneumonia is a dangerous lung infection that may occur in one or both lungs, usually caused by viruses, fungi, or bacteria. Early and accurate detection using chest X-rays is vital for effective treatment. This project implements a pre-trained VGG16 model, freezing its feature extraction layers and appending custom dense layers to classify chest X-ray scans into Pneumonia or Normal.🛠️ Tools and TechnologiesPythonTensorFlow & Keras: For building and training deep learning models.VGG16 Architecture: A widely used CNN architecture pre-trained on the ImageNet database.Transfer Learning (TL): Reusing pre-trained weights from ImageNet (1000 classes) to speed up training and boost performance on medical X-ray data.SciPy: Used for scientific computing and image transformations.Matplotlib & Glob: For data visualization and retrieving file/pathnames matching a specified pattern.📂 DatasetThe Chest X-ray dataset is sourced from Kaggle. It is structured into train, test, and validation directories, containing images separated into two folders: PNEUMONIA and NORMAL.chest_xray/
├── train/
│   ├── NORMAL/
│   └── PNEUMONIA/
├── test/
│   ├── NORMAL/
│   └── PNEUMONIA/
└── val/
    ├── NORMAL/
    └── PNEUMONIA/
⚙️ Step-by-Step Implementation1. Install Required Librariespip install tensorflow
pip install keras
pip install scipy
pip install glob2
2. Import Dependencies & Define Image ShapeWe use a fixed input image size of 224 x 224 with 3 color channels (RGB), which is required by the VGG16 architecture.from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plot
from glob import glob

IMAGESHAPE = [224, 224, 3] 
training_data = 'chest_xray/train'
testing_data = 'chest_xray/test'
3. Load VGG16 with Pre-trained ImageNet WeightsSetting include_top=False drops the original 1000-class output layer of ImageNet so we can attach our own custom classification layer.vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

# Freeze base layers so they are not re-trained
for each_layer in vgg_model.layers:
    each_layer.trainable = False
4. Add Custom Classification Layersclasses = glob('chest_xray/train/*')

flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)

final_model = Model(inputs=vgg_model.input, outputs=prediction)
final_model.summary()
5. Compile the Modelfinal_model.compile( 
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)
6. Data Augmentation & LoadingUsing ImageDataGenerator to rescale pixel values and apply random transformations (shear, zoom, horizontal flip) to prevent overfitting.from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale = 1./255,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True
)

testing_datagen = ImageDataGenerator(rescale = 1. / 255)

training_set = train_datagen.flow_from_directory(
    'chest_xray/train', 
    target_size = (224, 224),
    batch_size = 4,
    class_mode = 'categorical'
)

test_set = testing_datagen.flow_from_directory(
    'chest_xray/test',
    target_size = (224, 224),
    batch_size = 4,
    class_mode = 'categorical'
)
7. Train & Save the Modelfitted_model = final_model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)

# Save the trained model for future inference
final_model.save('our_model.h5')
📊 Model Training & ResultsEpochTraining AccuracyTraining LossValidation AccuracyValidation LossEpoch 1/588.30%0.441685.42%0.9806Epoch 3/594.52%0.215288.46%0.6164Epoch 5/595.32%0.221476.28%1.9244🔮 Making Predictions on New X-RaysOnce our_model.h5 is saved, you can load it to run inference on individual X-ray images:from keras.preprocessing import image
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
import numpy as np

# Load the saved model
model = load_model('our_model.h5')

# Load and preprocess the X-ray image
img = image.load_img('/content/chest_xray/test/PNEUMONIA/person100_bacteria_480.jpeg', target_size=(224, 224))
imagee = image.img_to_array(img)
imagee = np.expand_dims(imagee, axis=0)
img_data = preprocess_input(imagee)

# Predict
prediction = model.predict(img_data)

if prediction[0][0] > prediction[0][1]:
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')

print(f'Predictions: {prediction}')
Sample Output:1/1 ━━━━━━━━━━━━━━━━━━━━ 1s 779ms/step
Person is affected with Pneumonia.
Predictions: [[0. 1.]]
