<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Pneumonia Detection using Deep Learning</title>
</head>
<body>

    <h1>Pneumonia Detection using Deep Learning & Transfer Learning</h1>
    
    <p>
        A deep learning project utilizing <strong>Transfer Learning</strong> with the <strong>VGG16</strong> Convolutional Neural Network (CNN) architecture to classify chest X-ray images into two categories: <strong>Pneumonia</strong> and <strong>Normal</strong>.
    </p>

    <hr>

    <h2>📌 Table of Contents</h2>
    <ul>
        <li><a href="#about-the-project">About The Project</a></li>
        <li><a href="#tools-and-technologies">Tools & Technologies</a></li>
        <li><a href="#dataset">Dataset</a></li>
        <li><a href="#step-by-step-implementation">Step-by-Step Implementation</a></li>
        <li><a href="#model-training-results">Model Training & Results</a></li>
        <li><a href="#making-predictions">Making Predictions</a></li>
        <li><a href="#how-to-run">How to Run</a></li>
    </ul>

    <hr>

    <h2 id="about-the-project">📖 About The Project</h2>
    <p>
        Pneumonia is a serious lung infection that affects one or both lungs. Early and accurate detection using chest X-rays is vital for effective treatment. This project implements a pre-trained VGG16 model, freezing its feature extraction layers and appending custom dense layers to classify chest X-ray scans with high accuracy.
    </p>

    <h2 id="tools-and-technologies">🛠️ Tools and Technologies</h2>
    <ul>
        <li><strong>Python</strong></li>
        <li><strong>TensorFlow & Keras</strong>: For building and training deep learning models.</li>
        <li><strong>VGG16 Architecture</strong>: A robust CNN pre-trained on the ImageNet database.</li>
        <li><strong>Transfer Learning</strong>: Reusing pre-trained weights to speed up training and boost performance on specialized medical data.</li>
        <li><strong>SciPy</strong>: Used for scientific computing and image transformations.</li>
        <li><strong>Matplotlib & Glob</strong>: For visualization and file/path retrieval.</li>
    </ul>

    <h2 id="dataset">📂 Dataset</h2>
    <p>
        The chest X-ray dataset is sourced from Kaggle. It is structured into <code>train</code>, <code>test</code>, and <code>validation</code> directories, containing images separated into folders for <strong>Pneumonia</strong> and <strong>Normal</strong> classes.
    </p>

    <h2 id="step-by-step-implementation">⚙️ Step-by-Step Implementation</h2>

    <h3>1. Install Required Libraries</h3>
    <pre><code>!pip install tensorflow
!pip install keras
!pip install scipy
!pip install glob2</code></pre>

    <h3>2. Import Dependencies & Define Image Shape</h3>
    <pre><code>from keras.models import Model
from keras.layers import Flatten, Dense
from keras.applications.vgg16 import VGG16
import matplotlib.pyplot as plot
from glob import glob

IMAGESHAPE = [224, 224, 3] 
training_data = 'chest_xray/train'
testing_data = 'chest_xray/test'</code></pre>

    <h3>3. Load VGG16 with Pre-trained ImageNet Weights</h3>
    <p>We set <code>include_top=False</code> to drop the original 1000-class output layer of ImageNet and prepare the model for binary/multiclass classification.</p>
    <pre><code>vgg_model = VGG16(input_shape=IMAGESHAPE, weights='imagenet', include_top=False)

# Freeze base layers so they are not updated during training
for each_layer in vgg_model.layers:
    each_layer.trainable = False</code></pre>

    <h3>4. Add Custom Classification Layers</h3>
    <pre><code>classes = glob('chest_xray/train/*')

flatten_layer = Flatten()(vgg_model.output)
prediction = Dense(len(classes), activation='softmax')(flatten_layer)

final_model = Model(inputs=vgg_model.input, outputs=prediction)
final_model.summary()</code></pre>

    <h3>5. Compile the Model</h3>
    <pre><code>final_model.compile( 
  loss='categorical_crossentropy',
  optimizer='adam',
  metrics=['accuracy']
)</code></pre>

    <h3>6. Data Augmentation & Loading with ImageDataGenerator</h3>
    <pre><code>from tensorflow.keras.preprocessing.image import ImageDataGenerator

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
)</code></pre>

    <h3>7. Train the Model</h3>
    <pre><code>fitted_model = final_model.fit(
    training_set,
    validation_data=test_set,
    epochs=5,
    steps_per_epoch=len(training_set),
    validation_steps=len(test_set)
)</code></pre>

    <h3>8. Save the Model</h3>
    <pre><code>final_model.save('our_model.h5')</code></pre>

    <h2 id="model-training-results">📊 Model Training & Results</h2>
    <ul>
        <li><strong>Epoch 1/5:</strong> accuracy: 0.8830 | loss: 0.4416 | val_accuracy: 0.8542 | val_loss: 0.9806</li>
        <li><strong>Epoch 3/5:</strong> accuracy: 0.9452 | loss: 0.2152 | val_accuracy: 0.8846 | val_loss: 0.6164</li>
        <li><strong>Epoch 5/5:</strong> accuracy: 0.9532 | loss: 0.2214 | val_accuracy: 0.7628 | val_loss: 1.9244</li>
    </ul>

    <h2 id="making-predictions">🔮 Making Predictions on New X-Rays</h2>
    <pre><code>from keras.preprocessing import image
from keras.models import load_modelfrom keras.applications.vgg16 import preprocess_input
import numpy as np

model = load_model('our_model.h5') # Load trained model

img = image.load_img('/content/chest_xray/test/PNEUMONIA/person100_bacteria_480.jpeg', target_size=(224, 224))
imagee = image.img_to_array(img)
imagee = np.expand_dims(imagee, axis=0)
img_data = preprocess_input(imagee)

prediction = model.predict(img_data)

if prediction[0][0] > prediction[0][1]:
    print('Person is safe.')
else:
    print('Person is affected with Pneumonia.')
print(f'Predictions: {prediction}')</code></pre>

    <hr>
    
    <p><em>Modified and documented for GitHub.</em></p>

</body>
</html>
