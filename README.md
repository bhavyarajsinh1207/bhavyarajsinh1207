import numpy as np
import tensorflow as tf
import keras
from tensorflow.keras.layers import Dense, Input, Global, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

import cv2
import face_recognition
from PIL import Image 

# Generate batches of tensor image data with real-time data augmentation
# Transform every pixel from the range [0, 255] to [0, 1]
train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

# Takes the path to a directory & generates batches of augmented data.
training_set = train_datagen.flow_from_directory(
        'train/',
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

test_set = test_datagen.flow_from_directory(
        'test/',
        target_size=(48,48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')


# Creating a simple CNN
i = Input(shape=(training_set[0].shape))

x = Conv2D(32, kernel_size=(3, 3), kernel_constraint=max_norm(3.), activation='relu', input_shape=(48,48,1))(i)
x = MaxPool2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(64, kernel_size=(3, 3), kernel_constraint=max_norm(3.), activation='relu', input_shape=(48,48,1))(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)

x = Conv2D(128, kernel_size=(3, 3), kernel_constraint=max_norm(3.), activation='relu', input_shape=(48,48,1))(x)
x = MaxPool2D(pool_size=(2, 2))(x)
x = BatchNormalization()(x)

x = Flatten()(x)
x = Dense(512, kernel_constraint=max_norm(3.), activation='relu')(x)
x = Dropout(0.2)(x)

o = Dense(7, activation='softmax')(x)

model = Model(i, o)

optimizer = SGD(learning_rate=0.01, momentum=0.5)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])   
model.summary()

history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Plotting histories and accuracies 
plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='acc')
plt.plot(history.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

# Predicting on test set 
y_pred = model.predict(x_test)
y_pred = (y_pred > 0.5)
from sklearn.metrics import classification_report, confusion_matrix
cm = classification_report(y_test, y_pred)
print(cm)
conf = confusion_matrix(y_test, y_pred)
print(conf)


# GETTING REAL TIME USER IMAGE FROM WEBCAM 
# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)

# Initialize variables
face_locations = []

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    
    
    
    # Display the results
    for top, right, bottom, left in face_locations:
        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    if cv2.waitKey(1) & 0xFF == ord('t'):
        ret, frame = video_capture.read()
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        data = Image.fromarray(frame).convert('LA')
        
        for top, right, bottom, left in face_locations: 
            box = (left, top, right, bottom)
        
        cropped_image = data.crop(box)
        final_image = cropped_image.resize((48, 48))
        break
            
    
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()



# Predicting a single image 

final_image.save("def.png")

prediction_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

prediction_image = train_datagen.flow_from_directory(
        'pred/',
        target_size=(48,48),
        color_mode='grayscale',
        classes=None)

emotion_prediction = model.predict(prediction_image)

# CNN gives out probabilities for all the classes, we want the maximum one
final_prediction = int(np.argmax(emotion_prediction))

emotion_list = ['angry', 'disgusted', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

print('The final prediction is ', emotion_list[final_prediction])
