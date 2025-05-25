## Face Detection and Emotion Recognition

This project performs real-time face detection using OpenCV and predicts the emotion of the detected face using a pre-trained deep learning model built with TensorFlow/Keras.

## 🔍 Features

- Detect faces in live video frames or images using Haar Cascade classifier.
- Preprocess the detected face and classify emotion using a deep learning model.
- Normalize image input and resize to model input size.
- Display prediction results with confidence.

## 🧠 Model

- Input Shape: (224, 224, 3)
- Preprocessing: Normalization to [0, 1], resizing, and batch dimension expansion.
- Output: Class probabilities over 7 emotions.
- Emotion Labels:
  - `['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']`

## 🧪 Example Output

```python
Predictions = new_model.predict(final_image)
print(np.argmax(Predictions))  # Index of the highest confidence emotion
## Install the dependencies using pip:
pip install opencv-python tensorflow numpy matplotlib
##Load the model:
new_model = tf.keras.models.load_model("your_model_path.h5")
##Run face detection:
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
##Predict emotion for each detected face.

## 🖼 Sample Prediction Pipeline

# Resize and normalize
final_image = cv2.resize(face_roi, (224, 224))
final_image = np.expand_dims(final_image, axis=0)
final_image = final_image / 255.0

# Predict
predictions = new_model.predict(final_image)
predicted_emotion = class_names[np.argmax(predictions)]
```
