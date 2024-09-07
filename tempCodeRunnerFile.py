from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the ML model
model = load_model('cnn8grps_rad1_model.h5')

# Define the camera capture function
def capture_hand_gesture():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Preprocess the frame
        frame = cv2.resize(frame, (224, 224))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame / 255.0
        # Get the hand landmarks using MediaPipe
        mpHands = mp.solutions.hands
        hands = mpHands.Hands(static_image_mode=False, max_num_hands=1)
        results = hands.process(frame)
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Draw the hand landmarks on the frame
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            for landmark in hand_landmarks.landmark:
                x, y, z = landmark.x, landmark.y, landmark.z
                cv2.circle(frame, (int(x * 224), int(y * 224)), 2, (0, 255, 0), -1)
            # Make predictions using the ML model
            predictions = model.predict(np.array([frame]))
            predicted_class = np.argmax(predictions[0])
            # Display the predicted class on the screen
            cv2.putText(frame, f"Predicted: {predicted_class}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow('Hand Gesture Recognition', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# Define the Flask route for the camera capture
@app.route('/capture', methods=['GET'])
def capture():
    capture_hand_gesture()
    return 'Capturing hand gesture...'

# Define the Flask route for the ML model prediction
@app.route('/predict', methods=['POST'])
def predict():
    image = request.get_json()['image']
    image = np.array(image)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])
    return jsonify({'predicted_class': predicted_class})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)