from flask import Flask, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
import pickle
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Cho phép các request từ nguồn khác

# Load model từ file pickle
with open('./rf_model_new.pkl', 'rb') as f:
    model_dict = pickle.load(f)
model = model_dict['model']

# Định nghĩa ánh xạ nhãn
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'X', 23: 'Y'
}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.2,
    max_num_hands=2
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Biến toàn cục lưu kết quả frame cuối
current_prediction = {"result": "Unknown", "confidence": 0.0}

def gen_frames():
    global current_prediction
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Không thể mở camera.")
        return "Camera error", 500

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        predicted_character = "Unknown"
        confidence = 0.0

        if results.multi_hand_landmarks:
            # Lấy bàn tay đầu tiên
            hand_landmarks = results.multi_hand_landmarks[0]

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]

            min_x, max_x = min(x_list), max(x_list)
            min_y, max_y = min(y_list), max(y_list)
            range_x = max_x - min_x
            range_y = max_y - min_y

            data_aux = []
            for i in range(21):
                # Chuẩn hóa [0..1]
                norm_x = (hand_landmarks.landmark[i].x - min_x) / (range_x if range_x != 0 else 1)
                norm_y = (hand_landmarks.landmark[i].y - min_y) / (range_y if range_y != 0 else 1)
                data_aux.append(norm_x)
                data_aux.append(norm_y)

            print("Số đặc trưng (42) sau khi chuẩn hóa:", len(data_aux))

            if len(data_aux) == model.n_features_in_:
                prediction = model.predict([np.asarray(data_aux)])
                proba = model.predict_proba([np.asarray(data_aux)])
                confidence = float(np.max(proba))
                if prediction is not None and len(prediction) > 0:
                    predicted_label = prediction[0]
                    if isinstance(predicted_label, (int, np.int32, np.int64)):
                        predicted_character = labels_dict.get(int(predicted_label), "Unknown")
                    elif isinstance(predicted_label, str):
                        predicted_character = predicted_label

            # Vẽ khung xương
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
        else:
            print("Không phát hiện được bàn tay.")

        current_prediction = {"result": predicted_character, "confidence": confidence}

        # Vẽ text dự đoán lên frame (tuỳ chọn)
        # cv2.putText(frame, predicted_character, (50, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_data', methods=['GET'])
def current_data():
    return jsonify(current_prediction)

@app.route('/')
def index():
    return (
        "<h1>Welcome to the Sign Language Recognition API</h1>"
        "<p>Visit <code>/video_feed</code> for real-time video with skeleton.<br>"
        "Visit <code>/current_data</code> to get the latest prediction data.</p>"
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
