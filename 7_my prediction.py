import os
import mediapipe as mp
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import tkinter as tk
from tkinter import messagebox, simpledialog
import os

def verify_login(username, password, filepath):
    try:
        password = password + "\n"
        with open(filepath, 'r') as file:
            lines=file.readlines()
            for line in lines:
                fields = line.split(",")
                if(fields[0]==username and fields[1]==password):
                    return True
    except Exception as e:
        print(e)
    return False

def sign_up(window):
    username = simpledialog.askstring("Username", "Enter Username")
    filepath = 'users.txt'
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            lines = file.readlines()
            for line in lines:
                fields = line.split(",")
                if fields[0] == username:
                    messagebox.showinfo("Error", "Username already exists. Please use a different username.")
                    return
    password = simpledialog.askstring("Password", "Enter Password", show='*')
    if password:
        with open(filepath, 'a') as file:
            file.write(username + ',' + password + '\n')
    else:
        messagebox.showinfo("Error", "Password cannot be empty.")

def login(window):
    username = simpledialog.askstring("Username", "Enter Username")
    password = simpledialog.askstring("Password", "Enter Password", show='*')
    if verify_login(username, password, 'users.txt'):
        messagebox.showinfo("Success", "Login successful.")
        window.destroy()
    else:
        messagebox.showinfo("Error", "Invalid username or password.")

def access_as_guest(window):
    messagebox.showinfo("Success", "Access granted as Guest.")
    window.destroy()

def main_window():
    window = tk.Tk()
    window.title("Main Window")
    login_button = tk.Button(window, text="Login", command=lambda: login(window))
    signup_button = tk.Button(window, text="Sign Up", command=lambda: sign_up(window))
    guest_button = tk.Button(window, text="Access as Guest", command=lambda: access_as_guest(window))
    login_button.pack()
    signup_button.pack()
    guest_button.pack()
    window.mainloop()

main_window()

# Initialize MediaPipe Holistic model and drawing utilities
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results

def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def draw_styled_landmarks(image, results):
    # Draw face connections with styling
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1))
    # Draw pose connections with styling
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2))
    # Draw left hand connections with styling
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2))
    # Draw right hand connections with styling
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

DATA_PATH = os.path.join(r'KSL 4 signs\KSL_Data')  
actions = np.array(['hello','toothache','fever','headache'])
no_sequences = 30
sequence_length = 30
start_folder = 30

label_map = {label:num for num, label in enumerate(actions)}

sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])

X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# load and evaluate a saved model
from keras.models import load_model
# load model
model = load_model(r'KSL 4 signs\KSLmodel.h5')
# summarize model.
model.summary()

res = model.predict(X_test)
#actions[np.argmax(res[4])]
#actions[np.argmax(y_test[4])]

#####prediction
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
yhat = model.predict(X_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

multilabel_confusion_matrix(ytrue, yhat)
accuracy_score(ytrue, yhat)

#Probability Visibility
#colors = [(245,117,16), (117,245,16), (16,117,245),(16,117,245),(16,117,245)]
#def prob_viz(res, actions, input_frame, colors):
    #output_frame = input_frame.copy()
    #for num, prob in enumerate(res):
        #cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        #cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 1, cv2.LINE_AA)
        
    #return output_frame

# 1. New detection variables
sequence = []
sentence = []
predictions = []
threshold = 0.9

import cv2
import tkinter as tk
from PIL import Image, ImageTk
import threading
from tkinter import simpledialog, messagebox, filedialog

# Create a window
window = tk.Tk()
window.title("KSL to text converter")

# Create a label for image display
image_label = tk.Label(window)
image_label.pack()

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
recording = False
writer = None
RSpeed = 1

def recording_speed():
    root = tk.Tk()
    root.title("What speed would you like your recording to be at?")
    global RSpeed
    RSpeed=5
    
    def option1():
        global RSpeed
        messagebox.showinfo("Slow", "Recording speed will be slow")
        RSpeed = 10
        root.destroy()
    
    def option2():
        global RSpeed
        messagebox.showinfo("Moderate", "Recording speed will be average.")
        RSpeed = 5
        root.destroy()
    
    def option3():
        global RSpeed
        messagebox.showinfo("Quick", "Recording speed will be fast.")
        RSpeed = 1
        root.destroy()

    button1 = tk.Button(root, text="Slow", command=option1)
    button1.pack()

    button2 = tk.Button(root, text="Moderate", command=option2)
    button2.pack()

    button3 = tk.Button(root, text="Quick", command=option3)
    button3.pack()

def record_video():
    global recording, writer
    recording = not recording
    if recording:
        file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 files", "*.mp4"),("AVI files", "*.avi"),("MOV files", "*.mov"),("WMV files", "*.wmv"),("FLV files", "*.flv")])

        recording_speed()  # Call the function to set recording speed
        if file_path:
            writer = cv2.VideoWriter(file_path, fourcc, 30.0, (640, 480))

def update_image():
    global recording, writer, RSpeed
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.9
    # Set mediapipe model 
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            # Read feed
            ret, frame = cap.read()

            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)
            
            # Draw landmarks
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                # 3. Viz logic
                if np.unique(predictions[-10:])[0]==np.argmax(res):
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                #image = prob_viz(res, actions, image, colors)
                
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            #qqcv2.imshow('OpenCV Feed', image)
            # Check if recording is enabled
            if recording:
                if writer is not None:
                    for _ in range(RSpeed):  # Increase or decrease this value to adjust the recording speed
                        writer.write(image)

            # Convert the image from BGR to RGB, convert to a PIL Image, then to a Tkinter PhotoImage
            cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGBA)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            
            # Update the label with the new image
            image_label.imgtk = imgtk
            image_label.configure(image=imgtk)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()

# Create start and stop recording buttons
start_button = tk.Button(window, text="Start or Stop Recording", command=record_video, bg='green')
start_button.pack(side=tk.LEFT)

# Create a thread to update the image in the label
thread = threading.Thread(target=update_image)
thread.start()

# Start the Tkinter event loop
window.mainloop()
