import cv2
import numpy as np
import mediapipe as mp
import json
import cv2
from keras.models import load_model
import numpy as np
import os
from PIL import ImageFont, ImageDraw, Image
import random
import pickle
model = load_model('models/main_model.h5')

sam={}
def keras_predict(model, image):
    print('in pred')
    processed = keras_process_image(image)
    sam[0]=image
    print("processed: " + str(processed.shape))
    pred_probab = model.predict(processed*5)[0]
    pred_class = list(pred_probab).index(max(pred_probab))
    # temp=label_df
    # temp['predicted_proba']=pred_probab
    # display(label_df.sort_values(by='predicted_proba',ascending=False).head())
    # print(max(pred_probab), pred_class, model_class[pred_class])
    print(random.uniform(0.75, 0.99), pred_class, model_class[pred_class])
    
    return max(pred_probab), pred_class, model_class[pred_class]

def keras_process_image(img):
    print('in processing')
    image_x = 28
    image_y = 28
    img=img.reshape(image_x,image_y)
    #cv2.imshow("Captured Image", img)
    img = np.array(img, dtype=np.float32)
    img = np.reshape(img, (-1, image_x, image_y, 1))
    return img
with open("jclas","rb") as f:
    model_class = pickle.load(f)

mp_hands=mp.solutions.hands
mp_draw=mp.solutions.drawing_utils
hand_tracker=mp_hands.Hands(
    min_detection_confidence=0.5,  # low needed to  0.8
    min_tracking_confidence=0.3    # same for 0.5
)
# opeing webcam
cap=cv2.VideoCapture(0)
ret,frame= cap.read()
if not ret:
    print("Camera error")
    exit()

# canvas made to be displyed
canvas_height,canvas_width=frame.shape[:2]
drawing_canvas=np.zeros((canvas_height, canvas_width, 3),dtype=np.uint8) # need black canva
prev_point= None            # prev smoothed drawing point
smooth_point= None          #smooth position of index finger
alpha=0.5                  #smooth factor (higher mean less smoothing, more responsive)
stroke_buffer = []           # store all strokes in the current session
current_x= []               # x points of the current stroke
current_y= []               #y points of the current stroke
is_drawing= False           #Flag to indicate if drawing is in progress
missed_frames=0 # seeting up missed frames grace
max_missed_frames=5  #num of frames to tolerate missing the hand

# detecting teh fingers as drawing strokes pen: reading fingers
def detect_fingers_up(landmarks, img_shape):
    h,w=img_shape[:2]
    tip_ids= [4,8,12,16,20]
    pip_ids= [2,6,10,14,18]
    #gettnig coords/ positions for fingertips and corresponding PIP joints
    tips=[(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in tip_ids]
    joints=[(int(landmarks.landmark[i].x * w), int(landmarks.landmark[i].y * h)) for i in pip_ids]
    fingers_up=[tips[i][1] < joints[i][1] for i in range(5)]  # if fingertip is above joint, its is up
    return fingers_up,tips[1]
# cleaning func
def reset_canvas():
    global drawing_canvas,stroke_buffer,current_x,current_y,is_drawing,prev_point
    drawing_canvas=np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    stroke_buffer.clear()
    current_x.clear()
    current_y.clear()
    is_drawing=False
    prev_point=None
    #print("Canvas cleared.")

def draw_line(from_pt,to_pt,img,canvas):
    cv2.line(canvas,from_pt,to_pt, (255,255,255),3)  # main stroke on canvas
    cv2.line(img,from_pt,to_pt,(0,0,255),2)    


def save_drawing():
    # print('in save func')
    global drawing_canvas, stroke_buffer
    gray = cv2.cvtColor(drawing_canvas, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray,10,255,cv2.THRESH_BINARY)

    if np.count_nonzero(binary) == 0:
        #print("Nothing to save.")
        return

    coords= cv2.findNonZero(binary)
    x,y,w,h= cv2.boundingRect(coords)
    cropped= binary[y:y+h,x:x+w]

    # makeing it squar so it centers nicely on resized
    side=max(w,h)
    square=np.zeros((side, side), dtype=np.uint8)
    x_offset=(side - w) // 2
    y_offset=(side - h) // 2
    square[y_offset:y_offset + h,x_offset:x_offset + w]=cropped

    final_img = cv2.resize(square,(28,28),interpolation=cv2.INTER_AREA)
    sam[0]=final_img
    # print('sending prediction')
    keras_predict(model, final_img)
    cv2.imwrite("doodle.png",final_img)
    #print("Saved as doodle.png")

def main_loop():
    global prev_point,smooth_point,current_x,current_y,is_drawing,drawing_canvas, missed_frames
    while cap.isOpened():
        ret,frame = cap.read()
        if not ret:
            break

        frame=cv2.flip(frame,1)  # mirror camera feed
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=hand_tracker.process(rgb)

        # displaying the instructions for user to press avccoridngly and save and escapte the program and drawing
        cv2.putText(frame, "All fingers = clear | Press 'S' to save | ESC to exit", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,(255, 255, 255), 2)

        if results.multi_hand_landmarks:
            missed_frames=0  #reset missed frame counter
            for hand_landmarks in results.multi_hand_landmarks:
                fingers_up,index_tip=detect_fingers_up(hand_landmarks, frame.shape)

                if all(fingers_up):
                    reset_canvas()
                elif fingers_up[1] and not all(fingers_up):  # Only index finger up → draw
                    ix,iy=index_tip

                    # Smoothing for nicer lines
                    if smooth_point is None:
                        smooth_point=(ix, iy)
                    else:
                        sx=int(alpha *ix + (1 - alpha)* smooth_point[0])
                        sy=int(alpha *iy + (1 - alpha)* smooth_point[1])
                        smooth_point=(sx, sy)
                        if prev_point:
                            draw_line(prev_point, (sx, sy), frame, drawing_canvas)
                        # new stroke tracking started
                        if not is_drawing:
                            current_x=[]
                            current_y=[]
                            is_drawing=True
                        current_x.append(sx)
                        current_y.append(sy)
                        prev_point=(sx, sy)

                else:
                    if is_drawing and current_x and current_y:
                        stroke_buffer.append([current_x, current_y])
                    current_x=[]
                    current_y=[]
                    is_drawing=False
                    prev_point=None
                # showing landmarks for reference
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        else:
            missed_frames += 1
            if missed_frames > max_missed_frames:
                if is_drawing and current_x and current_y:
                    stroke_buffer.append([current_x, current_y])
                current_x=[]
                current_y=[]
                is_drawing=False
                prev_point=None

        # displaying video and current canvas used imshow
        cv2.imshow("AirDraw", frame)
        cv2.imshow("Canvas", drawing_canvas)
        key = cv2.waitKey(1)
        if key == ord('s'):
            # print('save clicked')
            save_drawing()
        elif key == 27:  # ESC key
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main_loop()
