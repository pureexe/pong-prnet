from api import PRN
import numpy as np
import cv2,config
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box
from utils.estimate_pose import estimate_pose
from game.pong_game import Pong


prn = PRN(is_dlib = True)
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)

def head_pose(frame):
    global prn
    try:
        pos = prn.process(frame)       
        kpt = prn.get_landmarks(pos)
        vertices = prn.get_vertices(pos)
        camera_matrix, pose = estimate_pose(vertices)
        frame = plot_pose_box(frame, camera_matrix, kpt)
    except:
        pose = (0.0,0.0,0.0)
    return frame, pose

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame = cv2.flip(frame, 1)
    half_width = int(config.CAMERA_WIDTH / 2)


    frame[:,:half_width], pose_left = head_pose(frame[:,:half_width])
    frame[:,half_width:], pose_right = head_pose(frame[:,half_width:])

    cv2.line(frame, (half_width, 0), (half_width, config.CAMERA_HEIGHT), (0,255,0), 2)
    # Display the resulting frame
    cv2.imshow(config.WINDOW_TITLE,frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()