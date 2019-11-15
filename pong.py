from api import PRN
import numpy as np
import cv2,config
from utils.cv_plot import plot_kpt, plot_vertices, plot_pose_box
from utils.estimate_pose import estimate_pose
from game_engine.pong_game import PongGame

prn = PRN(is_dlib = True)
pong_game = PongGame()
cap = cv2.VideoCapture(0)


cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.CAMERA_HEIGHT)
COLOR_UP = (0,255,0)
COLOR_DOWN = (0,0,255)
COLOR_STAY = (0,255,255)
def head_pose(frame):
    global prn
    try:
        pos = prn.process(frame)       
        kpt = prn.get_landmarks(pos)
        vertices = prn.get_vertices(pos)
        camera_matrix, pose = estimate_pose(vertices)
        if pose[1] > config.SPEED_STEP:
            color = COLOR_UP
        elif pose[1] < - config.SPEED_STEP:
            color = COLOR_DOWN
        else:
            color = COLOR_STAY
        frame = plot_pose_box(frame, camera_matrix, kpt, color=color)
    except:
        pose = (0.0,0.0,0.0)
    return frame, pose
def speed_step(s):
    step = config.SPEED_STEP
    s = -s
    if s > step * 3:
        return 3
    if s > step * 2:
        return 2
    if s > step:
        return 1
    if s > step * -1:
        return 0
    if s > step * -2:
        return -1
    if s > step * -3:
        return -2
    else:
        return -3

is_play_game = False
while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Our operations on the frame come here
    frame = cv2.flip(frame, 1)
    half_width = int(config.CAMERA_WIDTH / 2)


    frame[:,:half_width], pose_left = head_pose(frame[:,:half_width])
    frame[:,half_width:], pose_right = head_pose(frame[:,half_width:])

    cv2.line(frame, (half_width, 0), (half_width, config.CAMERA_HEIGHT), (0,255,0), 2)
    # Pong Game Control
    if is_play_game:
        pong_game.make_move(speed_step(pose_left[1]),speed_step(pose_right[1]))
    pong_game.draw(frame)
    # Display the resulting frame
    cv2.imshow(config.WINDOW_TITLE,frame)
    waitTime = 1
    if pose_left[1] == 0.0 and pose_right[1] == 0.0:
        waitTime = 30
    keyboard = cv2.waitKey(waitTime)
    if keyboard & 0xFF == ord('q'):
        break
    if keyboard & 0xFF == ord('r'):
        pong_game.reset_game()
        is_play_game = False
    if keyboard & 0xFF == ord('p'):
        is_play_game = not is_play_game

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()