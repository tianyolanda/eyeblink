# %% [markdown]
# # EYE BLINK DETECTION:
# # 1) Simple Model
# Based on tutorial:  
# https://www.pyimagesearch.com/2017/04/24/eye-blink-detection-opencv-python-dlib/
# 
# For every session, unfound packages have to be installed. You can install unfound packages via !pip install command i.e.  
# > !pip install opencv-python  
# > !pip install cmake dlib  
# > !pip install --upgrade imutils
# 
# Opencv and dlib comes with pre-installed in kaggle environement so installing only imutils would be enough.

# %% [markdown]
# **Initialization of the Code:**
# 
# Necesarry installations, imports and definitions. 
# 
# Basic model uses dlib's "get_frontal_face_detector" to detect faces. It uses HoG and SVM methods.  
# http://dlib.net/python/index.html#dlib.get_frontal_face_detector
# 
# Basic model also uses dlib's "shape_predictor". It's implementation of [paper by Kazemi and Sullivan (2014).](https://www.semanticscholar.org/paper/One-millisecond-face-alignment-with-an-ensemble-of-Kazemi-Sullivan/d78b6a5b0dcaa81b1faea5fb0000045a62513567)
# http://dlib.net/python/index.html#dlib.shape_predictor
# 
# You can download a trained facial landmark predictor from:  
# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# 
# Or train your own by:  
# http://dlib.net/train_shape_predictor.py.html

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:29:52.914155Z","iopub.execute_input":"2023-06-16T06:29:52.914537Z","iopub.status.idle":"2023-06-16T06:30:06.183024Z","shell.execute_reply.started":"2023-06-16T06:29:52.914477Z","shell.execute_reply":"2023-06-16T06:30:06.181959Z"}}
# install imutils
# !pip install --upgrade imutils

# import packages
from scipy.spatial import distance as dist
import imutils
from imutils import face_utils
import numpy as np
import pandas as pd
import dlib
import cv2
import os
import time
import h5py
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt
# %matplotlib inline

# define three constants. 
# You can later experiment with these constants by changing them to adaptive variables.
EAR_THRESHOLD = 0.21 # eye aspect ratio to indicate blink
EAR_CONSEC_FRAMES = 3 # number of consecutive frames the eye must be below the threshold
SKIP_FIRST_FRAMES = 0 # how many frames we should skip at the beggining

# initialize dlib variables
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("../input/eyeblink/shape_predictor_68_face_landmarks.dat")

# initialize output structures
scores_string = ""

# %% [markdown]
# **Datasets are ready-to-use in "../input" folder. Let's take a look at them.**  
# > yawning.avi --> a test video from NTHU dataset (no annotations)  
# > shape_predictor_68_face_landmarks.dat --> model of dlib's face predictor  
# > eyeblink8 folder --> 8 videos with annotations for blink detection purposes made by https://www.blinkingmatters.com/research  
# > talkingFace folder --> 1 video with annotation for blink detection purposes made by https://www.blinkingmatters.com/research
# 
# You can run this section **to see content of input folder:**

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:30:06.184872Z","iopub.execute_input":"2023-06-16T06:30:06.185186Z","iopub.status.idle":"2023-06-16T06:30:06.190174Z","shell.execute_reply.started":"2023-06-16T06:30:06.185130Z","shell.execute_reply":"2023-06-16T06:30:06.189007Z"}}
# print content of any folder
def display_folder(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:30:06.191406Z","iopub.execute_input":"2023-06-16T06:30:06.191669Z","iopub.status.idle":"2023-06-16T06:30:06.254633Z","shell.execute_reply.started":"2023-06-16T06:30:06.191629Z","shell.execute_reply":"2023-06-16T06:30:06.252631Z"}}
# print content of "../input" folder
display_folder("../input/eyeblink")

# %% [markdown]
# **We define a function that calculates EAR values:**
# 
# It's an implementation of [paper by Soukupova and Cech (2016).](http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf)

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:30:06.255928Z","iopub.execute_input":"2023-06-16T06:30:06.256231Z","iopub.status.idle":"2023-06-16T06:30:06.262998Z","shell.execute_reply.started":"2023-06-16T06:30:06.256189Z","shell.execute_reply":"2023-06-16T06:30:06.261702Z"}}
# define ear function
def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear

# %% [markdown]
# **We define a function that reads a video and calculate it's EAR values**:  

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:30:06.266009Z","iopub.execute_input":"2023-06-16T06:30:06.266299Z","iopub.status.idle":"2023-06-16T06:30:06.291452Z","shell.execute_reply.started":"2023-06-16T06:30:06.266257Z","shell.execute_reply":"2023-06-16T06:30:06.290441Z"}}
# process a given video file 
def process_video(input_file,detector=dlib_detector,predictor=dlib_predictor,\
                  lStart=42,lEnd=48,rStart=36,rEnd=42,ear_th=0.21,consec_th=3, up_to = None):
    #define necessary variables
    COUNTER = 0
    TOTAL = 0
    current_frame = 1
    blink_start = 0
    blink_end = 0
    closeness = 0
    output_closeness = []
    output_blinks = []
    blink_info = (0,0)
    processed_frames = []
    frame_info_list = []

    #define capturing method
    cap = cv2.VideoCapture(input_file)
    time.sleep(1.0)
    
    #build a dictionary video_info
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count/fps
    video_info_dict = {
                'fps': fps,
                'frame_count': frame_count,
                'duration(s)': duration,
            }

    while True:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels)
        (grabbed, frame) = cap.read()
        if not grabbed:
            break
        height = frame.shape[0]
        weight = frame.shape[1]
        frame = cv2.resize(frame, (480, int(480*height/weight)))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # detect faces in the grayscale frame
        rects = detector(gray, 0)

        # loop over the face detections
        for rect in rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, rect)
            shape = np.array([[p.x,p.y] for p in shape.parts()])
            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            # average the eye aspect ratio together for both eyes
            ear = (leftEAR + rightEAR) / 2.0

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < ear_th:
                COUNTER += 1
                closeness = 1
                output_closeness.append(closeness)
            # otherwise, the eye aspect ratio is not below the blink
            # threshold
            else:
                # if the eyes were closed for a sufficient number of
                # then increment the total number of blinks
                if COUNTER >= consec_th:
                    TOTAL += 1
                    blink_start = current_frame - COUNTER
                    blink_end = current_frame - 1
                    blink_info = (blink_start,blink_end)
                    output_blinks.append(blink_info)
                # reset the eye frame counter
                COUNTER = 0
                closeness = 0
                output_closeness.append(closeness)

            # draw the total number of blinks on the frame along with
            # the computed eye aspect ratio for the frame
            cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # build frame_info dictionary then add to list
            frame_info={
                    'frame_no': current_frame,
                    'face_detected': not(rect.is_empty()),
                    'face_coordinates': [[rect.tl_corner().x,rect.tl_corner().y],
                                        [rect.tr_corner().x,rect.tr_corner().y], 
                                        [rect.bl_corner().x,rect.bl_corner().y],
                                        [rect.br_corner().x,rect.br_corner().y]],
                    'left_eye_coordinates': [leftEye[0], leftEye[1]],
                    'right_eye_coordinates': [rightEye[0], rightEye[1]],
                    'left_ear': leftEAR, 
                    'right_ear': rightEAR,
                    'avg_ear': ear, 
                    'closeness': closeness, 
                    'blink_no': TOTAL,
                    'blink_start_frame': blink_start,
                    'blink_end_frame': blink_end,
                    'reserved_for_calibration': False
                }
            frame_info_list.append(frame_info)

        # show the frame (this part doesn't work in online kernel. If you are running on offline jupyter
        # notebook, you can uncomment this part and try displaying video frames)
#        cv2.imshow("Frame", frame)
#        key = cv2.waitKey(1) & 0xFF
#        # if the `q` key was pressed, break from the loop
#        if key == ord("q"):
#            break

        #append processed frame to list
        processed_frames.append(frame)
        current_frame += 1
        frame_info_df = pd.DataFrame(frame_info_list) #build a dataframe from frame_info_list
        if up_to==current_frame-1:
            break

    # a bit of clean-up
    cv2.destroyAllWindows()
    cap.release()
    
    # print status
    file_name = os.path.basename(input_file)
    output_str = "Processing {} has done.\n\n".format(file_name)
    print(output_str)
    
    return frame_info_df, output_closeness, output_blinks, processed_frames, video_info_dict, output_str

# %% [markdown]
# **Process a test video by using the function defined above:**  
# 
# For example you can use the video "../input/talkingFace/talking.avi" 
# 
# **NOTE:** Depending on your input video, this part may take a long time, about ~10-15 mins.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:30:06.294118Z","iopub.execute_input":"2023-06-16T06:30:06.294477Z","iopub.status.idle":"2023-06-16T06:32:56.145608Z","shell.execute_reply.started":"2023-06-16T06:30:06.294431Z","shell.execute_reply":"2023-06-16T06:32:56.143688Z"}}
# full path of a video file
file_path = "../input/eyeblink/talkingFace/talking.avi"

# process the video and get the results
frame_info_df, closeness_predictions, blink_predictions, frames, video_info, scores_string \
    = process_video(file_path, ear_th=EAR_THRESHOLD, consec_th=EAR_CONSEC_FRAMES)

# %% [markdown]
# **We define a function that recalculates data:**  
# 
# We will discard some of the first frames which will be reserved for calibration phase and recalculate inputs **frame_info_df**, **closeness_list** and **blinks_list**.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:32:56.147800Z","iopub.execute_input":"2023-06-16T06:32:56.148138Z","iopub.status.idle":"2023-06-16T06:32:56.157855Z","shell.execute_reply.started":"2023-06-16T06:32:56.148082Z","shell.execute_reply":"2023-06-16T06:32:56.156744Z"}}
# recalculate the data from processing video by skipping first n frames
def skip_first_n_frames(frame_info_df, closeness_list, blink_list, processed_frames, skip_n=0, consec_th=3):
    # recalculate closeness_list
    recalculated_closeness_list = closeness_list[skip_n:] # skip first n frames

    # update 'reserved_for_calibration' column of frame_info_df for first "skip_n" frames
    frame_info_df.loc[:skip_n-1, 'reserved_for_calibration'] = True # .loc includes second index -> [first:second] 
    
    # recalculate blink_list
    # get blink count in the first "SKIP_FIRST_FRAMES" frames
    blink_count_til_n = frame_info_df.loc[skip_n, 'blink_no']    
    # determine start of the blink that comes after first n frames
    start_of_blink = blink_list[blink_count_til_n][0] - 1   #(-1) since frame-codes in blink_list start from 1
    # if some frames of the the blink starts before n
    if start_of_blink < skip_n: 
        # find frames of the blink that comes before n
        frames_to_discard = skip_n - start_of_blink
        # find duration of the blink
        duration_of_blink = blink_list[blink_count_til_n][1] - blink_list[blink_count_til_n][0] + 1
        # calculate new duration of blink after discarding first n frames
        new_duration = duration_of_blink - frames_to_discard
        # if new duration of the blink that comes after first n frames is less than n 
        if new_duration < consec_th:
            # then reduce total blink count by (blink_count_til_n + 1)
            recalculated_blink_list = blink_list[blink_count_til_n + 1:]
        # if new duration of the blink is NOT less than n 
        else:
            # then reduce total blink count by (blink_count_til_n)
            recalculated_blink_list = blink_list[blink_count_til_n:]
    # if the blink starts after n
    else:
        # then reduce total blink count by (blink_count_til_n)
        recalculated_blink_list = blink_list[blink_count_til_n:]
            
    # re-assign the frame-codes of recalculated_blinks if some frames are discarded       
    if skip_n > 0: 
        recalculated_blink_list = [(blink[0]-skip_n, blink[1]-skip_n) for blink in recalculated_blink_list]
        
    # also discard first n frames of "processed_frames"
    recalculated_processed_frames = processed_frames[skip_n:]
    
    return frame_info_df, recalculated_closeness_list, recalculated_blink_list, recalculated_processed_frames

# %% [markdown]
# **Recalculate results:**
# 
# Discard SKIP_FIRST_FRAMES of the first frames and recalculate **frame_info_df**, **closeness_predictions** and **blinks** by using skip_first_n_frames() function.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:32:56.159375Z","iopub.execute_input":"2023-06-16T06:32:56.159688Z","iopub.status.idle":"2023-06-16T06:32:56.193708Z","shell.execute_reply.started":"2023-06-16T06:32:56.159637Z","shell.execute_reply":"2023-06-16T06:32:56.192292Z"}}
# recalculate data by skipping "SKIP_FIRST_FRAMES" frames
frame_info_df, closeness_predictions_skipped, blink_predictions_skipped, frames_skipped \
    = skip_first_n_frames(frame_info_df, closeness_predictions, blink_predictions, frames, \
                          skip_n = SKIP_FIRST_FRAMES)

# %% [markdown]
# **We define a function that displays statistics:**  

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:32:56.195326Z","iopub.execute_input":"2023-06-16T06:32:56.195802Z","iopub.status.idle":"2023-06-16T06:32:56.208715Z","shell.execute_reply.started":"2023-06-16T06:32:56.195712Z","shell.execute_reply":"2023-06-16T06:32:56.207819Z"}}
# display statistics
# if you want to display test scores set test=True to change headline
def display_stats(closeness_list, blinks_list, video_info = None, skip_n = 0, test = False):
    str_out = ""
    # write video info
    if video_info != None:
        str_out += ("Video info\n")
        str_out += ("FPS: {}\n".format(video_info["fps"]))
        str_out += ("FRAME_COUNT: {}\n".format(video_info["frame_count"]))
        str_out += ("DURATION (s): {:.2f}\n".format(video_info["duration(s)"]))
        str_out += ("\n")
    
    # if you skipped n frames previously
    if skip_n > 0:
        str_out += ("After skipping {} frames,\n".format(skip_n))   
        
    # if you are displaying prediction information
    if test == False:    
        str_out += ("Statistics on the prediction set are\n")
    
    # if you are displaying test information
    if test == True:    
        str_out += ("Statistics on the test set are\n")
    
    str_out += ("TOTAL NUMBER OF FRAMES PROCESSED: {}\n".format(len(closeness_list)))
    str_out += ("NUMBER OF CLOSED FRAMES: {}\n".format(closeness_list.count(1)))
    str_out += ("NUMBER OF BLINKS: {}\n".format(len(blinks_list)))
    str_out += ("\n")
    
    print(str_out)
    return str_out

# %% [markdown]
# **display stats:**  
# by using display_results() function

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:32:56.210189Z","iopub.execute_input":"2023-06-16T06:32:56.210626Z","iopub.status.idle":"2023-06-16T06:32:56.227440Z","shell.execute_reply.started":"2023-06-16T06:32:56.210573Z","shell.execute_reply":"2023-06-16T06:32:56.225883Z"}}
# first display statistics by using original outputs
scores_string += display_stats(closeness_predictions, blink_predictions, video_info)

# then display statistics by using outputs of skip_first_n_frames() function which are 
#"closeness_predictions_skipped" and "blinks_predictions_skipped"
if(SKIP_FIRST_FRAMES > 0):
    scores_string += display_stats(closeness_predictions_skipped, blink_predictions_skipped, video_info, \
                             skip_n = SKIP_FIRST_FRAMES)

# %% [markdown]
# **We define a function that display blinks found:**  
# 
# You don't need to run this section if you display frames via cv2.imshow() function on offline notebook.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:32:56.229362Z","iopub.execute_input":"2023-06-16T06:32:56.229860Z","iopub.status.idle":"2023-06-16T06:32:56.241028Z","shell.execute_reply.started":"2023-06-16T06:32:56.229796Z","shell.execute_reply":"2023-06-16T06:32:56.239952Z"}}
# display starting, middle and ending frames of all blinks
def display_blinks(blinks, processed_frames):    
    i=1
    # loop over blinks and determine starting, middle and ending frames
    for (frame_start,frame_end) in blinks:
        duration = frame_end - frame_start + 1
        frame_middle = frame_start + int(duration / 2)
        print("{}th blink started at: {}th frame, middle of action at: {}th frame, ended at: {}th frame".format(i,frame_start, frame_middle, frame_end))
        i+=1
        
        # show starting, middle and ending frames
        f, axarr = plt.subplots(1,3,figsize=(15,15))
        img1 = processed_frames[frame_start - 1] # -1 since index starts by 0, frame numbers starts by 1
        img2 = processed_frames[frame_middle - 1] 
        img3 = processed_frames[frame_end - 1] 
        axarr[0].imshow(img1)
        axarr[1].imshow(img2)
        axarr[2].imshow(img3)
        plt.show()
    return

# %% [markdown]
# **Display frames that consist blinks by using the function defined above:**
# 
# If we display all frames of blinks, it will be a huge load so we just display starting, middle and ending frames of blinks, by using display_blinks() fucntions we just wrote.
# 
# **NOTE:** Blink counter on displayed frames doesn't include the blink just happening, so it will always be -1 from the title.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:32:56.242881Z","iopub.execute_input":"2023-06-16T06:32:56.243396Z","iopub.status.idle":"2023-06-16T06:33:20.149927Z","shell.execute_reply.started":"2023-06-16T06:32:56.243319Z","shell.execute_reply":"2023-06-16T06:33:20.148891Z"}}
# display starting, middle and ending frames of all blinks by using "blinks" and "frames"
display_blinks(blink_predictions_skipped, frames_skipped)

# %% [markdown]
# **Read tag file of the video:**
# 
# We will read tag file (annotations made by https://www.blinkingmatters.com/research) to compare our predictions.  
# 
# But for an important note, we will compare closeness of eyes frame by frame, not the blinks for now.
# 
# The annotations starts with line "#start" and rows consist of the following information:  
# > frame ID : blink ID : NF : LE_FC : LE_NV : RE_FC : RE_NV : F_X : F_Y: F_W : F_H : LE_LX : LE_LY : LE_RX : LE_RY: RE_LX : RE_LY : RE_RX : RE_RY
# 
# **frame ID** - Frame counter based on which the time-stamp can be obtained (separate file).  
# **blink ID** - Unique blink ID, eye blink interval is defined as a sequence of the same blink ID frames.  
# **non frontal face (NF)** - While subject is looking sideways and eye blink occurred, given variable changes from X to N.  
# **left eye (LE), right eye (RE), face (F)**  
# **eye fully closed (FC)** - If subject's eyes are closed from 90% to 100%, given flag will change from X to C.  
# **eye not visible (NV)** - While subject's eye is not visible because of hand, bad light conditions, hair or even too fast head movement, this variable changes from X to N.  
# **face bounding box (F_X,F_Y,F_W,F_H)** - x and y coordinates, width, height  
# **left and right eye corners positions** - RX (right corner x coordinate), LY (left corner y coordinate)
# 
# So if a frame consist a blink it will be like:  
# > 2851:9:X:X:X:X:X:240:204:138:122:258:224:283:225:320:226:347:224
# 
# A blink may consist fully closed eyes or not. If it consists fully closed eyes (90% - 100%, this scale is determined by **blinkmatters.com**) it's row will be like:  
# > 2852:9:X:C:X:C:X:239:204:140:122:259:225:284:226:320:227:346:226
# 
# We are just interested in **blink ID** and **eye fully closed (FC)** columns, s we will only read  them, not other information. 

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:33:20.151265Z","iopub.execute_input":"2023-06-16T06:33:20.151804Z","iopub.status.idle":"2023-06-16T06:33:20.164121Z","shell.execute_reply.started":"2023-06-16T06:33:20.151735Z","shell.execute_reply":"2023-06-16T06:33:20.163436Z"}}
# read tag file and construct "closeness_list" and "blinks_list"
def read_annotations(input_file, skip_n = 0):
    # define variables 
    blink_start = 1
    blink_end = 1
    blink_info = (0,0)
    blink_list = []
    closeness_list = []

    # Using readlines() 
    file1 = open(input_file) 
    Lines = file1.readlines() 

    # find "#start" line 
    start_line = 1
    for line in Lines: 
        clean_line=line.strip()
        if clean_line=="#start":
            break
        start_line += 1

    # convert tag file to readable format and build "closeness_list" and "blink_list"
    for index in range(len(Lines[start_line+skip_n : -1])): # -1 since last line will be"#end"
        
        # read previous annotation and current annotation 
        prev_annotation=Lines[start_line+skip_n+index-1].split(':')
        current_annotation=Lines[start_line+skip_n+index].split(':')
        
        # if previous annotation is not "#start" line and not "blink" and current annotation is a "blink"
        if prev_annotation[0] != "#start\n" and prev_annotation[1] == "-1" and int(current_annotation[1]) > 0:
            # it means a new blink starts so save frame id as starting frame of the blink
            blink_start = int(current_annotation[0])
        
        # if previous annotation is not "#start" line and is a "blink" and current annotation is not a "blink"
        if prev_annotation[0] != "#start\n" and int(prev_annotation[1]) > 0 and current_annotation[1] == "-1":
            # it means a new blink ends so save (frame id - 1) as ending frame of the blink
            blink_end = int(current_annotation[0]) - 1
            # and construct a "blink_info" tuple to append the "blink_list"
            blink_info = (blink_start,blink_end)
            blink_list.append(blink_info)
        
        # if current annotation consist fully closed eyes, append it also to "closeness_list" 
        if current_annotation[3] == "C" and current_annotation[5] == "C":
            closeness_list.append(1)
        
        else:
            closeness_list.append(0)
    
    file1.close()
    return closeness_list, blink_list

# %% [markdown]
# **Read tag file:**  
# by using read_annotations() function we just wrote.
# 
# For example you can use "../input/eyeblink/talkingFace/talking.tag" 

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:33:20.165290Z","iopub.execute_input":"2023-06-16T06:33:20.165816Z","iopub.status.idle":"2023-06-16T06:33:20.208394Z","shell.execute_reply.started":"2023-06-16T06:33:20.165754Z","shell.execute_reply":"2023-06-16T06:33:20.207374Z"}}
# full path of a tag file
file_path = "../input/eyeblink/talkingFace/talking.tag"

# read tag file
closeness_test, blinks_test = read_annotations(file_path, skip_n = SKIP_FIRST_FRAMES)

# %% [markdown]
# * **display tag results:**  
# by using display_results() function we just wrote.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:33:20.209727Z","iopub.execute_input":"2023-06-16T06:33:20.210260Z","iopub.status.idle":"2023-06-16T06:33:20.216163Z","shell.execute_reply.started":"2023-06-16T06:33:20.210207Z","shell.execute_reply":"2023-06-16T06:33:20.215301Z"}}
# display results by using outputs of read_annotations() function which are "closeness_test", "blinks_test"
scores_string += display_stats(closeness_test, blinks_test, skip_n = SKIP_FIRST_FRAMES, test = True)

# %% [markdown]
# **Compare calculated blinks to test annotations:**
# 
# For now we will just compare our **basic model**'s predictions (eye closeness values, NOT blinks) to test results by defining a display_test_scores() function.
# 
# Later we will use same function to compare **adaptive model**'s predictions.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:33:20.217372Z","iopub.execute_input":"2023-06-16T06:33:20.217855Z","iopub.status.idle":"2023-06-16T06:33:20.230826Z","shell.execute_reply.started":"2023-06-16T06:33:20.217812Z","shell.execute_reply":"2023-06-16T06:33:20.229599Z"}}
# display test scores and return an "output string" to pass it to writer function 
def display_test_scores(closeness_list_test, closeness_list_pred):
    str_out = ""
    str_out += ("EYE CLOSENESS FRAME BY FRAME TEST SCORES\n")
    str_out += ("\n")

    #print accuracy
    accuracy = accuracy_score(closeness_list_test, closeness_list_pred)
    str_out += ("ACCURACY: {:.4f}\n".format(accuracy))
    str_out += ("\n")

    #print AUC score
    auc = roc_auc_score(closeness_list_test, closeness_list_pred)
    str_out += ("AUC: {:.4f}\n".format(auc))
    str_out += ("\n")

    #print confusion matrix
    str_out += ("CONFUSION MATRIX:\n")
    conf_mat = confusion_matrix(closeness_list_test, closeness_list_pred)
    str_out += ("{}".format(conf_mat))
    str_out += ("\n")
    str_out += ("\n")

    #print FP, FN
    str_out += ("FALSE POSITIVES:\n")
    fp = conf_mat[1][0]
    pos_labels = conf_mat[1][0]+conf_mat[1][1]
    str_out += ("{} out of {} positive labels ({:.4f}%)\n".format(fp, pos_labels,fp/pos_labels))
    str_out += ("\n")

    str_out += ("FALSE NEGATIVES:\n")
    fn = conf_mat[0][1]
    neg_labels = conf_mat[0][1]+conf_mat[0][0]
    str_out += ("{} out of {} negative labels ({:.4f}%)\n".format(fn, neg_labels, fn/neg_labels))
    str_out += ("\n")

    #print classification report
    str_out += ("PRECISION, RECALL, F1 scores:\n")
    str_out += ("{}".format(classification_report(closeness_list_test, closeness_list_pred)))
    
    print(str_out)
    return str_out

# %% [markdown]
# **display test scores:**  
# by using display_test_scores() function we just wrote.

# %% [code] {"scrolled":true,"execution":{"iopub.status.busy":"2023-06-16T06:33:20.232686Z","iopub.execute_input":"2023-06-16T06:33:20.233266Z","iopub.status.idle":"2023-06-16T06:33:20.276904Z","shell.execute_reply.started":"2023-06-16T06:33:20.233033Z","shell.execute_reply":"2023-06-16T06:33:20.275224Z"}}
# display results by using "closeness_predictions", "closeness_test"
scores_string += display_test_scores(closeness_test, closeness_predictions)

# %% [markdown]
# **We define a function that writes output files:**  

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:33:20.278662Z","iopub.execute_input":"2023-06-16T06:33:20.279064Z","iopub.status.idle":"2023-06-16T06:33:20.289963Z","shell.execute_reply.started":"2023-06-16T06:33:20.279007Z","shell.execute_reply":"2023-06-16T06:33:20.288683Z"}}
# write output files of model which are closeness_list, blinks_list and optionally frame_info_df and scores
# if you are writing test dataset results, set test=True so it will change output file's names to "_test" 
# if you want write only scores, not the other files, then set it to scores_only=True
def write_outputs(input_file_name, closeness_list, blinks_list, frame_info_df=None, scores=None, \
                  test=False, scores_only=False):
    # clean filename from path and extensions so you can pass input_file variable to function as it is.
    clean_filename=os.path.basename(os.path.splitext(input_file_name)[0])
    
    # if you are writing prediction outputs
    if test == False and scores_only == False:
        #write all lists to single .h5 file
        with h5py.File("{}_pred.h5".format(clean_filename), "w") as hf:
            g = hf.create_group('pred')
            g.create_dataset('closeness_list',data=closeness_list)
            g.create_dataset('blinks_list',data=blinks_list)
            if frame_info_df is not None:
                frame_info_df.to_parquet('{}_frame_info_df.parquet'.format(clean_filename), engine='pyarrow')
            
    # if you are writing test outputs
    if test == True and scores_only == False:
        #write all lists to single .h5 file
        with h5py.File("{}_test.h5".format(clean_filename), "w") as hf:
            g = hf.create_group('test')
            g.create_dataset('closeness_list',data=closeness_list)
            g.create_dataset('blinks_list',data=blinks_list)
            if frame_info_df is not None:
                frame_info_df.to_parquet('{}_frame_info_df.parquet'.format(clean_filename), engine='pyarrow')

   # if you are writing scores
    if scores != None:
        # use text files this time
        with open("{}_scores.txt".format(clean_filename),"w", encoding='utf-8') as f:
            f.write(scores)
    return

# %% [markdown]
# **write output files:**  
# by using write_outputs() function we just wrote.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:33:20.291403Z","iopub.execute_input":"2023-06-16T06:33:20.291911Z","iopub.status.idle":"2023-06-16T06:33:20.502441Z","shell.execute_reply.started":"2023-06-16T06:33:20.291840Z","shell.execute_reply":"2023-06-16T06:33:20.501415Z"}}
# write prediction output files by using outputs of skip_first_n_frames() function
write_outputs(file_path, closeness_predictions_skipped, blink_predictions_skipped, \
              frame_info_df, scores_string)

# write test output files by using outputs of skip_first_n_frames() function
# no need to write frame_info_df and scores_string since they already have written above
write_outputs(file_path, closeness_test, blinks_test, test = True)

# %% [markdown]
# **We define a function that reads output files to use it later purposes:**  

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:33:20.504420Z","iopub.execute_input":"2023-06-16T06:33:20.504947Z","iopub.status.idle":"2023-06-16T06:33:20.513105Z","shell.execute_reply.started":"2023-06-16T06:33:20.504863Z","shell.execute_reply":"2023-06-16T06:33:20.512081Z"}}
# read output files. 
# if you want to get prediction results, use test=False
# if you want to get test results set test=True
def read_outputs(h5_name, parquet_name=None, test=False):
    # read h5 file by name
    hf = h5py.File('{}.h5'.format(h5_name), 'r')
    
    # if you are reading prediction results
    if test == False:  
        g = hf.get("pred") # read group first   
        
    # if you are reading test results
    if test == True:
         g = hf.get("test") # read group first  
            
    # then get datasets
    closeness_list = list(g.get('closeness_list'))
    blink_list = list(g.get('blinks_list'))

    # if you want to read frame_df_info
    if parquet_name != None:
        frame_info_df = pd.read_parquet('{}.parquet'.format(parquet_name), engine='pyarrow')
        return closeness_list, blink_list, frame_info_df
    
    # if you don't want to read frame_df_info
    else:
        return closeness_list, blink_list

# %% [markdown]
# **We define a function to read all of output files:**

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:33:20.514555Z","iopub.execute_input":"2023-06-16T06:33:20.514994Z","iopub.status.idle":"2023-06-16T06:33:20.524788Z","shell.execute_reply.started":"2023-06-16T06:33:20.514918Z","shell.execute_reply":"2023-06-16T06:33:20.523895Z"}}
# load all of output files. 
def load_datasets(path, dataset_name):
    # build  full path
    full_path = os.path.join(path, dataset_name)
    
    # read prediction results and frame_info_df
    closeness_pred, blinks_pred, frame_info_df \
                = read_outputs("{}_pred".format(full_path),"{}_frame_info_df".format(full_path))

    # read test results
    closeness_test, blinks_test = read_outputs("{}_test".format(full_path), test = True)
    
    # read scores
    with open("{}_scores.txt".format(full_path),"r") as f:
        Lines = f.readlines() 
        # build a string that hold scores
        scores_str = ""
        for line in Lines: 
            scores_str += line

    return  closeness_pred, blinks_pred, frame_info_df, closeness_test, blinks_test, scores_str

# %% [markdown]
# **Load datasets by reading all of output files:**
# 
# We will call this function on next notebooks. This way, by reading outputs of previous notebooks, we will build a basic pipeline.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:33:20.526202Z","iopub.execute_input":"2023-06-16T06:33:20.526560Z","iopub.status.idle":"2023-06-16T06:33:21.570088Z","shell.execute_reply.started":"2023-06-16T06:33:20.526509Z","shell.execute_reply":"2023-06-16T06:33:21.569363Z"}}
# load datasets
c_pred, b_pred, df, c_test, b_test, s_str= load_datasets("../working","talking")

# check results
print(np.array(c_pred).shape, np.array(b_pred).shape)
print(np.array(c_test).shape, np.array(b_test).shape)
print()
print(s_str)
df

# %% [markdown]
# **And finally let's put it all together to build a model:**

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:33:21.571480Z","iopub.execute_input":"2023-06-16T06:33:21.571918Z","iopub.status.idle":"2023-06-16T06:33:21.583772Z","shell.execute_reply.started":"2023-06-16T06:33:21.571815Z","shell.execute_reply":"2023-06-16T06:33:21.582628Z"}}
# build simple_model pipeline
# if you want to display blinks set display_blinks=True (it requires long time and memory so default is False)
# if you want to read annotation file and run comparison metrics set test_extention="tag" or any file extension
# REMARK: your annotation file an video file must have the same name to use this function.
# if you want to write outputs set write_results=True
def simple_model(input_full_path, ear_th=0.21, consec_th=3, skip_n = 0, \
                    display_blinks=False, test_extention=False, write_results=False):
    # define variables
    scores_string = ""
    
    # process the video and get the results
    frame_info_df, closeness_predictions, blink_predictions, frames, video_info, scores_string \
        = process_video(input_full_path, ear_th=ear_th, consec_th=consec_th)
    
    # recalculate data by skipping "skip_n" frames
    frame_info_df, closeness_predictions_skipped, blink_predictions_skipped, frames_skipped \
        = skip_first_n_frames(frame_info_df, closeness_predictions, blink_predictions, frames, \
                              skip_n = skip_n)

    # first display statistics by using original outputs
    scores_string += display_stats(closeness_predictions, blink_predictions, video_info)

    # then display statistics by using outputs of skip_first_n_frames() function which are 
    #"closeness_predictions_skipped" and "blinks_predictions_skipped"
    if(skip_n > 0):
        scores_string += display_stats(closeness_predictions_skipped, blink_predictions_skipped, video_info, \
                                 skip_n = skip_n)
    
    # if you want to display blinks
    if display_blinks == True:
        # display starting, middle and ending frames of all blinks by using "blinks" and "frames"
        display_blinks(blink_predictions_skipped, frames_skipped)
        
    # if you want to read tag file
    if test_extention != False:
        extention = ""
        # default file extension is ".tag"
        if test_extention == True:
            extention = "tag"
        else:
            extention = test_extention
        # remove video extention i.e. ".avi"
        clean_path = os.path.splitext(input_full_path)[0]
        # read tag file
        closeness_test, blinks_test = read_annotations("{}.{}".format(clean_path, extention), skip_n = skip_n)
        # display results by using outputs of read_annotations() function 
        # which are "closeness_test", "blinks_test"
        scores_string += display_stats(closeness_test, blinks_test, skip_n = skip_n, test = True)
        # display results by using "closeness_test" and "closeness_predictions"
        scores_string += display_test_scores(closeness_test, closeness_predictions_skipped)
        
    # if you want to write results
    if write_results == True:
        # write prediction output files by using outputs of skip_first_n_frames() function
        write_outputs(input_full_path, closeness_predictions_skipped, blink_predictions_skipped, \
                      frame_info_df, scores_string)
        if test_extention != False:
            # write test output files by using outputs of skip_first_n_frames() function
            # no need to write frame_info_df and scores_string since they already have written above
            write_outputs(input_full_path, closeness_test, blinks_test, \
                          test = True)
            
    return frame_info_df, closeness_predictions_skipped, blink_predictions_skipped, frames_skipped, \
            video_info, scores_string

# %% [markdown]
# **Let's test the function simple_model():**
# 
# We can use one of the videos of other dataset (eyeblink8)  
# i.e "../input/eyeblink/eyeblink8/8/27122013_151644_cam.avi".  
# This may take a long time like 10-15 mins due to it's large size.
# 
# **NOTE TO MYSELF:** eyeblink8/1 doesn't work. There are inconsistency on frame counts between test and pred results. check version 16. eyeblink8/8 works well. others also need to be checked.

# %% [code] {"execution":{"iopub.status.busy":"2023-06-16T06:33:21.585064Z","iopub.execute_input":"2023-06-16T06:33:21.585389Z","iopub.status.idle":"2023-06-16T06:40:47.983916Z","shell.execute_reply.started":"2023-06-16T06:33:21.585338Z","shell.execute_reply":"2023-06-16T06:40:47.982986Z"}}
# test the function above
fdf, cp, bp, fr, vi, st = simple_model("../input/eyeblink/eyeblink8/8/27122013_151644_cam.avi", \
                                    test_extention = "tag", write_results = True)