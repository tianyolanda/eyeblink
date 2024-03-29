# utility functions
# based on eye_blink_detection_1_simple_model.ipynb

# import packages
from scipy.spatial import distance as dist
import numpy as np
import pandas as pd
import dlib
import cv2
import os
import time
import h5py
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report, roc_auc_score
import matplotlib.pyplot as plt

# initialize dlib variables
dlib_detector = dlib.get_frontal_face_detector()
dlib_predictor = dlib.shape_predictor("../input/eyeblink/shape_predictor_68_face_landmarks.dat")

############################################################################################################

# print content of any folder
def display_folder(path):
    for dirname, _, filenames in os.walk(path):
        for filename in filenames:
            print(os.path.join(dirname, filename))
            
############################################################################################################

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

############################################################################################################

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

############################################################################################################

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

############################################################################################################

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

############################################################################################################

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

############################################################################################################

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

############################################################################################################

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

############################################################################################################

# write output files of model which are closeness_list, blinks_list and optionally frame_info_df and scores
# if you are writing test dataset results, set test=True so it will change output file's names to "_test"
# if you want to write scores also, set write_scores to True, 
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

############################################################################################################

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
    
############################################################################################################

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

############################################################################################################

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
