import pandas as pd
import numpy as np
import colorsys
import argparse
import imutils
import random
import cv2
import os
import time
from google.colab.patches import cv2_imshow
from google.colab import files
from PIL import Image, ImageDraw, ImageFont
from datetime import datetime
import io
import pandas as pd
import matplotlib.pyplot as plt
import h5py

import tensorflow as tf

def load_gaze_data(path, gaze_filename):

  """ Loads in the gaze data. Returns list for X and Y coordinates """

  raw_gaze = open(os.path.join(path, gaze_filename)).read().strip().split("\n")

  raw_gaze.pop(0) # remove the x and y
  GX = [round(float(pair.split(",")[0])) for pair in raw_gaze]
  GY = [round(float(pair.split(",")[1])) for pair in raw_gaze]
  return GX, GY

def load_video(path, video_filename):
  """ Loads in the video and returns a frame-by-frame video variable vs. Also returns video output path"""
  vid_input = os.path.join(path, video_filename)
  vid_output = os.path.join(path, "output_" + video_filename)

  #Initialise the video stream and pointer to output video file
  vs = cv2.VideoCapture(vid_input)
  writer = None

  return vs, vid_output

def check_frames(GX, GY, vs):
  """ Check that the number of frames in which the eye tracker failed to collect any data isn't too high"""

  # Count number of frames in video
  prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() else cv2.CAP_PROP_FRAME_COUNT
  total = int(vs.get(prop))
  print("[INFO] {} total frames in video".format(total))

  # compare to number of gaze datapoints to check the gaze tracker didn't miss to many
  dropped_frames = total - len(GX)

  if dropped_frames < 10:
    [GX.append(0) for i in range(0,dropped_frames)]
    [GY.append(0) for i in range(0,dropped_frames)]
    print('[INFO] The eye tracker dropped {} frames. {} 0s have been added to each gaze coordinate list.'.format(dropped_frames, dropped_frames))

  elif dropped_frames > 10:
    raise Exception('The number of dropped frames is too high: {}'.format(dropped_frames))
  

def draw_mrcnn_output(frame, startX, startY, endX, endY, color, label, score):
  
  """Overlay mrcnn bounding boxes and labels over frames"""
  
  cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
  text = "{}: {:.3f}".format(label, score)
  y = startY - 10 if startY - 10 > 10 else startY + 10
  cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX,
    0.6, color, 2)   

def apply_green_overlay(frame, gmask, greenery_score):

    """Add a blue mask over the areas of the frame identified as natural greenery"""
    
    overlay = frame.copy()
    grindex = np.where(gmask == 255) # Find the pixels that are green
    overlay[grindex[0][:],grindex[1][:],0] = 250 # Change these pixels in the overlay - make blue value high
    overlay[grindex[0][:],grindex[1][:],1] = 20 # Green value low
    overlay[grindex[0][:],grindex[1][:],2] = 20 # Red value low
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0 , frame)

    if len(grindex[0][:]):
      temp_greenery_score = (len(grindex[0][:]) / (720*1280)) * 100
      greenery_score.append(temp_greenery_score) #  This stores the percentage of the given frame that was identified as green. 

    return greenery_score

 
def identify_inframe_objects(GY, GX, count, mask, inframe_gaze_checklist, confidence_list, inframe_object_loc_X,\
                          inframe_object_loc_Y, startX, startY, classID, CLASS_NAMES):
 

    """Identify object labels within given frames"""

    if GY[count] > 0 and GY[count] < 720 and GX[count] > 0 and GX[count] < 1280: # If the gaze fell within the headview camera's boundaries
        if mask[GY[count],GX[count]]: # == True:
            #label_winner = classID
            inframe_gaze_checklist.append(CLASS_NAMES[classID]) # inframe_gaze_checklist is the objects appearing in the given frame
            confidence_list.append(2)
            inframe_object_loc_X.append(startX)
            inframe_object_loc_Y.append(startY)

        else:
            #label_winner = 'Background'
            inframe_gaze_checklist.append(0)
            confidence_list.append(0)
            inframe_object_loc_X.append(0)
            inframe_object_loc_Y.append(0)
        
    else: # If gaze did not fall within the head view camera boundaries then add 'outofbounds' to the gazed_upon_object_list list
        label_winner = 'Out Of Bounds'
        inframe_gaze_checklist.append('OOB')
        confidence_list.append(0)
        inframe_object_loc_X.append(0)
        inframe_object_loc_Y.append(0)

    return inframe_gaze_checklist, confidence_list, inframe_object_loc_X, inframe_object_loc_Y


def get_gazed_upon_object(inframe_gaze_checklist, confidence_list, inframe_object_loc_X, inframe_object_loc_Y, gazed_upon_object_list, confidence, gmask):

  gazed_upon_index = [i for i, e in enumerate(inframe_gaze_checklist) if e != 0] # gazed_upon_index is the index of the winning classID (if there is one)


  if len(gazed_upon_index) > 0: # If there is a single winning object then add the winner to the gazed_upon_object_list list    
    new_champ = inframe_gaze_checklist[gazed_upon_index[0]]  
    current_startX = inframe_object_loc_X[gazed_upon_index[0]] # This defines the current location of the gazed upon object
    current_startY = inframe_object_loc_Y[gazed_upon_index[0]]
    
    if new_champ == 59: # if it's a potted plant, then mark as green
      gazed_upon_object_list.append('Greenery')
    else: # If the recognised object is not a potted plant, then add that classID to the gazed_upon_object_list list
      gazed_upon_object_list.append(new_champ)

  elif len(gazed_upon_index) == 0: # If no winning object was found, then check if green is being looked at
    
    if GY[count] > 0 and GY[count] < 720 and GX[count] > 0 and GX[count] < 1280: # If the gaze fell within the headview camera's boundaries 
      if gmask[GY[count],GX[count]] == 255: # If the gaze coords falls on a green area (The mask is flipped - so is GY,GX)
        new_champ = 'Greenery'
        gazed_upon_object_list.append('Greenery') # 82 is greenery
        confidence.append(0)
        
      elif gmask[GY[count],GX[count]] == 0:
        new_champ = 'Background'
        gazed_upon_object_list.append('Background')
        confidence.append(0)

    else:
      new_champ = 'OOB'

  return gazed_upon_object_list, confidence, new_champ


def overlay_label(CLASS_NAMES, new_champ, frame):
  # Draw the Text on top
  font = cv2.FONT_HERSHEY_SIMPLEX 
  org = (50, 50)       
  fontScale = 2      
  color = (0, 0, 255)  # Red     
  thickness = 2      
  # winner_label = CLASS_NAMES[new_champ] # text to draw
  image = cv2.putText(frame, new_champ, org, font,  
                    fontScale, color, thickness, cv2.LINE_AA) 
  return image


def overlay_gaze_cursor(GX, GY, count, frame):
  
  """Add overlay representing gaze location"""
  
  center_coordinates = (GX[count], GY[count]) 
  radius = 30 
  color = (0, 0, 255) 
  thickness = -1
    
  # Using cv2.circle() method 
  # Draw a circle of red color of thickness -1 px 
  image = cv2.circle(frame, center_coordinates, radius, color, thickness) 
  return image

def print_user_info(count, total, end, start):
  """Print timing information at given iterations"""
  
  if count == 8:
    elap = (end - start)
    print("[INFO] single frame took {:.4f} seconds".format(elap))
    print("[INFO] estimated total time to finish: {:.4f}".format((elap * (total/4))))    
  elif count == round(total/8): #round(quartal/2):
    print("[INFO] Halfway!")