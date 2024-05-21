from ultralytics import YOLO
import cv2
import numpy as np
import os
import imutils
from collections import defaultdict
import time
import uuid

def is_wrong_direction(track_history, track_id, direction, wrong_count):    
    current_y = track_history[track_id][-1][1]
    if len(track_history[track_id])<2:
        previous_y = track_history[track_id][0][1]
    else: 
        previous_y = track_history[track_id][-2][1]
    #print(current_y,previous_y)
    wrong_direction = (direction == 'top-to-bottom' and current_y < previous_y) or (direction == 'bottom-to-top' and current_y > previous_y)
    if wrong_direction == True:
        wrong_count[track_id] += 1
    #print(track_id,wrong_count[track_id])
    if wrong_count[track_id] > 10:
        return True
    return False

def distance(previous_center,current_center):
    distance = np.linalg.norm(np.array(current_center) - np.array(previous_center))
    return distance
  
def is_stop_or_stall(track_history, track_id, stop_count, stall_count, stall_threshold, frame_counter):
    current_center=track_history[track_id][-1]
    if frame_counter % 10 == 0 :
        if len(track_history[track_id])<10:
            previous_center = track_history[track_id][0]
        else: 
            previous_center = track_history[track_id][-10]
        dis =  distance(previous_center,current_center)  
        if dis == 0:
            stop_count[track_id] += 1
        if dis < stall_threshold:
            stall_count[track_id] += 1
    return stop_count,stall_count

def copy_image_with_box(img,scaled_x_min, scaled_y_min,scaled_x_max, scaled_y_max, color):
    copy_img=img.copy()
    cv2.rectangle(copy_img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), color, 2)
    return copy_img

def copy_image_without_box(img,scaled_x_min, scaled_y_min,scaled_x_max, scaled_y_max):
    copy_img=img.copy()
    return copy_img    

#zone1
detection_area_z1= [(185, 28), (276, 29), (181, 181), (1, 123)]            #for vehicles
detection_area_z1_2 = [(218, 29), (255, 29), (140,168), (65, 144)]         #for animals and padestrians
direction_z1='bottom-to-top'

#zone2
detection_area_z2 = [(296, 38), (369, 51), (399, 169), (250, 184)]         #for vehicles
detection_area_z2_2 = [(308, 42), (334, 47), (366, 172), (290, 181)]       #for animals and padestrians
direction_z2='top-to-bottom'

#zone3                                                                     #only wrong side
detection_area_z3=[(139, 7), (161, 19), (3, 73), (1, 43)]    
direction_z3='bottom-to-top'

main_stream = "rtsp://admin:123456@192.168.1.133/stream0"
images_path = "./image_"
sub_video_path = "./video_"
vids_model = "vids.pt"
saved_images = defaultdict(bool)
saved_videos=defaultdict(bool)
model = YOLO(vids_model, task="detect")
cap = cv2.VideoCapture(main_stream)
track_history = defaultdict(list)
frame_counter= 0
wrong_count=defaultdict(int)
stop_count=defaultdict(int)
stall_count=defaultdict(int)
stall_threshold = 10
frames_to_save = [] 
frames_to_print = []
max_length_frames_to_save = 60         
event_counter = 0
record_length = 20
flag = False

while True:
    ret, img = cap.read()
    if not ret:
        break
    frame = imutils.resize(img,400)
    frame_counter += 1
    width_ratio = img.shape[1] / frame.shape[1]
    height_ratio = img.shape[0] / frame.shape[0]
    results = model.track(frame, conf=0.45, persist=True, stream=True, verbose=False, half=False )
    vehicle_history = defaultdict(list)

    for result in results:
        for box in result.boxes:
            if box.id is not None:
                track_id = int(box.id)
            else:
                track_id = 0  
            class_id = int(box.cls[0].item())

            if class_id in (0,1,2,3,4,5,6):           #(for vehicle, human, animal, debris, fire, accident and non-accident) 
                cords = box.xyxy[0].tolist()
                cords = [round(x) for x in cords]
                x_min, y_min, x_max, y_max = cords
                scaled_x_min = int(x_min * width_ratio)
                scaled_y_min = int(y_min * height_ratio)
                scaled_x_max = int(x_max * width_ratio)
                scaled_y_max = int(y_max * height_ratio)
                frames_to_save.append(copy_image_without_box(img,scaled_x_min, scaled_y_min,scaled_x_max, scaled_y_max))
                if len(frames_to_save) > max_length_frames_to_save:
                    frames_to_save = frames_to_save[-max_length_frames_to_save:]
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                in_detection_z1 = cv2.pointPolygonTest(np.array(detection_area_z1, np.int32), (center_x, center_y), False)
                in_detection_z2 = cv2.pointPolygonTest(np.array(detection_area_z2, np.int32), (center_x, center_y), False)
                in_detection_z3 = cv2.pointPolygonTest(np.array(detection_area_z3, np.int32), (center_x, center_y), False)
                in_detection_z1_2 = cv2.pointPolygonTest(np.array(detection_area_z1_2, np.int32), (center_x, center_y), False)
                in_detection_z2_2 = cv2.pointPolygonTest(np.array(detection_area_z1_2, np.int32), (center_x, center_y), False)

                if class_id in (0,1,2,3):                                     #(for vehicle, human, animal and debris)
                    if ( in_detection_z1 >= 0 or in_detection_z2 >= 0 or in_detection_z3 >= 0 ) and track_id != 0:  
                        if in_detection_z1 >= 0 :
                            direction = direction_z1
                            zone = 'zone1'
                        if in_detection_z2 >= 0 :
                            direction = direction_z2
                            zone = 'zone2'
                        if in_detection_z3 >= 0 :
                            direction = direction_z3
                            zone = 'zone3'
                        track_history[track_id].append((center_x, center_y))

                        if class_id == 0:                                                           #(For Vehicles)     
                            vehicle_history[track_id].append((x_min,x_max,y_min,y_max,class_id))
                            if track_id not in wrong_count :
                                wrong_count[track_id]=0
                            if track_id not in stop_count :
                                stop_count[track_id]=0
                            if track_id not in stall_count :
                                stall_count[track_id]=0
                            if is_wrong_direction(track_history, track_id, direction, wrong_count):
                                event_counter += 1
                                color=(0, 0, 255)                                                  #Red (WRONG SIDE)
                                frames_to_save=frames_to_save[:-2]
                                frames_to_save.append(copy_image_with_box(img,scaled_x_min, scaled_y_min,scaled_x_max, scaled_y_max, color))
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                                if not saved_images[track_id]:
                                    frames_to_print=frames_to_save[-30:]
                                    frames_to_save=[]
                                    flag = True
                                    saved_images[track_id] = True  
                                    curr_uuid = uuid.uuid4()
                                    file_name = f"ws{curr_uuid}.jpg"
                                    cv2.rectangle(img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), color, 2)
                                    cv2.imwrite(os.path.join(images_path, file_name), img)
                                    print("Vehicle Moving WRONG SIDE", ", saved", file_name)
                            if zone == 'zone1' or zone == 'zone2':
                                stop_count,stall_count = is_stop_or_stall(track_history, track_id, stop_count, stall_count, stall_threshold, frame_counter)
                                if stall_count[track_id] > 5:                
                                    color=(0, 255, 255)                                               #Yellow (HALT)
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                                    print("Vehicle STALLED")
                                if stop_count[track_id] > 1:
                                    event_counter += 1
                                    color = (255, 0, 0)                                                  #Blue (STOP)
                                    frames_to_save=frames_to_save[:-2]
                                    frames_to_save.append(copy_image_with_box(img,scaled_x_min, scaled_y_min,scaled_x_max, scaled_y_max, color))
                                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                                    if not saved_images[track_id]:
                                        frames_to_print=frames_to_save[-30:]
                                        frames_to_save=[]
                                        flag = True
                                        saved_images[track_id] = True 
                                        curr_uuid = uuid.uuid4()
                                        file_name = f"stp{curr_uuid}.jpg"
                                        cv2.rectangle(img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), color, 2)
                                        cv2.imwrite(os.path.join(images_path, file_name), img)
                                        print("Vehicle STOP", ", saved", file_name)   

                        if class_id == 3:                                                                 #(for Debris)
                            color = (0, 255, 0)                                                           #Green (DEBRIS)
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                            if not saved_images[track_id]:                                        
                                saved_images[track_id] = True 
                                curr_uuid = uuid.uuid4()                                 
                                file_name = f"db{curr_uuid}.jpg"
                                cv2.rectangle(img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), color, 2)                                
                                cv2.imwrite(os.path.join(images_path, file_name), img)
                                print("Debris on Road", ", saved", file_name)

                    if ( in_detection_z1_2 >= 0 or  in_detection_z2_2 >= 0 ) and track_id != 0: 
                        track_history[track_id].append((center_x, center_y))    
                        if class_id in (1,2):                                               #(For Pedestrians and Animals)
                            inside=0
                            for key in vehicle_history:
                                out_points=list(vehicle_history[key][0])
                                box_points = np.array([[out_points[0], out_points[3]], [out_points[1], out_points[3]], [out_points[1], out_points[2]], [out_points[0], out_points[2]]], dtype=np.float32)
                                in_point = (center_x,center_y)
                                inside = cv2.pointPolygonTest(box_points, in_point, False)
                            if not (inside == 1) or len(vehicle_history) == 0:
                                if class_id == 1:
                                    color=(0, 128, 225)                                             #Orange (Pedestrian)
                                else:
                                    color=(255, 51, 255)                                                 #Pink (Animals)
                                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                                if not saved_images[track_id]:                                        
                                    saved_images[track_id] = True 
                                    curr_uuid = uuid.uuid4()     
                                    if class_id == 1:
                                        file_name = f"hmn{curr_uuid}.jpg"
                                    else:
                                        file_name = f"anm{curr_uuid}.jpg"
                                    cv2.rectangle(img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), color, 2)                                
                                    cv2.imwrite(os.path.join(images_path, file_name), img)
                                    if class_id == 1:
                                        print("Pedestrian Crossing", ", saved", file_name)
                                    else:
                                        print("Animal Crossing", ", saved", file_name)
                    
                if class_id in (4,5,6):                                          #(for fire, accident and non-accident)
                    if class_id == 4:                                                                       #(for fire)
                        color = (255, 255, 51)                                                       #Light Blue (FIRE)
                        if not saved_images[track_id]:  
                            frames_to_print=frames_to_save[-30:]
                            frames_to_save=[]
                            flag = True                                      
                            saved_images[track_id] = True 
                            curr_uuid = uuid.uuid4()                                 
                            file_name = f"fr{curr_uuid}.jpg"
                            cv2.rectangle(img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), color, 2)                                
                            cv2.imwrite(os.path.join(images_path, file_name), img)
                            print("Fire Detected", ", saved", file_name)
                    if class_id == 5:                                                                  #(for accident)      
                        color = (160, 160, 160)                                                       #Gray (ACCIDENT)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        if not saved_images[track_id]:                                        
                            frames_to_print=frames_to_save[-30:]
                            frames_to_save=[]
                            flag = True                                      
                            saved_images[track_id] = True 
                            curr_uuid = uuid.uuid4()                                 
                            file_name = f"acc{curr_uuid}.jpg"
                            cv2.rectangle(img, (scaled_x_min, scaled_y_min), (scaled_x_max, scaled_y_max), color, 2)                                
                            cv2.imwrite(os.path.join(images_path, file_name), img)
                            print("Accident Spotted", ", saved", file_name)                           
                    if class_id == 6:                                                              #(for non-accident)      
                        color = (160, 160, 160)                                                      #White (ACCIDENT)
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
                        print("Non-Accident")    
                                                                                                                                                          
            #time.sleep(0.02)
        
    if event_counter > record_length and flag == True:
        frames_to_print += frames_to_save
        if not saved_videos[track_id]:
            saved_videos[track_id]=True
            if len(frames_to_print) > 0:  
                fourcc = cv2.VideoWriter_fourcc(*'XVID')  
                #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_filename = f"{sub_video_path}/{uuid.uuid4()}.avi"  
                size=frames_to_print[0].shape[:2]
                video_writer = cv2.VideoWriter(video_filename, fourcc, fps=cap.get(cv2.CAP_PROP_FPS), frameSize=(size[1],size[0]))
                for frm in frames_to_print:
                    video_writer.write(frm)
                video_writer.release()
                print(f"Saved sub-video: {video_filename}")
                frames_to_print = []
                event_counter = 0
                flag = False

    if cv2.waitKey(1)== ord("q"):
        break

    cv2.polylines(frame, [np.array(detection_area_z1_2 , np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(detection_area_z1 , np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(detection_area_z2_2 , np.int32)], True, (0, 255, 0), 2)
    cv2.polylines(frame, [np.array(detection_area_z2 , np.int32)], True, (255, 0, 0), 2)
    cv2.polylines(frame, [np.array(detection_area_z3 , np.int32)], True, (255, 0, 0), 2)

    cv2.imshow("Output", frame)

cap.release()
cv2.destroyAllWindows()
