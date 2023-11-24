import cv2
import torch
from ultralytics import YOLO
from ultralytics import RTDETR
import json
import numpy
import time
import pandas as pd

from utils import inPolygon, crop_image, calculate_rectangle_area
from config import STEP, video_paths, text_paths

from ..mobile_net.classificator import classify_image


# Load the YOLOv8 model
# model = YOLO('yolov8x.pt')
model = RTDETR('rtdetr-l.pt').to('cuda:0')
model.fuse()
# Open the video file
#video_path = rf"./home/nailmarsel/Documents/KRA-44-169-2023-08-17-evening.mp4"


file_name = []
car = []
quantity_car = []
average_speed_car = []
van = []
quantity_van = []
average_speed_van = []
bus = []
quantity_bus = []
average_speed_bus = []

for video_path, text_path in zip(video_paths, text_paths):
    print('EXECUTE FILE: ', video_path)
    cap = cv2.VideoCapture(video_path)
    with open(text_path) as f:
        data = json.load(f)
    areas=data['areas']
    zones_1=numpy.array(data['zones'][0])
    zones_2=numpy.array(data['zones'][1])
    # print(data['zones'])
    zones_1_1=zones_1[:,0]
    # print(zones_1_1)
    zones_2_1=zones_2[:,0]
    zones_1_2=zones_1[:,1]
    zones_2_2=zones_2[:,1]
    # Loop through the video frames
    counter = 0
    speed_count={} #'id':['start','stop']
    class_count={} #'id':['start','stop']
    flags={}
    fps = cap.get(cv2.CAP_PROP_FPS)

    while cap.isOpened():
        # if counter % 1000:
        #     print(counter)
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            # if counter==5000:
            #     break
            if counter % STEP == 0:
                # Run YOLOv8 tracking on the frame, persisting tracks between frames
                # inp = torch.Tensor(frame).to('cuda').view()
                results = model.track(frame, persist=True, imgsz=(416, 416), classes=[2, 5, 7], device='cuda:0', conf=0.41)
                annotated_frame = results[0].plot(line_width=2)
                for i in range(len(results[0].boxes.cls)):
                    box = results[0].boxes.xyxy[i]
                    cv2.circle(annotated_frame, (int((box[0]+box[2])//2), int((box[1]+box[3])//2)), 10, (0, 0, 255), -1)
                    h, w = results[0].boxes.orig_shape
                    id = int(results[0].boxes.id[i].tolist())
                    cls = int(results[0].boxes.cls[i].tolist())

                    if cls == 5:  # it is the bus
                        box = results[0].boxes.xyxy[i]
                        x_left, y_left, x_right, y_right = list(box)
                        bus_frame = crop_image(
                            frame,
                            (int(x_left), int(y_left)),
                            (int(x_right), int(y_right))
                        )
                        bus_frame_area = calculate_rectangle_area(x_left, y_left, x_right, y_right)
                        # TODO На класифкацию мы должны отдавать самый большой по площади frame, причем
                        # Данная логика должна происходить, пока машина находится в полигоне

                        # TODO Зейчас захаркодим, и будем отдавать каждый кард на классификацию
                        bus_class = classify_image(bus_frame)
                        # TODO Обработка если автобус ... иначе ...

                    if id not in speed_count.keys():
                        speed_count[id]=[-1,-1]
                        flags[id] = 'False'
                    x=speed_count[id][0]
                    for area in areas:
                        if inPolygon((box[0]+box[2])/2, (box[1]+box[3])/2, numpy.array(area)[:,0], numpy.array(area)[:,1], w, h) and speed_count[id][0]==-1 and flags[id]=='False':
                            speed_count[id][0]=counter
                            flags[id]=area
                        elif not inPolygon((box[0]+box[2])/2, (box[1]+box[3])/2, numpy.array(area)[:,0], numpy.array(area)[:,1], w, h) and speed_count[id][1]==-1 and flags[id]==area:

                            speed_count[id][1]=counter
                            flags[id]='None'
                            break
                        if id not in class_count.keys():
                            class_count[id] = []
                        class_count[id].append(cls)

                        cv2.circle(
                            annotated_frame, (int(area[0][0] * 1920), int(area[0][1] * 1080)), 10, (255, 255, 0), -1)
                        cv2.circle(
                            annotated_frame, (int(area[1][0] * 1920), int(area[1][1] * 1080)), 10, (255, 255, 0), -1)
                        cv2.circle(
                            annotated_frame, (int(area[2][0] * 1920), int(area[2][1] * 1080)), 10, (255, 255, 0), -1)
                        cv2.circle(
                            annotated_frame, (int(area[3][0] * 1920), int(area[3][1] * 1080)), 10, (255, 255, 0), -1)


                # Visualize the results on the frame
                # Display the annotated frame
                # dim = (int(1920 * 0.5), int(1080 * 0.5))
                # # print(frame_width, frame_height)
                # # print(dim)
                # # resize image
                #
                # annotated_frame = cv2.resize(annotated_frame, dim, interpolation=cv2.INTER_AREA)
                # cv2.imshow("YOLOv8 Tracking", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            counter += 1
        else:
            # Break the loop if the end of the video is reached
            break
    print(class_count,speed_count)
    print('---------------------')

    classes=[0,0,0]
    mean={'car':[],'bus':[],'truck':[]}
    for key in class_count.keys():
        classe=max(set(class_count[key]), key=class_count[key].count)
        if classe==2 and speed_count[key][1]!=-1 and speed_count[key][0]!=-1:
            classes[0]+=1
            mean['car'].append((3.6 * 20) / ((speed_count[key][1]-speed_count[key][0]) / fps))
        elif classe==5 and speed_count[key][1]!=-1 and speed_count[key][0]!=-1:
            classes[1]+=1
            mean['bus'].append((3.6 * 20) / ((speed_count[key][1]-speed_count[key][0]) / fps))

        elif speed_count[key][1]!=-1 and speed_count[key][0]!=-1:
            classes[2]+=1
            mean['truck'].append((3.6 * 20) / ((speed_count[key][1]-speed_count[key][0]) / fps))

    print(mean)
    print(classes)
    car_mean = (sum(mean['car'])/len(mean['car']))
    bus_mean = (sum(mean['bus'])/len(mean['bus']))
    van_mean = (sum(mean['truck'])/len(mean['truck']))
    print(car_mean)
    print(bus_mean)
    print(van_mean)

    file_name.append(text_path[:text_path.find('.')])
    car.append('car')
    quantity_car.append(classes[0])
    average_speed_car.append(car_mean)
    van.append('van')
    quantity_van.append(classes[2])
    average_speed_van.append(van_mean)
    bus.append('bus')
    quantity_bus.append(classes[1])
    average_speed_bus.append(bus_mean)
    # Release the video capture object and close the display window
    # cap.release()
    # cv2.destroyAllWindows()

pd.DataFrame({'file_name': file_name, 'car': car, 'quantity_car': quantity_car,
              'average_speed_car': average_speed_car, 'van': van, 'quantity_van': quantity_van,
              'average_speed_van': average_speed_van, 'bus': bus, 'quantity_bus': quantity_bus,
              'average_speed_bus': average_speed_bus}).to_csv('result1.csv')