import os
import numpy as np
import imutils
import dlib
import time
import cv2
import json
from flask import Flask, flash, request, redirect, url_for, render_template,jsonify
from werkzeug.utils import secure_filename

from imutils.video import VideoStream
from imutils.video import FPS

UPLOAD_FOLDER = 'dataset/'
# ALLOWED_EXTENSIONS = set(['.png', '.jpg', '.jpeg', '.gif', '.mp4'])

backup_folder = 'dataset_backup/'
files = os.listdir(backup_folder)
for f in files:
    file_name = backup_folder+f
    os.remove(file_name)

maximumTotal=[0]
trackers = []
trackableObjects = {}

def showDetection(vd):
    # load the COCO class labels our YOLO model was trained on
    # labelsPath = os.path.sep.join( ["yolo", "coco.names"] )
    labelsPath= r"yolo/coco.names"
    LABELS = open( labelsPath ).read().strip().split( "\n" )
    totalFrames = 0
    # initialize a list of colors to represent each possible class label
    np.random.seed( 42 )
    COLORS = np.random.randint( 0, 255, size=(len( LABELS ), 3),
                                dtype="uint8" )

    # derive the paths to the YOLO weights and model configuration
    weightsPath = r"yolo/yolov3.weights"
    configPath = r"yolo/yolov3.cfg"

    # load our YOLO object detector trained on COCO dataset (80 classes)
    # and determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet( configPath, weightsPath )
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # initialize the video stream, pointer to output video file, and
    # frame dimensions
    vs = cv2.VideoCapture( vd )
    writer = None
    (W, H) = (None, None)

    # try to determine the total number of frames in the video file
    try:
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2() \
            else cv2.CAP_PROP_FRAME_COUNT
        total = int( vs.get( prop ) )
        print("[INFO] {} total frames in video".format( total ))

    # an error occurred while trying to determine the total
    # number of frames in the video file
    except:
        print("[INFO] could not determine # of frames in video")
        print("[INFO] no approx. completion time can be provided")
        total = -1

    # loop over frames from the video file stream
    for i in range(total):
        if i % 27 == 0:
            car = 0
            truck = 0
            person = 0
            bus = 0
            motorbike = 0
            bicycle = 0
            # read the next frame from the file
            (grabbed, frame) = vs.read()

            # if the frame was not grabbed, then we have reached the end
            # of the stream
            if not grabbed:
                break
            rgb = cv2.cvtColor( frame, cv2.COLOR_BGR2RGB )

            if totalFrames % 12 != 0:

                totalFrames += 1
            else:
                # if the frame dimensions are empty, grab them
                if W is None or H is None:
                    (H, W) = frame.shape[:2]

                # construct a blob from the input frame and then perform a forward
                # pass of the YOLO object detector, giving us our bounding boxes
                # and associated probabilities
                blob = cv2.dnn.blobFromImage( frame, 1 / 255.0, (416, 416), swapRB=True, crop=False )
                net.setInput( blob )
                start = time.time()
                layerOutputs = net.forward( ln )

                end = time.time()

                # initialize our lists of detected bounding boxes, confidences,
                # and class IDs, respectively
                boxes = []
                confidences = []
                classIDs = []
                dsa = []
                rects = []

                # loop over each of the layer outputs
                for output in layerOutputs:
                    # loop over each of the detections
                    for detection in output:
                        # extract the class ID and confidence (i.e., probability)
                        # of the current object detection

                        scores = detection[5:]
                        classID = np.argmax( scores )
                        confidence = scores[classID]

                        # filter out weak predictions by ensuring the detected
                        # probability is greater than the minimum probability
                        if confidence > 0.6:
                            # scale the bounding box coordinates back relative to
                            # the size of the image, keeping in mind that YOLO
                            # actually returns the center (x, y)-coordinates of
                            # the bounding box followed by the boxes' width and
                            # height
                            box = detection[0:4] * np.array( [W, H, W, H] )

                            (centerX, centerY, width, height) = box.astype( "int" )

                            # use the center (x, y)-coordinates to derive the top
                            # and and left corner of the bounding box
                            x = int( centerX - (width / 2) )
                            y = int( centerY - (height / 2) )
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle( x, y, x + width, y + height )

                            tracker.start_track( rgb, rect )

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            trackers.append( tracker )

                            # update our list of bounding box coordinates,
                            # confidences, and class IDs
                            
                            boxes.append( [x, y, int( width ), int( height )] )
                            confidences.append( float( confidence ) )
                            idxs = cv2.dnn.NMSBoxes( boxes, confidences, 0.6, 0.3 )

                            dsa.append( (tracker, classID) )
                            classIDs.append( idxs[0] )
                            cv2.rectangle( frame, (x, y), (x + width, y + height), (255, 0, 0), 1 )
                            cv2.putText( frame, str( LABELS[classID] + str( confidence ) ), (int( x ) + 10, int( y ) - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1 )

                            rects.append( (x, y, x + width, y + height) )

                            if (LABELS[classID] == "car"):
                                car += 1
                            elif (LABELS[classID] == "truck"):
                                truck += 1
                            elif (LABELS[classID] == "person"):
                                person += 1
                            elif (LABELS[classID] == "bus"):
                                bus += 1
                            elif (LABELS[classID] == "bicycle"):
                                bicycle += 1
                            else:
                                motorbike += 1

                if maximumTotal[0] > (bicycle + car + truck + person + bus + motorbike):
                    maximumTotal[0] = maximumTotal[0]
                else:
                    maximumTotal[0] = bicycle + car + truck + person + bus + motorbike
            
                cv2.putText( frame, "TOTAL" + str( maximumTotal[0] ), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255),
                            2 )

                if cv2.waitKey( 1 ) & 0xFF == ord( "q" ):
                    break

                totalFrames += 1

    print("[INFO] cleaning up...")
    print("[INFO] {} total detected objects in video".format( maximumTotal[0] ))
    vs.release()
    return maximumTotal[0]

def detection_logic():
    
    files_in_dir = []

    # r=>root, d=>directories, f=>files
    for r, d, f in os.walk(UPLOAD_FOLDER):
        for item in f:
            if '.mp4' in item:
                files_in_dir.append(os.path.join(r, item))

    for item in files_in_dir:
        print("file in dir: ", item)
    
    signal_dict = {}
    red_light_list = []
    green_light_list = []

    lane1 = [20]
    lane2 = [20]
    lane3 = [20]
    lane4 = [20]

    for i in range( 1 ):
        lane10 = [1]
        for i in lane10:

            a = showDetection( files_in_dir[0] )
            print(a)
            if a <= 10:
                green_light = 10
                print("this is signal timer for lane1 {}".format( green_light ))
                lane1.clear()
                lane1.append( green_light )
                green_light_list.append(green_light)
            elif a <= 20:
                green_light = 15
                print("this is signal timer for lane1 {}".format( green_light ))
                lane1.clear()
                lane1.append( green_light )
                green_light_list.append(green_light)
            elif a <= 30:
                green_light = 25
                print("this is signal timer for lane1 {}".format( green_light ))
                lane1.clear()
                lane1.append( green_light )
                green_light_list.append(green_light)
            else:
                green_light = 30
                print("this is signal timer for lane1 {}".format( green_light ))
                lane1.clear()
                lane1.append( green_light )
                green_light_list.append(green_light)
            
            
            #red_light_lane1 = lane2[0] + lane3[0] + lane4[0]
            # red_light_list.append(red_light_lane1)
            signal_dict.update({"green_light_lane_1" : green_light})
            # signal_dict.update({"red_light_lane_1" : red_light_lane1})
            print(lane1)

        lane20 = [2]
        for i in lane20:
            
            a = showDetection( files_in_dir[1]  )
            print(a)
            if a <= 10:
                green_light = 10
                print("this is signal timer for lane2 {}".format( green_light ))
                lane2.clear()
                lane2.append( green_light )
                green_light_list.append(green_light)
            elif a <= 20:
                green_light = 15
                print("this is signal timer for lane2 {}".format( green_light ))
                lane2.clear()
                lane2.append( green_light )
                green_light_list.append(green_light)
            elif a <= 30:
                green_light = 25
                print("this is signal timer for lane2 {}".format( green_light ))
                lane2.clear()
                lane2.append( green_light )
                green_light_list.append(green_light)
            else:
                green_light = 30
                print("this is signal timerfor lane2 {}".format( green_light ))
                lane2.clear()
                lane2.append( green_light )
                green_light_list.append(green_light)
            
            #red_light_lane2 = lane1[0] + lane3[0] + lane4[0]
            # red_light_list.append(red_light_lane2)
            signal_dict.update({"green_light_lane_2" : green_light})
            # signal_dict.update({"red_light_lane_2" : red_light_lane2})
            print(lane2)

        lane30 = [3]
        for i in lane30:
            a = showDetection( files_in_dir[2]  )
            print(a)
            if a <= 10:
                green_light = 10
                print("this is signal timer for lane3 {}".format( green_light ))
                lane3.clear()
                lane3.append( green_light )
                green_light_list.append(green_light)
            elif a <= 20:
                green_light = 15
                print("this is signal timer lane3 {}".format( green_light ))
                lane3.clear()
                lane3.append( green_light )
                green_light_list.append(green_light)
            elif a <= 30:
                green_light = 25
                print("this is signal timer lane3 {}".format( green_light ))
                lane3.clear()
                lane3.append( green_light )
                green_light_list.append(green_light)
            else:
                green_light = 30
                print("this is signal timer lane3 {}".format( green_light ))
                lane3.clear()
                lane3.append( green_light )
                green_light_list.append(green_light)

            #red_light_lane3 = lane2[0] + lane1[0] + lane4[0]
            # red_light_list.append(red_light_lane3)
            signal_dict.update({"green_light_lane_3" : green_light})
            # signal_dict.update({"red_light_lane_3" : red_light_lane3})
            print(lane3)

        lane40 = [1]
        for i in lane40:

            a = showDetection( files_in_dir[3]  )
            print(a)
            if a <= 10:
                green_light = 10
                print("this is signal timer for lane4 {}".format( green_light ))
                lane4.clear()
                lane4.append( green_light )
                green_light_list.append(green_light)
            elif a <= 20:
                green_light = 15
                print("this is signal timer for lane4 {}".format( green_light ))
                lane4.clear()
                lane4.append( green_light )
                green_light_list.append(green_light)
            elif a <= 30:
                green_light = 25
                print("this is signal timer for lane4 {}".format( green_light ))
                lane4.clear()
                lane4.append( green_light )
                green_light_list.append(green_light)
            else:
                green_light = 30
                print("this is signal timer for lane4 {}".format( green_light ))
                lane4.clear()
                lane4.append( green_light )
                green_light_list.append(green_light)

            red_light_lane1 = lane2[0] + lane3[0] + lane4[0]
            red_light_lane2 = lane1[0] + lane3[0] + lane4[0]
            red_light_lane3 = lane2[0] + lane1[0] + lane4[0]
            red_light_lane4 = lane2[0] + lane3[0] + lane1[0]

            red_light_list.append(red_light_lane1)
            red_light_list.append(red_light_lane2)
            red_light_list.append(red_light_lane3)
            red_light_list.append(red_light_lane4)
            signal_dict.update({"green_light_lane_4" : green_light})
            signal_dict.update({"red_light_lane_1" : red_light_lane1})
            signal_dict.update({"red_light_lane_2" : red_light_lane2})
            signal_dict.update({"red_light_lane_3" : red_light_lane3})
            signal_dict.update({"red_light_lane_4" : red_light_lane4})
            print(lane4)

    
    print("lane timing of 4 lane {}".format(green_light_list))
    print("red_light time for 4  lane {}".format(red_light_list))

    files = os.listdir(UPLOAD_FOLDER)
    files.sort()
    for f in files:
        src = UPLOAD_FOLDER+f
        dst = backup_folder+f
        os.rename(src,dst)

    return signal_dict


app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def upload_form():
	return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    fileslist = request.files.getlist('files[]')
    for file in fileslist:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    return redirect(url_for('detection'))

@app.route('/api-detection', methods=['POST'])
def upload_file_api():
	# check if the post request has the file part
	if 'files[]' not in request.files:
		resp = jsonify({'message' : 'No file part in the request'})
		resp.status_code = 400
		return resp
	
	files = request.files.getlist('files[]')
	
	errors = {}
	success = False
	
	for file in files:		
		if file:
			filename = secure_filename(file.filename)
			file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
			success = True
		else:
			errors[file.filename] = 'File type is not allowed'

	if success:
            value = detection_logic()
            return jsonify(value)

@app.route('/detection')
def detection():
    
    dict_signal_time = detection_logic()
    #dict_signal_time = json.loads(json_detect)
    return render_template('detection.html',dict_signal_time=dict_signal_time)

    # start flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
