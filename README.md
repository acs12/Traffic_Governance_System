

## Traffic Governance System

### Team
* Mrugesh Master
* Aayush Sukhadia
* Harshil Shah
* Deepen Patel

### Project Idea Description: -
Traffic system shows a great scope of trade with the environment and is directly connected to it. Manual traffic systems are proving to be insufficient due to rapid urbanization. Central monitoring systems are facing scalability issues as they process increasing amounts of data received from hundreds of traffic cameras. Major traffic problems like congestion, safety, pollution (leading to various health issues) and increased need for mobility. A solution to most of them is the construction of newer and safer highways and additional lanes on existing ones, but it proves to be expensive and often not feasible. Cities are limited by space and construction cannot keep up with ever-growing demand. Hence a need for an improved system with a minimal manual interface is persisting. Smart traffic governance system uses Artificial Intelligence to determine the flow of traffic, automated enforcement and communication to change the face of the traffic scenarios in urban cities suffering from such traffic issues.
Traffic governance system will deal with the traffic on the crossroads with the help of cameras on the crossroads which will detect the congestion with the help of image processing and get the count of vehicles in the heavy lane. After detecting the heavy lane, the timer for that particular will increase and the other lanes timer will be balanced accordingly.

### Goal: 
The scope of this project is very vast; people travelling long distances will be benefited the most by this application as the travel time will be greatly reduced.
People travelling in the day to day life for work will also be benefited by reducing the waiting time on the signals. It will help to create fewer congestion on the crossroads and the heavy lane.

### Technology Stack:
Frontend: HTML, CSS, Bootstrap, JS <br>
Backend: Python, Flask (Web Framework)<br>
Machine Learning Tools: YOLO-v3, OpenCV, dlib<br>
Cloud Tools: Docker, Kubernetes<br>

### Docker Image
Pull Docker image <br>
``` docker pull mrugeshmaster/yolo_tgs:1.1  ``` <br><br>
Run Docker image <br>
``` docker run -d -p 5000:5000 mrugeshmaster/yolo_tgs:1.1 ``` <br>

### API via Postman
``` http://localhost:5000/api-detection ```
### Kubernetes
Since the algorithm consumes atleast 3GB of memory and needs atleast 4 CPUs to execute at a moderate pace, it was tough to get such processing power in free tiers in Cloud Platforms. Hence, we installed ```minikube``` to create a local Kubernetes cluster and ```virtualbox``` to create VM to simulate Kubernetes Engine.

Update: By trimming the video, we were able to utilize IBM Kubernetes resources to host the docker image and run the application. IBM Kubernetes creates a free Cluster with 2 vCPUs and 4GB RAM and allows NodePort service to communicate with the pods.

### Object Detection
We used YOLOv3 to implement object detection to detect number of objects in a video, OpenCV to convert the video in frames. Using pre-trained weights by YOLO and coco.names labelset, the program detects a person, car, truck, bus and bicycle. Each frame is consumed by YOLO algorithm to count the number of objects. To filter out weak predictions, we set the confidence > 0.6. YOLOv3 uses a neural network with 53 Convolution layers as per [YOLOv3 Paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf)<br>
To avoid, duplicate count of the same object, we used dlib to implement a correlation tracker. The tracker assigns a unique ID to the object. While detecting the objects, the program checks for the presence of unique ID. If a unique ID exits, the program will skip the object. We learnt about correlation trackers from [IEEE Paper : Vehicle Counting for Traffic Management System using YOLO and Correlation Filter](https://ieeexplore.ieee.org/document/8482380)

### Architecture Diagram
![Architecture Diagram](/images/TrafficGovernanceSystem.jpg)


> Note: yolov3.weights is too large to upload in Github Repo. Download [yolov3.weights](https://pjreddie.com/media/files/yolov3.weights) and place it in yolo directory.
