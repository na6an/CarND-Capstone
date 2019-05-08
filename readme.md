# System Integration
### Programming a Real Self-Driving Car

## Team & Responsibility
### Hafizur Rahman
Mail: bibagimon[at]gmail.com  
TZ: UTC+9(Tokyo)  
Lead, Traffic light detector DL model

### Ryo Takaki
Mail: takaki.ryou[at]gmail.com  
TZ: UTC-5 (Michigan)  
Implement DBW node

### Henry Gridley
Mail: henry.gridley[at]gmail.com  
TZ: UTC-7 (San Francisco)  
Implement traffic light detector

### Freedom Challis
Mail: freedomtrain7[at]gmail.com  
TZ: UTC-6 (Illinois)  
Implement traffic light detector

### Nathan Byun
Mail: jbyun6[at]gmail.com  
TZ: UTC-5 (Virginia -> Ohio)  
Implement waypoint updater


## Setup 
Original repo can be found here:  
https://github.com/udacity/CarND-Capstone  

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

An alternative frozen train model is also available from here:  
https://drive.google.com/open?id=1UiCU6oJ7Lbld8lEQq52j9r3J0dtCIXN9  
Place this file in `src/tl_detector/light_classification/models` directory and rename to "ssd_sim.pb" to use this.  

## Run

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator  
`cd ~(path-to)/CarND-Capstone/ros`  
Then,  
```
cd ros  
catkin_make  
source devel/setup.sh  
roslaunch launch/styx.launch  
```
## Troubleshoot  

```
CMake Error at /opt/ros/kinetic/share/pcl_ros/cmake/pcl_rosConfig.cmake:113 (message):  
  Project 'pcl_ros' specifies '/usr/include/pcl-1.7' as an include dir, which  
  is not found.  It does neither exist as an absolute directory nor in  
  '/opt/ros/kinetic//usr/include/pcl-1.7'.  Check the issue tracker  
  'https://github.com/ros-perception/perception_pcl/issues' and consider  
  creating a ticket if the problem has not been reported yet.  
Call Stack (most recent call first):  
  /opt/ros/kinetic/share/catkin/cmake/catkinConfig.cmake:76 (find_package)  
  waypoint_follower/CMakeLists.txt:10 (find_package)
```

`sudo apt install ros-kinetic-pcl-ros`


==

`ERROR: cannot launch node of type [tl_detector/tl_detector.py]: can't locate node [tl_detector.py] in package [tl_detector]`

`find ~/(path-to)/CarND-Capstone/ros -type f -iname "*.py" -exec chmod +x {} \;`

==

AttributeError: 'module' object has no attribute 'poll'
Then,




## Code Structure

  <img src="https://github.com/na6an/sdcnd-t3/blob/master/P3_System_Integration/img/final-project-ros-graph-v2.png" alt="alt text" width="920" height="640">  

Below is a brief overview of the repo structure, along with descriptions of the ROS nodes.  
The code that you will need to modify for the project will be contained entirely within the  
/ros/src/ directory. Within this directory, you will find the following ROS packages:

/ros/src/tl_detector/  
This package contains the traffic light detection node: tl_detector.py.  
This node takes in data from the /image_color, /current_pose, and /base_waypoints topics  
and publishes the locations to stop for red traffic lights to the /traffic_waypoint topic.

The /current_pose topic provides the vehicle's current position,  
and /base_waypoints provides a complete list of waypoints the car will be following.

You will build both a traffic light detection node and a traffic light classification node.  
Traffic light detection should take place within tl_detector.py,  
whereas traffic light classification should take place within ../tl_detector/light_classification_model/tl_classfier.py.


/ros/src/waypoint_updater/
This package contains the waypoint updater node: waypoint_updater.py.  
The purpose of this node is to update the target velocity property of each waypoint  
based on traffic light and obstacle detection data.  
This node will subscribe to the /base_waypoints, /current_pose, /obstacle_waypoint,  
and /traffic_waypoint topics, and publish a list of waypoints ahead of the car  
with target velocities to the /final_waypoints topic.


/ros/src/twist_controller/
Carla is equipped with a drive-by-wire (dbw) system, meaning the throttle, brake, and steering have electronic control.  
This package contains the files that are responsible for control of the vehicle: the node dbw_node.py and the file twist_controller.py, along with a pid and lowpass filter that you can use in your implementation. The dbw_node subscribes to the /current_velocity topic along with the /twist_cmd topic to receive target linear and angular velocities. Additionally, this node will subscribe to /vehicle/dbw_enabled, which indicates if the car is under dbw or driver control. This node will publish throttle, brake, and steering commands to the /vehicle/throttle_cmd, /vehicle/brake_cmd, and /vehicle/steering_cmd topics.
