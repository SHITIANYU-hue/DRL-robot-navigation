# DRL-robot-navigation

Deep Reinforcement Learning for mobile robot navigation in ROS Gazebo simulator. Using Twin Delayed Deep Deterministic Policy Gradient (TD3) neural network, a robot learns to navigate to a random goal point in a simulated environment while avoiding obstacles. Obstacles are detected by laser readings and a goal is given to the robot in polar coordinates. Trained ir ROS Gazebo simulator with PyTorch. Developed with ROS Melodic on Ubuntu 18.04. (The current repository has not yet been fully tested yet, so if there any issues please post it in the issues section).

Main dependencies: 

* [ROS Melodic](http://wiki.ros.org/melodic/Installation)
* [PyTorch](https://pytorch.org/get-started/locally/)

Clone the repository:
```shell
$ cd ~
### Clone this repo
$ git clone https://github.com/reiniscimurs/DRL-robot-navigation
```
The network can be run with a standard 2D laser, but this implementation uses a simulated [3D Velodyne sensor](https://github.com/lmark1/velodyne_simulator)
Compile the workspace:
```shell
$ cd ~/DRL-robot-navigation/catkin_ws
### Compile
$ catkin_make_isolated
```

Open a terminal and set up sources:
```shell
$ export ROS_HOSTNAME=localhost
$ export ROS_MASTER_URI=http://localhost:11311
$ export ROS_PORT_SIM=11311
$ export GAZEBO_RESOURCE_PATH=~/DRL-robot-navigation/catkin_ws/src/multi_robot_scenario/launch
$ source ~/.bashrc
$ cd ~/DRL-robot-navigation/catkin_ws
$ source devel/setup.bash
### Run the training
$ cd ~/DRL-robot-navigation/TD3
$ python3 velodyne_td3.py
```