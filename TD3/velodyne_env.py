import math
import os
import random
import subprocess
import time
from os import path

import numpy as np
import rospy
import sensor_msgs.point_cloud2 as pc2
from gazebo_msgs.msg import ModelState
from gazebo_msgs.msg import ModelStates
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from squaternion import Quaternion
from std_srvs.srv import Empty
from visualization_msgs.msg import Marker
from visualization_msgs.msg import MarkerArray
from gazebo_msgs.srv import SetModelState

from pathplaner import *

GOAL_REACHED_DIST = 0.3
COLLISION_DIST = 0.35
TIME_DELTA = 0.1
MAXENVSIZE  = 14.5  # 边长为30的正方形作为环境的大小


# Check if the random goal position is located on an obstacle and do not accept it if it is
def check_pos(x, y):
    goal_ok = True

    # if -3.8 > x > -6.2 and 6.2 > y > 3.8:
    #     goal_ok = False
    #
    # if -1.3 > x > -2.7 and 4.7 > y > -0.2:
    #     goal_ok = False
    #
    # if -0.3 > x > -4.2 and 2.7 > y > 1.3:
    #     goal_ok = False
    #
    # if -0.8 > x > -4.2 and -2.3 > y > -4.2:
    #     goal_ok = False
    #
    # if -1.3 > x > -3.7 and -0.8 > y > -2.7:
    #     goal_ok = False
    #
    # if 4.2 > x > 0.8 and -1.8 > y > -3.2:
    #     goal_ok = False
    #
    # if 4 > x > 2.5 and 0.7 > y > -3.2:
    #     goal_ok = False
    #
    # if 6.2 > x > 3.8 and -3.3 > y > -4.2:
    #     goal_ok = False
    #
    # if 4.2 > x > 1.3 and 3.7 > y > 1.5:
    #     goal_ok = False
    #
    # if -3.0 > x > -7.2 and 0.5 > y > -1.5:
    #     goal_ok = False
    #
    if x > 7 or x < -7 or y > 7 or y < -7:
        goal_ok = False

    return goal_ok


class GazeboEnv:
    """Superclass for all Gazebo environments."""

    def __init__(self, launchfile, environment_dim):
        self.environment_dim = environment_dim
        self.odom_x = 0
        self.odom_y = 0

        self.goal_x = 1
        self.goal_y = 0.0

        self.upper = 5.0
        self.lower = -5.0
        self.velodyne_data = np.ones(self.environment_dim) * 10 # environment_dim = 20
        self.last_odom = None

        self.set_self_state = ModelState()
        self.set_self_state.model_name = "r1"
        self.set_self_state.pose.position.x = 0.0
        self.set_self_state.pose.position.y = 0.0
        self.set_self_state.pose.position.z = 0.0
        self.set_self_state.pose.orientation.x = 0.0
        self.set_self_state.pose.orientation.y = 0.0
        self.set_self_state.pose.orientation.z = 0.0
        self.set_self_state.pose.orientation.w = 1.0

        self.gaps = [[-np.pi / 2 - 0.03, -np.pi / 2 + np.pi / self.environment_dim]]
        for m in range(self.environment_dim - 1):
            self.gaps.append(
                [self.gaps[m][1], self.gaps[m][1] + np.pi / self.environment_dim]
            )
        self.gaps[-1][-1] += 0.03

        port = "11311"
        subprocess.Popen(["roscore", "-p", port])

        print("Roscore launched!")

        # Launch the simulation with the given launchfile name
        rospy.init_node("gym", anonymous=True)
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", launchfile)
        if not path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        subprocess.Popen(["roslaunch", "-p", port, fullpath])
        print("Gazebo launched!")

        # Set up the ROS publishers and subscribers
        self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
        self.set_state = rospy.Publisher(
            "gazebo/set_model_state", ModelState, queue_size=10
        )
        self.unpause = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)
        self.pause = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.reset_proxy = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.publisher = rospy.Publisher("goal_point", MarkerArray, queue_size=3)
        self.publisher2 = rospy.Publisher("linear_velocity", MarkerArray, queue_size=1)
        self.publisher3 = rospy.Publisher("angular_velocity", MarkerArray, queue_size=1)
        self.velodyne = rospy.Subscriber(
            "/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1
        )
        self.odom = rospy.Subscriber(
            "/r1/odom", Odometry, self.odom_callback, queue_size=1
        )

        #-------------------------------------hzx0608--------------------------------#
        self.dict_name_id = {'jackal0': 0, 'jackal1': 1, 'jackal2': 2, 'jackal3': 3, 'jackal4': 4, 'jackal5': 5,
                             'jackal6': 6, 'jackal7': 7, 'jackal8': 8, 'jackal9': 9, 'jackal10': 10}

        self.dict_id_name = {0: 'jackal0', 1: 'jackal1', 2: 'jackal2', 3: 'jackal3', 4: 'jackal4', 5: 'jackal5',
                             6: 'jackal6', 7: 'jackal7', 8: 'jackal8', 9: 'jackal9', 10: 'jackal10'}

        # 动态障碍机器人的名字列表
        self.obs_robot_namelist = ['jackal0', 'jackal1']
        # 储存static障碍机器人的position
        self.static_obs_pos = []
        # 储存障碍机器人的控制指令
        self.pub_obs = []
        # 给障碍机器人发布控制指令
        for i in range(len(self.obs_robot_namelist)):
            pub_obs = rospy.Publisher('/' + self.obs_robot_namelist[i] + '/jackal_velocity_controller/cmd_vel', Twist, queue_size=10)
            self.pub_obs.append(pub_obs)
        # 动态障碍的个数
        self.num_obs = 2

        # 人工势场法相关参数
        # 慢速
        self.V = 0.2
        # 快速
        # self.V = 0.4
        self.planer = CPFpathplaner()
        # self.robs =  [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]
        self.robs = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
        # 储存动态障碍机器人每次的目标点
        self.dynamic_obs_goal_x = np.zeros(self.num_obs)
        self.dynamic_obs_goal_y = np.zeros(self.num_obs)
        # 储存动态障碍机器人每次的起点
        self.dynamic_obs_start_pos = np.zeros([self.num_obs, 2])
        # 初始化的障碍位置
        self.xobs = np.zeros(9)
        self.yobs = np.zeros(9)
        self.vxobs = np.zeros(9)
        self.vyobs = np.zeros(9)
        self.sub = rospy.Subscriber('/gazebo/model_states', ModelStates, self.gazebo_states_callback)
        self.gazebo_model_states = ModelStates()

        self.resetval()
        # -------------------------------------hzx0608--------------------------------#
    def resetval(self,):
        # 障碍机器人的相关参数
        self.obs_robot_state = []  # obs_robot_state--->x,y,v,w,yaw,vx,vy
        self.obs_d           = []  # 障碍机器人到各自目标的距离
        for _ in range(len(self.obs_robot_namelist)):
            self.obs_robot_state.append([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # x,y,v,w,yaw,vx,vy
            self.obs_d.append(0.0)
    def gazebo_states_callback(self, data):
        self.gazebo_model_states = data
        # name: ['ground_plane', 'jackal1', 'jackal2', 'jackal0',...]
        for i in range(len(data.name)):
            # 障碍机器人
            if data.name[i] in self.obs_robot_namelist:
                # obs_robot_state--->x,y,v,w,yaw,vx,vy
                # data.name[i]='jackal0'
                self.obs_robot_state[self.dict_name_id[data.name[i]]][0] = data.pose[i].position.x
                self.obs_robot_state[self.dict_name_id[data.name[i]]][1] = data.pose[i].position.y
                v = math.sqrt(data.twist[i].linear.x**2 + data.twist[i].linear.y**2)
                self.obs_robot_state[self.dict_name_id[data.name[i]]][2] = v
                self.obs_robot_state[self.dict_name_id[data.name[i]]][3] = data.twist[i].angular.z
                rpy = self.euler_from_quaternion(data.pose[i].orientation.x,data.pose[i].orientation.y,
                data.pose[i].orientation.z,data.pose[i].orientation.w)
                self.obs_robot_state[self.dict_name_id[data.name[i]]][4] = rpy[2]
                self.obs_robot_state[self.dict_name_id[data.name[i]]][5] = data.twist[i].linear.x
                self.obs_robot_state[self.dict_name_id[data.name[i]]][6] = data.twist[i].linear.y
    def euler_from_quaternion(self, x, y, z, w):
        euler = [0, 0, 0]
        Epsilon = 0.0009765625
        Threshold = 0.5 - Epsilon
        TEST = w * y - x * z
        if TEST < -Threshold or TEST > Threshold:
            if TEST > 0:
                sign = 1
            elif TEST < 0:
                sign = -1
            euler[2] = -2 * sign * math.atan2(x, w)
            euler[1] = sign * (math.pi / 2.0)
            euler[0] = 0
        else:
            euler[0] = math.atan2(2 * (y * z + w * x),
                                  w * w - x * x - y * y + z * z)
            euler[1] = math.asin(-2 * (x * z - w * y))
            euler[2] = math.atan2(2 * (x * y + w * z),
                                  w * w + x * x - y * y - z * z)

        return euler
    # Read velodyne pointcloud and turn it into distance data, then select the minimum value for each angle
    # range as state representation
    def velodyne_callback(self, v):
        data = list(pc2.read_points(v, skip_nans=False, field_names=("x", "y", "z")))
        self.velodyne_data = np.ones(self.environment_dim) * 10
        for i in range(len(data)):
            if data[i][2] > -0.2:
                dot = data[i][0] * 1 + data[i][1] * 0
                mag1 = math.sqrt(math.pow(data[i][0], 2) + math.pow(data[i][1], 2))
                mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
                beta = math.acos(dot / (mag1 * mag2)) * np.sign(data[i][1])
                dist = math.sqrt(data[i][0] ** 2 + data[i][1] ** 2 + data[i][2] ** 2)

                for j in range(len(self.gaps)):
                    if self.gaps[j][0] <= beta < self.gaps[j][1]:
                        self.velodyne_data[j] = min(self.velodyne_data[j], dist)
                        break

    def odom_callback(self, od_data):
        self.last_odom = od_data

    # Perform an action and read a new state
    def step(self, action):
        target = False

        # Publish the robot action
        vel_cmd = Twist()
        vel_cmd.linear.x = action[0]
        vel_cmd.angular.z = action[1]
        self.vel_pub.publish(vel_cmd)
        self.publish_markers(action)

        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        # propagate state for TIME_DELTA seconds
        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            pass
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")

        # read velodyne laser state
        done, collision, min_laser = self.observe_collision(self.velodyne_data)
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        # Calculate robot heading from odometry data
        self.odom_x = self.last_odom.pose.pose.position.x
        self.odom_y = self.last_odom.pose.pose.position.y
        quaternion = Quaternion(
            self.last_odom.pose.pose.orientation.w,
            self.last_odom.pose.pose.orientation.x,
            self.last_odom.pose.pose.orientation.y,
            self.last_odom.pose.pose.orientation.z,
        )
        euler = quaternion.to_euler(degrees=False)
        angle = round(euler[2], 4)

        # Calculate distance to the goal from the robot
        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        # Calculate the relative angle between the robots heading and heading toward the goal
        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y
        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))
        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle
        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        # Detect if the goal has been reached and give a large positive reward
        if distance < GOAL_REACHED_DIST:
            target = True
            done = True

        robot_state = [distance, theta, action[0], action[1]]
        state = np.append(laser_state, robot_state)
        reward = self.get_reward(target, collision, action, min_laser)
        info = [distance, theta, action[0], action[1]]
        return state, reward, done, target, info
        # 随机初始化全部障碍的位置

    def random_square(self, a):
        obstacle_robot_position = 2 * a * np.random.random_sample((2 * self.num_obs, 2)) - a
        return obstacle_robot_position
    def set_dynamic(self, x, y):
        # 获取障碍的随机初始位置
        while (True):
            flag = True
            obs_pos = self.random_square(MAXENVSIZE / 2)
            # 给不同的障碍赋值
            # ----------------------------------------------------------------------------------------------------
            # 0-9是动态障碍机器人的初始位置
            for obs_robot_id in range(self.num_obs):
                self.dynamic_obs_start_pos[obs_robot_id][0] = obs_pos[obs_robot_id][0]
                self.dynamic_obs_start_pos[obs_robot_id][1] = obs_pos[obs_robot_id][1]
            # 10-19是动态障碍机器人的目标位置
            # print("obs_pos={}".format(obs_pos))
            for j in range(self.num_obs):
                self.dynamic_obs_goal_x[j] = obs_pos[self.num_obs + j][0]
                self.dynamic_obs_goal_y[j] = obs_pos[self.num_obs + j][1]
            # ----------------------------------------------------------------------------------------------------
            # 起点\终点不能在动态障碍机器人的起点\终点4m范围内
            # ----------------------------------------------------------------------------------------------------
            for i in range(len(self.obs_robot_namelist)):
                if (math.sqrt((x - self.dynamic_obs_start_pos[i][0]) ** 2 + (
                        y - self.dynamic_obs_start_pos[i][1]) ** 2) < 3.0) or \
                        (math.sqrt((self.goal_x-self.dynamic_obs_goal_x[i])**2+
                                   (self.goal_y-self.dynamic_obs_goal_y[i])**2) < 3.0):
                    flag = False
                    break
            for i in range(len(self.obs_robot_namelist)):
                for j in range(len(self.static_obs_pos)):
                    if math.sqrt((self.static_obs_pos[j][0] - self.dynamic_obs_start_pos[i][0]) ** 2 + (
                            self.static_obs_pos[j][1] - self.dynamic_obs_start_pos[i][1]) ** 2) < 3.0:
                        flag = False
                        break
            # 动态障碍机器人起点之间的距离要大于n米
            # ----------------------------------------------------------------------------------------------------
            for i in range(len(self.obs_robot_namelist)):
                for j in range(len(self.obs_robot_namelist)):
                    if j != i:
                        if math.sqrt((self.dynamic_obs_start_pos[i][0] - self.dynamic_obs_start_pos[j][0]) ** 2 + (
                                self.dynamic_obs_start_pos[i][1] - self.dynamic_obs_start_pos[j][1]) ** 2) < 3.0:
                            flag = False
                            break
                if flag == False:
                    break
            # ----------------------------------------------------------------------------------------------------
            # 动态障碍机器人终点之间的距离要大于n米
            # ----------------------------------------------------------------------------------------------------
            for i in range(len(self.obs_robot_namelist)):
                for j in range(len(self.obs_robot_namelist)):
                    if j != i:
                        if math.sqrt((self.dynamic_obs_goal_x[i] - self.dynamic_obs_goal_x[j]) ** 2 + (
                                self.dynamic_obs_goal_y[i] - self.dynamic_obs_goal_y[j]) ** 2) < 3.0:
                            flag = False
                            break
                if flag == False:
                    break
            # ----------------------------------------------------------------------------------------------------
            # 满足所有的条件
            if flag == True:
                break
    def reset(self):

        # Resets the state of the environment and returns an initial observation.
        rospy.wait_for_service("/gazebo/reset_world")
        try:
            self.reset_proxy()

        except rospy.ServiceException as e:
            print("/gazebo/reset_simulation service call failed")

        angle = np.random.uniform(-np.pi, np.pi)
        quaternion = Quaternion.from_euler(0.0, 0.0, angle)
        object_state = self.set_self_state

        # start point
        x = 0
        y = 0
        position_ok = False
        while not position_ok:
            x = np.random.uniform(-4.5, 4.5)
            y = np.random.uniform(-4.5, 4.5)
            position_ok = check_pos(x, y)
        object_state.pose.position.x = x
        object_state.pose.position.y = y
        # object_state.pose.position.z = 0.
        object_state.pose.orientation.x = quaternion.x
        object_state.pose.orientation.y = quaternion.y
        object_state.pose.orientation.z = quaternion.z
        object_state.pose.orientation.w = quaternion.w
        self.set_state.publish(object_state)

        self.odom_x = object_state.pose.position.x
        self.odom_y = object_state.pose.position.y

        # set a random goal in empty space in environment
        print("change_goal")
        self.change_goal()
        # static obstacle (randomly scatter boxes in the environment)
        print("static obstacle")
        self.random_box()
        self.publish_markers([0.0, 0.0])
        print("dynamic obstacle")
        # dynamic obstacle
        self.set_dynamic(x,y)
        print("dynamic obstacle finish")
        rospy.wait_for_service("/gazebo/unpause_physics")
        try:
            self.unpause()
        except (rospy.ServiceException) as e:
            print("/gazebo/unpause_physics service call failed")

        time.sleep(TIME_DELTA)

        rospy.wait_for_service("/gazebo/pause_physics")
        try:
            self.pause()
        except (rospy.ServiceException) as e:
            print("/gazebo/pause_physics service call failed")
        v_state = []
        v_state[:] = self.velodyne_data[:]
        laser_state = [v_state]

        distance = np.linalg.norm(
            [self.odom_x - self.goal_x, self.odom_y - self.goal_y]
        )

        skew_x = self.goal_x - self.odom_x
        skew_y = self.goal_y - self.odom_y

        dot = skew_x * 1 + skew_y * 0
        mag1 = math.sqrt(math.pow(skew_x, 2) + math.pow(skew_y, 2))
        mag2 = math.sqrt(math.pow(1, 2) + math.pow(0, 2))
        beta = math.acos(dot / (mag1 * mag2))

        if skew_y < 0:
            if skew_x < 0:
                beta = -beta
            else:
                beta = 0 - beta
        theta = beta - angle

        if theta > np.pi:
            theta = np.pi - theta
            theta = -np.pi - theta
        if theta < -np.pi:
            theta = -np.pi - theta
            theta = np.pi - theta

        robot_state = [distance, theta, 0.0, 0.0]
        state = np.append(laser_state, robot_state)
        return state

    def change_goal(self):
        # Place a new goal and check if its location is not on one of the obstacles
        if self.upper < 10:
            self.upper += 0.004
        if self.lower > -10:
            self.lower -= 0.004

        goal_ok = False

        while not goal_ok:
            self.goal_x = self.odom_x + random.uniform(self.upper, self.lower)
            self.goal_y = self.odom_y + random.uniform(self.upper, self.lower)
            goal_ok = check_pos(self.goal_x, self.goal_y)

    def random_box(self):
        # Randomly change the location of the boxes in the environment on each reset to randomize the training
        # environment
        for i in range(4):
            name = "cardboard_box_" + str(i)

            x = 0
            y = 0
            box_ok = False
            while not box_ok:
                x = np.random.uniform(-6, 6)
                y = np.random.uniform(-6, 6)
                box_ok = check_pos(x, y)
                distance_to_robot = np.linalg.norm([x - self.odom_x, y - self.odom_y])
                distance_to_goal = np.linalg.norm([x - self.goal_x, y - self.goal_y])
                if distance_to_robot < 1.5 or distance_to_goal < 1.5:
                    box_ok = False
            box_state = ModelState()
            box_state.model_name = name
            box_state.pose.position.x = x
            box_state.pose.position.y = y
            box_state.pose.position.z = 0.0
            box_state.pose.orientation.x = 0.0
            box_state.pose.orientation.y = 0.0
            box_state.pose.orientation.z = 0.0
            box_state.pose.orientation.w = 1.0
            self.set_state.publish(box_state)
            self.static_obs_pos.append((x,y))

    def publish_markers(self, action):
        # Publish visual data in Rviz
        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = marker.CYLINDER
        marker.action = marker.ADD
        marker.scale.x = 0.1
        marker.scale.y = 0.1
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = self.goal_x
        marker.pose.position.y = self.goal_y
        marker.pose.position.z = 0

        markerArray.markers.append(marker)

        self.publisher.publish(markerArray)

        markerArray2 = MarkerArray()
        marker2 = Marker()
        marker2.header.frame_id = "odom"
        marker2.type = marker.CUBE
        marker2.action = marker.ADD
        marker2.scale.x = abs(action[0])
        marker2.scale.y = 0.1
        marker2.scale.z = 0.01
        marker2.color.a = 1.0
        marker2.color.r = 1.0
        marker2.color.g = 0.0
        marker2.color.b = 0.0
        marker2.pose.orientation.w = 1.0
        marker2.pose.position.x = 5
        marker2.pose.position.y = 0
        marker2.pose.position.z = 0

        markerArray2.markers.append(marker2)
        self.publisher2.publish(markerArray2)

        markerArray3 = MarkerArray()
        marker3 = Marker()
        marker3.header.frame_id = "odom"
        marker3.type = marker.CUBE
        marker3.action = marker.ADD
        marker3.scale.x = abs(action[1])
        marker3.scale.y = 0.1
        marker3.scale.z = 0.01
        marker3.color.a = 1.0
        marker3.color.r = 1.0
        marker3.color.g = 0.0
        marker3.color.b = 0.0
        marker3.pose.orientation.w = 1.0
        marker3.pose.position.x = 5
        marker3.pose.position.y = 0.2
        marker3.pose.position.z = 0

        markerArray3.markers.append(marker3)
        self.publisher3.publish(markerArray3)

    @staticmethod
    def observe_collision(laser_data):
        # Detect a collision from laser data
        min_laser = min(laser_data)
        if min_laser < COLLISION_DIST:
            return True, True, min_laser
        return False, False, min_laser

    @staticmethod
    def get_reward(target, collision, action, min_laser):
        if target:
            return 100.0
        elif collision:
            return -100.0
        else:
            r3 = lambda x: 1 - x if x < 1 else 0.0
            return action[0] / 2 - abs(action[1]) / 2 - r3(min_laser) / 2

    def run(self):
        # rate = rospy.Rate(50)

        cmd_vel = Twist()
        for obs_robot_id in range(len(self.obs_robot_namelist)):
            # self.obs_robot_state------x,y,v,w,yaw,vx,vy
            self.obs_d[obs_robot_id] = math.sqrt(
                (self.dynamic_obs_goal_x[obs_robot_id] - self.obs_robot_state[obs_robot_id][0]) ** 2 + (
                            self.dynamic_obs_goal_y[obs_robot_id] - self.obs_robot_state[obs_robot_id][1]) ** 2)
            if self.obs_d[obs_robot_id] > 0.1:
                qx = (self.dynamic_obs_goal_x[obs_robot_id] - self.obs_robot_state[obs_robot_id][0]) / self.obs_d[
                    obs_robot_id]
                qy = (self.dynamic_obs_goal_y[obs_robot_id] - self.obs_robot_state[obs_robot_id][1]) / self.obs_d[
                    obs_robot_id]
                vx = qx * self.V
                vy = qy * self.V
                # *************************CPF path planning*****************************#
                for obs_robot_id2 in range(len(self.obs_robot_namelist)):
                    # 其余4个动态障碍机器人对于某一个动态障碍机器人而言也是障碍
                    # 所以对于某一个动态障碍机器人而言，一共有9个障碍
                    if obs_robot_id2 < obs_robot_id:
                        self.xobs[-1 - obs_robot_id2] = self.obs_robot_state[obs_robot_id2][0]
                        self.yobs[-1 - obs_robot_id2] = self.obs_robot_state[obs_robot_id2][1]
                        self.vxobs[-1 - obs_robot_id2] = self.obs_robot_state[obs_robot_id2][5]
                        self.vyobs[-1 - obs_robot_id2] = self.obs_robot_state[obs_robot_id2][6]
                    if obs_robot_id2 > obs_robot_id:
                        self.xobs[-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][0]
                        self.yobs[-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][1]
                        self.vxobs[-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][5]
                        self.vyobs[-obs_robot_id2] = self.obs_robot_state[obs_robot_id2][6]
                for i in range(4):
                    self.xobs[i + 2]=self.static_obs_pos[i][0]
                    self.yobs[i + 2] = self.static_obs_pos[i][1]
                    self.robs[i + 2] = 1
                # print("after xobs={}".format(self.xobs))
                self.planer.get_obs_state(self.xobs, self.yobs, self.robs, self.vxobs, self.vyobs, len(self.xobs))
                self.planer.get_target_state(self.dynamic_obs_goal_x[obs_robot_id],
                                             self.dynamic_obs_goal_y[obs_robot_id], 0, 0)
                vx, vy = self.planer.fn_pf_vc(self.obs_robot_state[obs_robot_id][0],
                                              self.obs_robot_state[obs_robot_id][1],
                                              self.obs_robot_state[obs_robot_id][5],
                                              self.obs_robot_state[obs_robot_id][6])

                if self.obs_d[obs_robot_id] < 0.5:
                    while True:
                        flag = True

                        rand_position = 2 * (MAXENVSIZE / 2) * np.random.random_sample(2) - (MAXENVSIZE / 2)
                        self.dynamic_obs_goal_x[obs_robot_id] = rand_position[0]
                        self.dynamic_obs_goal_y[obs_robot_id] = rand_position[1]

                        # 动态障碍机器人终点之间的距离要大于n米
                        for j in range(len(self.obs_robot_namelist)):
                            if j != obs_robot_id:
                                if math.sqrt(
                                        (self.dynamic_obs_goal_x[obs_robot_id] - self.dynamic_obs_goal_x[j]) ** 2 + (
                                                self.dynamic_obs_goal_y[obs_robot_id] - self.dynamic_obs_goal_y[
                                            j]) ** 2) < 3.0:
                                    flag = False
                                    break

                        # 如果起点和终点在动态障碍机器人的终点4m范围内则需要重新生成障碍物
                        '''
                        if math.sqrt((self.sp[0]-self.dynamic_obs_goal_x[obs_robot_id])**2+(self.sp[1]-self.dynamic_obs_goal_y[obs_robot_id])**2) < 4.0:
                            flag = False
                        if math.sqrt((self.gp[0]-self.dynamic_obs_goal_x[obs_robot_id])**2+(self.gp[1]-self.dynamic_obs_goal_y[obs_robot_id])**2) < 4.0:
                            flag = False
                        '''

                        if flag == True:
                            break

                    rospy.wait_for_service('/gazebo/set_model_state')
                    val = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

                    state = ModelState()
                    for i in range(len(self.gazebo_model_states.name)):
                        # 放置动态障碍机器人的终点位置
                        NAME_GOAL = 'obs_point_goal' + str(obs_robot_id)
                        if self.gazebo_model_states.name[i] == NAME_GOAL:
                            state.reference_frame = 'world'
                            state.pose.position.z = 0.0
                            state.model_name = self.gazebo_model_states.name[i]
                            state.pose.position.x = self.dynamic_obs_goal_x[obs_robot_id]
                            state.pose.position.y = self.dynamic_obs_goal_y[obs_robot_id]
                            val(state)
                # *************************path planning*****************************#
                yawcmd = math.atan2(vy, vx)

                vcmd = (vx ** 2 + vy ** 2) ** 0.5
                vcmd = self.planer.limvar(vcmd, -self.V, self.V)
                cmd_vel.linear.x = vcmd

                if yawcmd - self.obs_robot_state[obs_robot_id][4] > math.pi:
                    yawcmd = yawcmd - 2 * math.pi
                elif yawcmd - self.obs_robot_state[obs_robot_id][4] < -math.pi:
                    yawcmd = yawcmd + 2 * math.pi
                wz = 3.0 * (yawcmd - self.obs_robot_state[obs_robot_id][4]) - 0.5 * self.obs_robot_state[obs_robot_id][
                    3]
                if wz > 1.0:
                    wz = 1.0
                elif wz < -1.0:
                    wz = -1.0

                cmd_vel.angular.z = wz

            self.pub_obs[obs_robot_id].publish(cmd_vel)

        # rate.sleep()