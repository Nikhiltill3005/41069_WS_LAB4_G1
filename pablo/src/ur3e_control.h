#ifndef UR3E_CONTROL_H
#define UR3E_CONTROL_H

#include <memory>
#include <fstream>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/bool.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/planning_scene_interface/planning_scene_interface.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

using namespace std;

/**
 * @class ur3e_control 
 * @brief This is the subsystem that is responsible for controling the UR3e robotic arm, by reading a .CSV file that contains the planned path to draw the processed image captured from the user. 
 */

class ur3eControl : public rclcpp::Node {
    public:

    /**
     * @brief Constructor for UR3e Control 
     * Initialises the robot by setting up publishers and subscribers, and moves the robot to it's home position. Start the drawing thread
     */
        ur3eControl();

    /**
     * @brief Destructor for UR3e Control 
     * Joins and deletes the drawing thread.
     */
        ~ur3eControl();


    /**
     * @brief Moves the robot to a predefined home position 
     */
        void home_position();

    /**
     * @brief Commands the robot to follow a Cartesian path using the provided poses.
     * @param points Vector of waypoints to follow.
     */
        void move_along_cartesian_path(std::vector<geometry_msgs::msg::Pose> points);
    
     /**
     * @brief Reads the planned path of x,y,z coordinates provided in the .CSV, appending each point to a vector of points, that are scaled down and translated
     * to match the geometric constraints of our canvas 
     */   
        std::vector<geometry_msgs::msg::Pose> readCSVposes();

     /**
     * @brief publishes the points from the readCSVposes() function to the topic "waypoints_cloud" to verify the image that is being drawn is correct.
     * @param poses that come from the readCSVposes()
     */   
        void publishPointCloud(const std::vector<geometry_msgs::msg::Pose> &poses);

     /**
     * @brief Waits for a trigger from the pablo GUI, that verify that the UR3e is going to start drawing.
     */   
        void executeDrawing();

    /**
     * @breif Creating collision object
     */

     void avoidance();
    
    private:
    /**
     * @brief Callback function triggered when a path planning completion message is received.
     * @param msg Shared pointer to the Bool message indicating path planning status.
     */
        void pathPlanningCallback(const std_msgs::msg::Bool::SharedPtr msg);

     /**
     * @brief Callback function triggered when a start drawing message is received.
     * @param msg Shared pointer to the Bool message indicating start command.
     */
        void startDrawingCallback(const std_msgs::msg::Bool::SharedPtr msg);

    /**
     * @brief Converts degrees to radians.
     * @param degrees Angle in degrees.
     * @return Angle in radians.
     */
        constexpr double degToRad(double degrees);
    
    /// Subscriber for path planning completion status.
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pathPlanningSub_;
    
    /// Subscriber for start drawing command.
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr startDrawingSub_;

    /// Publisher for visualization markers.
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub;
    
    /// MoveIt MoveGroupInterface for controlling the robot arm.
        moveit::planning_interface::MoveGroupInterface move_group_interface_;

    /// Vector of waypoints read from the CSV file.    
        std::vector<geometry_msgs::msg::Pose> waypoints_;

    /// Thread for executing the drawing routine.  
        std::thread* drawing_thread_;

    /// Flag indicating whether to start the drawing routine.
        bool start_;
    
    /// Flag indicating whether path planning is complete.
        bool planningComplete_;

    /// Directory path to the CSV file containing waypoints.
        std::string csvDirectory_ = "/home/edan/git/41069_WS_LAB4_G1/pablo/output/waypoints.csv";
        //  std::string csvDirectory_ = "/home/niku/git/41069_WS_LAB4_G1/pablo/output/waypoints.csv";
        
    };

#endif // UR3E_CONTROL_H
