#ifndef UR3E_CONTROL_H
#define UR3E_CONTROL_H

#include <memory>
#include <fstream>
#include <sstream>
#include <rclcpp/rclcpp.hpp>
#include "std_msgs/msg/bool.hpp"
#include <moveit/move_group_interface/move_group_interface.h>
#include <moveit/robot_trajectory/robot_trajectory.h>
#include <moveit/trajectory_processing/iterative_time_parameterization.h>
#include <geometry_msgs/msg/pose.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
using namespace std;

class ur3eControl : public rclcpp::Node {
    public:
        ur3eControl();
        ~ur3eControl();

        void home_position();
        void move_along_cartesian_path(std::vector<geometry_msgs::msg::Pose> points);
        std::vector<geometry_msgs::msg::Pose> generate_target_poses();
        void printCurrentPosition();
        std::vector<geometry_msgs::msg::Pose> readCSVposes();
        void publishMarkers(const std::vector<geometry_msgs::msg::Pose> &poses);
        void publishPointCloud(const std::vector<geometry_msgs::msg::Pose> &poses);

        void executeDrawing();
    
    private:
        void pathPlanningCallback(const std_msgs::msg::Bool::SharedPtr msg);
        void startDrawingCallback(const std_msgs::msg::Bool::SharedPtr msg);

        constexpr double degToRad(double degrees);
    
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pathPlanningSub_;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr startDrawingSub_;

        rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub;
        rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_pub;
        moveit::planning_interface::MoveGroupInterface move_group_interface_;

        std::vector<geometry_msgs::msg::Pose> waypoints_;

        std::thread* drawing_thread_;

        bool start_;
        bool planningComplete_;

        std::string csvDirectory_ = "/home/edan/git/41069_WS_LAB4_G1/pablo/output/waypoints.csv";
    };

#endif // UR3E_CONTROL_H