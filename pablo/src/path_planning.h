#ifndef PATH_PLANNING_H
#define PATH_PLANNING_H

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include <thread>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>

class pathPlanning : public rclcpp::Node {
public:
    pathPlanning();

    struct Waypoint {
        float x, y, z;
        float distance_from_last;

        Waypoint(float x_val, float y_val, float z_val, float dist)
            : x(x_val), y(y_val), z(z_val), distance_from_last(dist) {}
    };

    void tspSolver();

private:
    void imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg);
    float calculateDistance(const cv::Point& p1, const cv::Point& p2);
    void saveWaypointsToFile(const std::vector<Waypoint>& waypoints, const std::string& directory, const std::string& filename);

    std::string csvDirectory_ = "/home/edan/git/41069_WS_LAB4_G1/pablo/output";
    std::string csvFilename_ = "waypoints.csv";

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr imageProcessorSub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pathPlanningPub_;
};

#endif // PATH_PLANNING_H