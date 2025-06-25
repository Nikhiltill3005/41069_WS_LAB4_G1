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

/**
 * @struct Waypoint 
 * @brief Represents a  3D drawing point with a distance from the last waypoint. 
 */
    struct Waypoint {
        float x, y, z;
        float distance_from_last;
/**
     * @brief Constructs a new Waypoint object.
     * 
     * @param x_val X coordinate
     * @param y_val Y coordinate
     * @param z_val Z coordinate
     * @param dist Distance from the previous waypoint
 */
    Waypoint(float x_val, float y_val, float z_val, float dist)
        : x(x_val), y(y_val), z(z_val), distance_from_last(dist) {}
};

/**
 * @class pathPlanning
 * @brief A ROS 2 node that performs contour-based path planning from a sketch image.
 * 
 * The node listens for a boolean trigger to begin processing, extracts contours from 
 * a preprocessed image, optimizes the drawing order, and generates waypoints.
 */

class pathPlanning : public rclcpp::Node {
public:

    /**
     * @brief Construct a new pathPlanning node.
     * 
     * Initializes ROS publishers/subscribers and loads parameters.
     */
        pathPlanning();
    /**
     * @brief Main logic to detect contours, merge, optimize sequence, and generate waypoints.
     */
        void tspSolver();

private:

    /**
     * @brief Callback function triggered by a boolean message indicating when to plan a path.
     * 
     * @param msg Incoming Bool message used as a trigger.
     */
        void imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg);

    /**
     * @brief Utility function to compute Euclidean distance between two 2D points.
     * 
     * @param p1 First point.
     * @param p2 Second point.
     * @return float Distance between p1 and p2.
     */

        float calculateDistance(const cv::Point& p1, const cv::Point& p2);

    /**
     * @brief Saves generated waypoints into a CSV file.
     * 
     * @param contourwaypoints Nested vector of waypoints for each contour.
     * @param directory Output directory path.
     * @param filename Filename to save waypoints to.
     */
        void saveWaypointsToFile(const std::vector<std::vector<Waypoint>>& contourwaypoints, 
            const std::string& directory, const std::string& filename);

    /**
     * @brief Creates visualization overlays to verify contours follow the original white lines.
     * 
     * @param originalImage The original grayscale image with white lines
     * @param contours Vector of extracted contours
     * @param contour_order Order in which contours will be drawn
     */
        void visualizeContourOverlay(const cv::Mat& originalImage, 
                                   const std::vector<std::vector<cv::Point>>& contours,
                                   const std::vector<int>& contour_order);
        
    // === Parameters and State === //
    
    // std::string csvDirectory_ = "/home/sachinhanel/git/41069_WS_LAB4_G1/pablo/output";
    // std::string csvDirectory_ = "/home/niku/git/41069_WS_LAB4_G1/pablo/output";
    std::string csvDirectory_ = "/home/edan/git/41069_WS_LAB4_G1/pablo/output";
    std::string csvFilename_ = "waypoints.csv";

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr imageProcessorSub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pathPlanningPub_;

    int raiseZ_;
    int drawZ_;
    int penHeight_;
    int canvasHeight_;
};

#endif // PATH_PLANNING_H