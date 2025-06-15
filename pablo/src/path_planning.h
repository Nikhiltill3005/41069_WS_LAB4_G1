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
#include <filesystem>

/**
 * @struct Waypoint 
 * @brief Represents a 3D drawing point with a distance from the last waypoint. 
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
 * @brief A ROS 2 node that performs centerline-based path planning from a sketch image.
 * 
 * The node listens for a boolean trigger to begin processing, extracts centerlines from 
 * a preprocessed image using skeletonization, optimizes the drawing order using greedy 
 * nearest-neighbor algorithm with centerline merging, and generates waypoints with pen 
 * up/down movements.
 */
class pathPlanning : public rclcpp::Node {
public:
    /**
     * @brief Construct a new pathPlanning node.
     * 
     * Initializes ROS publishers/subscribers, loads parameters, and sets up
     * pen height configurations for drawing operations.
     */
    pathPlanning();

private:
    /**
     * @brief Callback function triggered by image processing completion.
     * 
     * @param msg Incoming Bool message indicating image processing is complete
     */
    void imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg);

    /**
     * @brief Utility function to compute Euclidean distance between two 2D points.
     * 
     * @param p1 First point
     * @param p2 Second point
     * @return float Distance between p1 and p2
     */
    float calculateDistance(const cv::Point& p1, const cv::Point& p2);

    /**
     * @brief Saves generated waypoints into a CSV file with pen up/down movements.
     * 
     * Exports waypoints organized by centerline with proper Z-height values for
     * pen raising/lowering between centerlines and travel movements.
     * 
     * @param contour_waypoints Nested vector of waypoints grouped by centerline
     * @param directory Output directory path
     * @param filename Filename to save waypoints to
     */
    void saveWaypointsToFile(const std::vector<std::vector<Waypoint>>& contour_waypoints, 
                           const std::string& directory, const std::string& filename);
    
    /**
     * @brief Applies spline smoothing to a centerline and samples points at regular intervals.
     * 
     * Takes a pixelated centerline and creates a smoother version by sampling points
     * along the curve at specified intervals, reducing noise and creating more
     * natural drawing movements.
     * 
     * @param contour Input centerline points
     * @param spacing Distance between sampled points along the spline
     * @return std::vector<cv::Point> Smoothed centerline with regularly spaced points
     */
    std::vector<cv::Point> samplePointsAlongSpline(const std::vector<cv::Point>& contour, float spacing);

    /**
     * @brief Extracts true centerlines from binary image using skeletonization
     * @param binary_image Input binary image with white lines on black background
     * @return Vector of centerline paths (single lines down the middle)
     */
    std::vector<std::vector<cv::Point>> extractCenterlines(const cv::Mat& binary_image);
    
    /**
     * @brief Creates skeleton using morphological thinning
     */
    cv::Mat createSkeleton(const cv::Mat& binary_image);
    
    /**
     * @brief Traces skeleton to extract connected paths
     */
    std::vector<std::vector<cv::Point>> traceSkeleton(const cv::Mat& skeleton);
    
    /**
     * @brief Finds endpoints and junctions in skeleton
     */
    void findSkeletonFeatures(const cv::Mat& skeleton, 
                             std::vector<cv::Point>& endpoints, 
                             std::vector<cv::Point>& junctions);
    
    /**
     * @brief Traces a single path through the skeleton
     */
    std::vector<cv::Point> traceSkeletonPath(const cv::Mat& skeleton, 
                                           cv::Mat& visited, 
                                           const cv::Point& start);
    
    /**
     * @brief Cleans up traced centerlines
     */
    std::vector<std::vector<cv::Point>> cleanCenterlines(const std::vector<std::vector<cv::Point>>& centerlines);
    
    /**
     * @brief Main path planning algorithm using centerline extraction
     * 
     * Loads sketch image, extracts centerlines using skeletonization, performs endpoint 
     * and proximity-based merging, optimizes centerline traversal order, and generates 
     * waypoints with proper pen movements. Includes comprehensive visualization.
     */
    void tspSolverWithCenterlines();
        
    // === ROS Communication === //
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr imageProcessorSub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pathPlanningPub_;
    
    // === File Paths === //
    std::string csvDirectory_ = "/home/sachinhanel/git/41069_WS_LAB4_G1/pablo/output";
    std::string csvFilename_ = "waypoints.csv";

    // === Robot Configuration Parameters === //
    
    /**
     * @brief Z-coordinate for raised pen position (pen up for travel)
     * 
     * Calculated as drawZ_ + 100 units for safe clearance during movement
     */
    int raiseZ_;
    
    /**
     * @brief Z-coordinate for drawing position (pen touching paper)
     * 
     * Calculated as canvasHeight_ + penHeight_ for proper contact
     */
    int drawZ_;
    
    /**
     * @brief Height of the pen mechanism from robot base (in encoder units)
     * 
     * Default: 640 units (160mm when scaled by 1/4000)
     */
    int penHeight_;
    
    /**
     * @brief Height of the canvas/paper from robot base (in encoder units)
     * 
     * Default: 24 units (6mm when scaled by 1/4000)
     */
    int canvasHeight_;
    
    // === Path Planning Parameters === //
    
    /**
     * @brief Distance between points when sampling spline curves (in pixels)
     * 
     * Controls the resolution of smoothed centerline paths. Lower values create
     * more detailed paths, higher values create simplified paths.
     */
    float splinePointSpacing_;
};

#endif // PATH_PLANNING_H