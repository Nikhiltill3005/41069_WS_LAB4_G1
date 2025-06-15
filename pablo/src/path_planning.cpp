#include "path_planning.h"
#include <filesystem>

pathPlanning::pathPlanning() : Node("path_planning") {
    RCLCPP_INFO(this->get_logger(), "Path Planning Node Started");

    // Create subscribers and publishers
    imageProcessorSub_ = this->create_subscription<std_msgs::msg::Bool>(
        "image_processed", 10, std::bind(&pathPlanning::imageProcessedCallback, this, std::placeholders::_1));
    pathPlanningPub_ = this->create_publisher<std_msgs::msg::Bool>("path_planned", 10);

    // Set the height to raise the pen
    penHeight_ = 640; // 640/4000 = 0.16m = 160mm
    canvasHeight_ = 24; // 24/4000 = 0.006m = 6mm
    drawZ_ = canvasHeight_ + penHeight_; // 664/4000 = 0.166m = 166mm
    raiseZ_ = drawZ_ + 100; // 100/4000 = 0.025m = 25mm
    
    // Spline parameters
    splinePointSpacing_ = 10.0f; // Distance between sampled spline points (in pixels)
}

void pathPlanning::imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    try {
        if (msg->data) {
            RCLCPP_INFO(this->get_logger(), "Image processing complete. Starting path planning...");
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            
            tspSolverWithCenterlines();
            
            std_msgs::msg::Bool pathMsg;
            pathMsg.data = true;
            pathPlanningPub_->publish(pathMsg);
            RCLCPP_INFO(this->get_logger(), "Path planning complete. Starting drawing...");
        }
    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in imageProcessedCallback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in imageProcessedCallback");
    }
}

// Function to fit a spline to a contour and sample points at regular intervals
std::vector<cv::Point> pathPlanning::samplePointsAlongSpline(const std::vector<cv::Point>& contour, float spacing) {
    if (contour.size() < 2) {
        return contour; // Not enough points for a spline
    }
    
    // Fit a cubic spline to the contour points
    cv::Mat curve;
    cv::approxPolyDP(contour, curve, 1.0, false); // Approximate with a polygon first to reduce noise
    
    // Check if we have enough points for a smooth curve
    if (curve.rows < 4) {
        return contour; // Not enough points for a cubic spline
    }
    
    // Calculate the total length of the contour
    float total_length = 0.0f;
    for (size_t i = 1; i < contour.size(); i++) {
        total_length += calculateDistance(contour[i-1], contour[i]);
    }
    
    // Sample points along the contour at regular intervals
    std::vector<cv::Point> sampled_points;
    
    // Always include the first point
    sampled_points.push_back(contour.front());
    
    // Calculate number of points to sample based on the contour length
    int num_points = std::max(2, static_cast<int>(total_length / spacing));
    
    // Create a parametric representation of the curve
    std::vector<float> cumulative_lengths(contour.size());
    cumulative_lengths[0] = 0.0f;
    
    for (size_t i = 1; i < contour.size(); i++) {
        cumulative_lengths[i] = cumulative_lengths[i-1] + 
                               calculateDistance(contour[i-1], contour[i]);
    }
    
    // Normalize the cumulative lengths to [0, 1]
    for (size_t i = 0; i < cumulative_lengths.size(); i++) {
        cumulative_lengths[i] /= cumulative_lengths.back();
    }
    
    // Sample at regular intervals
    for (int i = 1; i < num_points - 1; i++) {
        float t = static_cast<float>(i) / (num_points - 1);
        
        // Find the segment that contains t
        size_t idx = 0;
        while (idx < cumulative_lengths.size() - 1 && cumulative_lengths[idx + 1] < t) {
            idx++;
        }
        
        // Interpolate between the points
        if (idx < contour.size() - 1) {
            float segment_t = (t - cumulative_lengths[idx]) / 
                             (cumulative_lengths[idx + 1] - cumulative_lengths[idx]);
            
            int x = static_cast<int>(contour[idx].x + segment_t * (contour[idx + 1].x - contour[idx].x));
            int y = static_cast<int>(contour[idx].y + segment_t * (contour[idx + 1].y - contour[idx].y));
            
            sampled_points.push_back(cv::Point(x, y));
        }
    }
    
    // Always include the last point
    sampled_points.push_back(contour.back());
    
    return sampled_points;
}

// Extract true centerlines instead of contours
std::vector<std::vector<cv::Point>> pathPlanning::extractCenterlines(const cv::Mat& binary_image) {
    std::cout << "Extracting centerlines using skeletonization..." << std::endl;
    
    // Step 1: Create skeleton using morphological operations
    cv::Mat skeleton = createSkeleton(binary_image);
    
    // Step 2: Trace the skeleton to get centerline paths
    std::vector<std::vector<cv::Point>> centerlines = traceSkeleton(skeleton);
    
    // Step 3: Clean up the centerlines
    centerlines = cleanCenterlines(centerlines);
    
    // Save skeleton image for debugging
    cv::imwrite(csvDirectory_ + "/skeleton.jpg", skeleton);
    
    return centerlines;
}

cv::Mat pathPlanning::createSkeleton(const cv::Mat& binary_image) {
    cv::Mat skeleton = cv::Mat::zeros(binary_image.size(), CV_8UC1);
    cv::Mat temp, eroded;
    
    // Get structuring element
    cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
    
    binary_image.copyTo(temp);
    
    // Iterative thinning
    bool done = false;
    while (!done) {
        cv::morphologyEx(temp, eroded, cv::MORPH_ERODE, element);
        cv::morphologyEx(eroded, temp, cv::MORPH_DILATE, element);
        cv::subtract(temp, eroded, temp);
        cv::bitwise_or(skeleton, temp, skeleton);
        
        eroded.copyTo(temp);
        
        // Check if we're done (no more pixels to erode)
        done = (cv::countNonZero(temp) == 0);
    }
    
    return skeleton;
}

std::vector<std::vector<cv::Point>> pathPlanning::traceSkeleton(const cv::Mat& skeleton) {
    std::vector<std::vector<cv::Point>> centerlines;
    cv::Mat visited = cv::Mat::zeros(skeleton.size(), CV_8UC1);
    
    // Find all endpoint and junction points
    std::vector<cv::Point> endpoints;
    std::vector<cv::Point> junctions;
    findSkeletonFeatures(skeleton, endpoints, junctions);
    
    std::cout << "Found " << endpoints.size() << " endpoints and " << junctions.size() << " junctions" << std::endl;
    
    // Start tracing from each endpoint
    for (const auto& endpoint : endpoints) {
        if (visited.at<uchar>(endpoint.y, endpoint.x) > 0) continue;
        
        std::vector<cv::Point> path = traceSkeletonPath(skeleton, visited, endpoint);
        
        if (path.size() > 5) { // Only keep paths with sufficient length
            centerlines.push_back(path);
        }
    }
    
    // Handle any remaining untraced skeleton pixels (loops, etc.)
    for (int y = 0; y < skeleton.rows; y++) {
        for (int x = 0; x < skeleton.cols; x++) {
            if (skeleton.at<uchar>(y, x) > 0 && visited.at<uchar>(y, x) == 0) {
                std::vector<cv::Point> path = traceSkeletonPath(skeleton, visited, cv::Point(x, y));
                if (path.size() > 5) {
                    centerlines.push_back(path);
                }
            }
        }
    }
    
    return centerlines;
}

void pathPlanning::findSkeletonFeatures(const cv::Mat& skeleton, 
                                       std::vector<cv::Point>& endpoints, 
                                       std::vector<cv::Point>& junctions) {
    // 8-connectivity offsets
    const int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    
    for (int y = 1; y < skeleton.rows - 1; y++) {
        for (int x = 1; x < skeleton.cols - 1; x++) {
            if (skeleton.at<uchar>(y, x) == 0) continue;
            
            // Count neighbors
            int neighbor_count = 0;
            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (skeleton.at<uchar>(ny, nx) > 0) {
                    neighbor_count++;
                }
            }
            
            if (neighbor_count == 1) {
                endpoints.push_back(cv::Point(x, y));
            } else if (neighbor_count > 2) {
                junctions.push_back(cv::Point(x, y));
            }
        }
    }
}

std::vector<cv::Point> pathPlanning::traceSkeletonPath(const cv::Mat& skeleton, 
                                                      cv::Mat& visited, 
                                                      const cv::Point& start) {
    std::vector<cv::Point> path;
    cv::Point current = start;
    
    // 8-connectivity offsets
    const int dx[] = {-1, -1, -1, 0, 0, 1, 1, 1};
    const int dy[] = {-1, 0, 1, -1, 1, -1, 0, 1};
    
    while (true) {
        // Add current point to path
        path.push_back(current);
        visited.at<uchar>(current.y, current.x) = 255;
        
        // Find next unvisited skeleton pixel
        cv::Point next(-1, -1);
        for (int i = 0; i < 8; i++) {
            int nx = current.x + dx[i];
            int ny = current.y + dy[i];
            
            if (nx >= 0 && nx < skeleton.cols && ny >= 0 && ny < skeleton.rows &&
                skeleton.at<uchar>(ny, nx) > 0 && visited.at<uchar>(ny, nx) == 0) {
                next = cv::Point(nx, ny);
                break;
            }
        }
        
        // If no next point found, end the path
        if (next.x == -1) break;
        
        current = next;
    }
    
    return path;
}

std::vector<std::vector<cv::Point>> pathPlanning::cleanCenterlines(const std::vector<std::vector<cv::Point>>& centerlines) {
    std::vector<std::vector<cv::Point>> cleaned;
    
    for (const auto& line : centerlines) {
        if (line.size() < 3) continue; // Skip very short lines
        
        // Remove duplicate consecutive points
        std::vector<cv::Point> deduplicated;
        deduplicated.push_back(line[0]);
        
        for (size_t i = 1; i < line.size(); i++) {
            if (calculateDistance(line[i], line[i-1]) > 0.5f) { // Only add if moved
                deduplicated.push_back(line[i]);
            }
        }
        
        if (deduplicated.size() >= 3) {
            cleaned.push_back(deduplicated);
        }
    }
    
    return cleaned;
}

// Modified tspSolver to use centerline extraction with full visualization
void pathPlanning::tspSolverWithCenterlines() {
    // Load the image
    std::filesystem::path image_path = std::filesystem::path(getenv("HOME")) / "pablo/output/2_sketch.jpg";

    if (!std::filesystem::exists(image_path)) {
        std::cerr << "File does not exist: " << image_path << std::endl;
        return;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not load the image!" << std::endl;
        return;
    }

    // Apply binary threshold
    cv::Mat binary;
    cv::threshold(image, binary, 128, 255, cv::THRESH_BINARY);
    
    // Extract centerlines instead of contours
    std::vector<std::vector<cv::Point>> centerlines = extractCenterlines(binary);
    
    std::cout << "Initial centerline count: " << centerlines.size() << std::endl;

    // Filter out very small centerlines
    std::vector<std::vector<cv::Point>> filtered_centerlines;
    for (const auto& centerline : centerlines) {
        if (centerline.size() > 5) {  // Filter centerlines with fewer than 5 points
            filtered_centerlines.push_back(centerline);
        }
    }
    centerlines = filtered_centerlines;
    
    // Create image to visualize original centerlines
    cv::Mat originalCenterlineImage;
    cv::cvtColor(image, originalCenterlineImage, cv::COLOR_GRAY2BGR);
    
    // Draw original centerlines with different colors
    for (size_t i = 0; i < centerlines.size(); i++) {
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
        for (size_t j = 1; j < centerlines[i].size(); j++) {
            cv::line(originalCenterlineImage, centerlines[i][j-1], centerlines[i][j], color, 2);
        }
        
        // Mark start and end points
        if (!centerlines[i].empty()) {
            cv::circle(originalCenterlineImage, centerlines[i][0], 4, cv::Scalar(0, 255, 0), -1); // Green for start
            cv::circle(originalCenterlineImage, centerlines[i].back(), 4, cv::Scalar(0, 0, 255), -1); // Red for end
        }
    }
    
    cv::imwrite(std::string(getenv("HOME")) + "/pablo/output/original_centerlines.jpg", originalCenterlineImage);
    
    // Create spline-smoothed centerlines
    std::vector<std::vector<cv::Point>> smoothed_centerlines;
    for (const auto& centerline : centerlines) {
        smoothed_centerlines.push_back(samplePointsAlongSpline(centerline, splinePointSpacing_));
    }
    
    // Create image to visualize smoothed centerlines
    cv::Mat smoothedCenterlineImage;
    cv::cvtColor(image, smoothedCenterlineImage, cv::COLOR_GRAY2BGR);
    
    // Draw smoothed centerlines with different colors
    for (size_t i = 0; i < smoothed_centerlines.size(); i++) {
        // Create a random color
        cv::Scalar color(rand() & 255, rand() & 255, rand() & 255);
        
        // Draw the spline-smoothed centerline
        for (size_t j = 1; j < smoothed_centerlines[i].size(); j++) {
            cv::line(smoothedCenterlineImage, smoothed_centerlines[i][j-1], smoothed_centerlines[i][j], color, 2);
        }
        
        // Draw the sampled points
        for (const auto& pt : smoothed_centerlines[i]) {
            cv::circle(smoothedCenterlineImage, pt, 2, cv::Scalar(0, 0, 255), -1);
        }
    }
    
    cv::imwrite(std::string(getenv("HOME")) + "/pablo/output/smoothed_centerlines.jpg", smoothedCenterlineImage);
    
    // Function to calculate proximity score between two centerlines
    auto calculateCenterlineProximity = [this](const std::vector<cv::Point>& centerline1, const std::vector<cv::Point>& centerline2, 
                                       float threshold, int& c1_idx, int& c2_idx) -> float {
        // Check how many points in centerline1 are close to any point in centerline2
        int close_point_count = 0;
        int total_comparisons = 0;
        float min_distance = std::numeric_limits<float>::max();
        
        // Sample points from both centerlines for efficiency
        int sample_rate = std::max(1, static_cast<int>(centerline1.size() / 20)); 
        int sample_rate2 = std::max(1, static_cast<int>(centerline2.size() / 20)); 
        
        for (size_t i = 0; i < centerline1.size(); i += sample_rate) {
            for (size_t j = 0; j < centerline2.size(); j += sample_rate2) {
                float dist = calculateDistance(centerline1[i], centerline2[j]);
                total_comparisons++;
                
                if (dist < threshold) {
                    close_point_count++;
                }
                
                if (dist < min_distance) {
                    min_distance = dist;
                    c1_idx = i;
                    c2_idx = j;
                }
            }
        }
        
        // Proximity score is the percentage of points that are close
        return static_cast<float>(close_point_count) / total_comparisons;
    };
    
    // Now work with the smoothed centerlines
    std::vector<std::vector<cv::Point>> contours = smoothed_centerlines;
    
    // First do endpoint-based merging (within 4 pixels)
    bool centerlines_merged = true;
    float endpoint_merge_threshold = 4.0f;
    
    std::cout << "Starting endpoint-based centerline concatenation..." << std::endl;
    std::cout << "Initial number of centerlines: " << contours.size() << std::endl;
    
    // Keep merging until no more endpoint merges are possible
    while (centerlines_merged && contours.size() > 1) {
        centerlines_merged = false;
        
        // Find the closest pair of centerlines based on endpoints
        float min_distance = std::numeric_limits<float>::max();
        int centerline1_idx = -1;
        int centerline2_idx = -1;
        bool connect_end_to_start = true;  // Whether to connect end of centerline1 to start of centerline2
        
        for (size_t i = 0; i < contours.size(); ++i) {
            for (size_t j = i + 1; j < contours.size(); ++j) {
                // Check distance from end of centerline i to start of centerline j
                float dist_end_to_start = calculateDistance(contours[i].back(), contours[j].front());
                
                // Check distance from end of centerline j to start of centerline i
                float dist_end_to_start_reverse = calculateDistance(contours[j].back(), contours[i].front());
                
                // Check distance from start of centerline i to start of centerline j
                float dist_start_to_start = calculateDistance(contours[i].front(), contours[j].front());
                
                // Check distance from end of centerline i to end of centerline j
                float dist_end_to_end = calculateDistance(contours[i].back(), contours[j].back());
                
                // Find the minimum of these distances
                float min_dist_between_centerlines = std::min({dist_end_to_start, dist_end_to_start_reverse, 
                                                         dist_start_to_start, dist_end_to_end});
                
                if (min_dist_between_centerlines < min_distance) {
                    min_distance = min_dist_between_centerlines;
                    centerline1_idx = i;
                    centerline2_idx = j;
                    
                    // Determine how to connect these centerlines
                    if (min_dist_between_centerlines == dist_end_to_start) {
                        connect_end_to_start = true;  // Connect end of i to start of j
                    } else if (min_dist_between_centerlines == dist_end_to_start_reverse) {
                        connect_end_to_start = false; // Connect end of j to start of i
                    } else if (min_dist_between_centerlines == dist_start_to_start) {
                        // Need to reverse centerline i
                        std::reverse(contours[i].begin(), contours[i].end());
                        connect_end_to_start = true;  // Now connect end of i to start of j
                    } else if (min_dist_between_centerlines == dist_end_to_end) {
                        // Need to reverse centerline j
                        std::reverse(contours[j].begin(), contours[j].end());
                        connect_end_to_start = true;  // Now connect end of i to start of j
                    }
                }
            }
        }
        
        // If the closest centerlines are within the threshold, merge them
        if (min_distance <= endpoint_merge_threshold && centerline1_idx != -1 && centerline2_idx != -1) {
            std::cout << "Merging centerlines " << centerline1_idx << " and " << centerline2_idx 
                      << " (endpoint distance: " << min_distance << " pixels)" << std::endl;
            
            std::vector<cv::Point> merged_centerline;
            
            if (connect_end_to_start) {
                // Connect end of centerline1 to start of centerline2
                merged_centerline = contours[centerline1_idx];
                merged_centerline.insert(merged_centerline.end(), contours[centerline2_idx].begin(), contours[centerline2_idx].end());
            } else {
                // Connect end of centerline2 to start of centerline1
                merged_centerline = contours[centerline2_idx];
                merged_centerline.insert(merged_centerline.end(), contours[centerline1_idx].begin(), contours[centerline1_idx].end());
            }
            
            // Replace centerline1 with the merged centerline and remove centerline2
            contours[centerline1_idx] = merged_centerline;
            contours.erase(contours.begin() + centerline2_idx);
            
            centerlines_merged = true;
        }
    }
    
    std::cout << "Number of centerlines after endpoint-based merging: " << contours.size() << std::endl;
    
    // Now perform proximity-based merging (centerlines with many close points)
    std::cout << "Starting proximity-based centerline concatenation..." << std::endl;
    
    centerlines_merged = true;
    float proximity_threshold = 8.0f;  // Points closer than this are considered "close"
    float proximity_score_threshold = 0.15f;  // If 15% or more of sampled points are close, merge the centerlines
    
    // Keep merging until no more proximity-based merges are possible
    while (centerlines_merged && contours.size() > 1) {
        centerlines_merged = false;
        
        // Find the pair of centerlines with highest proximity score
        float max_proximity_score = 0.0f;
        int centerline1_idx = -1;
        int centerline2_idx = -1;
        int best_c1_idx = 0;
        int best_c2_idx = 0;
        
        for (size_t i = 0; i < contours.size(); ++i) {
            for (size_t j = i + 1; j < contours.size(); ++j) {
                int c1_idx = 0, c2_idx = 0;
                float proximity_score = calculateCenterlineProximity(contours[i], contours[j], 
                                                                 proximity_threshold, c1_idx, c2_idx);
                
                if (proximity_score > max_proximity_score) {
                    max_proximity_score = proximity_score;
                    centerline1_idx = i;
                    centerline2_idx = j;
                    best_c1_idx = c1_idx;
                    best_c2_idx = c2_idx;
                }
            }
        }
        
        // If proximity score is high enough, merge the centerlines
        if (max_proximity_score >= proximity_score_threshold && centerline1_idx != -1 && centerline2_idx != -1) {
            std::cout << "Merging centerlines " << centerline1_idx << " and " << centerline2_idx 
                      << " (proximity score: " << max_proximity_score << ")" << std::endl;
            
            std::vector<cv::Point> merged_centerline;
            
            // Rearrange centerlines for optimal connection
            // Split and reconnect the centerlines at the closest points
            std::vector<cv::Point> c1_first_part(contours[centerline1_idx].begin(), 
                                               contours[centerline1_idx].begin() + best_c1_idx);
            std::vector<cv::Point> c1_second_part(contours[centerline1_idx].begin() + best_c1_idx, 
                                                contours[centerline1_idx].end());
            
            std::vector<cv::Point> c2_first_part(contours[centerline2_idx].begin(), 
                                               contours[centerline2_idx].begin() + best_c2_idx);
            std::vector<cv::Point> c2_second_part(contours[centerline2_idx].begin() + best_c2_idx, 
                                                contours[centerline2_idx].end());
            
            // Create the merged centerline by connecting at the closest points
            merged_centerline = c1_first_part;
            merged_centerline.insert(merged_centerline.end(), c2_second_part.begin(), c2_second_part.end());
            merged_centerline.insert(merged_centerline.end(), c2_first_part.rbegin(), c2_first_part.rend());
            merged_centerline.insert(merged_centerline.end(), c1_second_part.rbegin(), c1_second_part.rend());
            
            // Replace centerline1 with the merged centerline and remove centerline2
            contours[centerline1_idx] = merged_centerline;
            contours.erase(contours.begin() + centerline2_idx);
            
            centerlines_merged = true;
        }
    }
    
    std::cout << "Final number of centerlines after concatenation: " << contours.size() << std::endl;

    // Variables to track jumps
    int jump_count = 0;
    float total_jump_distance = 0.0f;
    std::vector<std::pair<cv::Point, cv::Point>> jump_points;
    
    // Process all centerlines in optimal order to minimize jumps
    std::vector<int> centerline_order;
    std::vector<bool> processed_centerlines(contours.size(), false);
    
    // Vector to store waypoints grouped by centerline
    std::vector<std::vector<Waypoint>> centerline_waypoints;
    
    // Start with the first centerline
    int current_centerline_idx = 0;
    centerline_order.push_back(current_centerline_idx);
    processed_centerlines[current_centerline_idx] = true;
    
    // Add first centerline's points to waypoints
    std::vector<Waypoint> first_centerline_waypoints;
    const auto& first_centerline = contours[current_centerline_idx];
    if (!first_centerline.empty()) {
        // Add the first point with zero distance
        first_centerline_waypoints.push_back(Waypoint(first_centerline[0].x, first_centerline[0].y, 0.0, 0.0));
        
        // Add remaining points from the first centerline
        for (size_t i = 1; i < first_centerline.size(); ++i) {
            float dist = calculateDistance(first_centerline[i-1], first_centerline[i]);
            first_centerline_waypoints.push_back(Waypoint(first_centerline[i].x, first_centerline[i].y, 0.0, dist));
        }
    }
    centerline_waypoints.push_back(first_centerline_waypoints);
    
    // Process all centerlines in optimal order to minimize jumps
    int centerlines_processed = 1;  // Already processed the first one
    while (centerlines_processed < contours.size()) {
        // Get the last point of the current centerline (endpoint)
        cv::Point current_endpoint = contours[current_centerline_idx].back();
        
        // Find the closest startpoint of an unprocessed centerline
        float min_distance = std::numeric_limits<float>::max();
        int closest_centerline_idx = -1;
        cv::Point closest_start_point;
        
        for (size_t i = 0; i < contours.size(); ++i) {
            if (!processed_centerlines[i] && !contours[i].empty()) {
                // Check distance to the start point of this centerline
                float dist = calculateDistance(current_endpoint, contours[i][0]);
                if (dist < min_distance) {
                    min_distance = dist;
                    closest_centerline_idx = i;
                    closest_start_point = contours[i][0];
                }
            }
        }
        
        if (closest_centerline_idx != -1) {
            // Record the jump
            jump_count++;
            total_jump_distance += min_distance;
            jump_points.push_back(std::make_pair(current_endpoint, closest_start_point));
            
            std::cout << "Jump #" << jump_count << ": From centerline " << current_centerline_idx 
                    << " to centerline " << closest_centerline_idx
                    << " (distance: " << min_distance << " pixels)" << std::endl;
            
            // Create waypoints for this centerline
            std::vector<Waypoint> centerline_points;
            const auto& closest_centerline = contours[closest_centerline_idx];
            
            // Add first point (with jump distance)
            centerline_points.push_back(Waypoint(closest_start_point.x, closest_start_point.y, 0.0, min_distance));
            
            // Add remaining points
            for (size_t i = 1; i < closest_centerline.size(); ++i) {
                float dist = calculateDistance(closest_centerline[i-1], closest_centerline[i]);
                centerline_points.push_back(Waypoint(closest_centerline[i].x, closest_centerline[i].y, 0.0, dist));
            }
            
            // Add this centerline's waypoints to the list
            centerline_waypoints.push_back(centerline_points);
            
            // Update current centerline and mark as processed
            current_centerline_idx = closest_centerline_idx;
            centerline_order.push_back(current_centerline_idx);
            processed_centerlines[current_centerline_idx] = true;
            centerlines_processed++;
        } else {
            // This shouldn't happen unless there's a logic error
            std::cerr << "Error: Could not find next centerline." << std::endl;
            break;
        }
    }
    
    // Print jump statistics to console
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total jumps detected: " << jump_count << std::endl;
    std::cout << "Total jump distance: " << total_jump_distance << " pixels" << std::endl;
    std::cout << "Average jump distance: " << (jump_count > 0 ? total_jump_distance / jump_count : 0) << " pixels" << std::endl;
    std::cout << "Optimal centerline order: ";
    for (int idx : centerline_order) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Draw centerlines in order with gradient color
    cv::Mat orderedCenterlineImage = cv::Mat::zeros(image.size(), CV_8UC3);
    for (size_t i = 0; i < centerline_order.size(); ++i) {
        int idx = centerline_order[i];
        // Generate a color based on position in order (gradient from blue to red)
        int blue = 255 - static_cast<int>(255 * (i / static_cast<float>(centerline_order.size())));
        int red = static_cast<int>(255 * (i / static_cast<float>(centerline_order.size())));
        cv::Scalar color(blue, 0, red);
        
        // Draw the centerline
        const auto& centerline = contours[idx];
        for (size_t j = 1; j < centerline.size(); j++) {
            cv::line(orderedCenterlineImage, centerline[j-1], centerline[j], color, 2);
        }
        
        // Add centerline number
        cv::Point center = contours[idx][0]; // Use first point for label
        cv::putText(orderedCenterlineImage, std::to_string(i), center, 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    // Draw jumps as green lines
    for (const auto& jump : jump_points) {
        cv::line(orderedCenterlineImage, jump.first, jump.second, cv::Scalar(0, 255, 0), 1);
    }
    
    // Save the visualization
    std::string vis_path = std::string(getenv("HOME")) + "/pablo/output/centerline_path.jpg";
    cv::imwrite(vis_path, orderedCenterlineImage);
    RCLCPP_INFO(this->get_logger(), "Centerline path visualization saved to: %s", vis_path.c_str());
    
    // Save waypoints to CSV with pen up/down movements
    saveWaypointsToFile(centerline_waypoints, csvDirectory_, csvFilename_);
}

float pathPlanning::calculateDistance(const cv::Point& p1, const cv::Point& p2) {
    return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

// Updated function to save waypoints with pen up/down movements
void pathPlanning::saveWaypointsToFile(const std::vector<std::vector<Waypoint>>& contour_waypoints, 
                                     const std::string& directory, const std::string& filename) {
    // Construct the full file path
    std::filesystem::path dir_path(directory);
    std::filesystem::path file_path = dir_path / filename;

    // Check if the directory exists
    if (!std::filesystem::exists(dir_path)) {
        if (!std::filesystem::create_directories(dir_path)) {
            std::cerr << "Failed to create directory: " << directory << std::endl;
            return;
        }
    }

    std::ofstream file(file_path);
    if (file.is_open()) {
        file << "x,y,z,distance_from_last\n"; // CSV header
        
        for (size_t i = 0; i < contour_waypoints.size(); ++i) {
            const auto& waypoints = contour_waypoints[i];
            
            if (waypoints.empty()) continue;
            
            // First point of a contour - add with pen lowering sequence
            if (i > 0) { // Not the first contour
                // Get last point of previous contour and first point of current contour
                const Waypoint& prev_end = contour_waypoints[i-1].back();
                const Waypoint& curr_start = waypoints[0];
                
                // 1. Raise pen at the end of previous contour
                file << prev_end.x << "," << prev_end.y << "," << raiseZ_ << "," << 0.0 << " # Raise pen\n";
                
                // 2. Move to above the starting point of the next contour
                float travel_dist = std::sqrt(std::pow(curr_start.x - prev_end.x, 2) + 
                                             std::pow(curr_start.y - prev_end.y, 2));
                file << curr_start.x << "," << curr_start.y << "," << raiseZ_ << "," 
                     << travel_dist << " # Move to next contour\n";
                
                // 3. Lower pen at the start of the new contour
                file << curr_start.x << "," << curr_start.y << "," << drawZ_ << "," 
                     << 0.0 << " # Lower pen\n";
            }
            
            // Write all points in this contour
            for (size_t j = 0; j < waypoints.size(); ++j) {
                const auto& wp = waypoints[j];
                // For the first point of the first contour, ensure it starts at drawing height
                if (i == 0 && j == 0) {
                    file << wp.x << "," << wp.y << "," << drawZ_ << "," << wp.distance_from_last << "\n";
                } else {
                    file << wp.x << "," << wp.y << "," << drawZ_ << "," << wp.distance_from_last << "\n";
                }
            }
        }
        
        // After the last contour, raise the pen
        if (!contour_waypoints.empty() && !contour_waypoints.back().empty()) {
            const Waypoint& last_wp = contour_waypoints.back().back();
            file << last_wp.x << "," << last_wp.y << "," << raiseZ_ << "," << 0.0 << " # Final pen raise\n";
        }
        
        file.close();
        std::cout << "Waypoints saved to " << file_path << " with pen up/down movements" << std::endl;
    } else {
        std::cerr << "Unable to open file for saving waypoints!" << std::endl;
    }
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<pathPlanning>());
    rclcpp::shutdown();
    return 0;
}