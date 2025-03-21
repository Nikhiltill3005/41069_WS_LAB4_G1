#include "path_planning.h"
#include <filesystem>

pathPlanning::pathPlanning() : Node("path_planning") {
    RCLCPP_INFO(this->get_logger(), "Path Planning Node Started");

    // Create subscribers and publishers
    imageProcessorSub_ = this->create_subscription<std_msgs::msg::Bool>(
        "image_processed", 10, std::bind(&pathPlanning::imageProcessedCallback, this, std::placeholders::_1));
    pathPlanningPub_ = this->create_publisher<std_msgs::msg::Bool>("path_planned", 10);
}

void pathPlanning::imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    try {
        if (msg->data) {
            RCLCPP_INFO(this->get_logger(), "Image processing complete. Starting path planning...");
            tspSolver(); // Start Path Planning
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            
            // Publish message to indicate path planning is complete
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

void pathPlanning::tspSolver(){
    // Load the image
    std::filesystem::path image_path = std::filesystem::path(getenv("HOME")) / "pablo/output/2_sketch.jpg";

    // Check if the file exists
    if (!std::filesystem::exists(image_path)) {
        std::cerr << "File does not exist: " << image_path << std::endl;
        return;
    }

    cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (image.empty()) {
        std::cerr << "Could not load the image!" << std::endl;
        return;
    }

    // Apply Canny edge detection
    cv::Mat edges;
    cv::threshold(image, edges, 128, 255, cv::THRESH_BINARY);
    
    // Display the threshold result (optional for debugging)
    // cv::namedWindow("Thresholded Image", cv::WINDOW_NORMAL);
    // cv::imshow("Thresholded Image", edges);
    // cv::waitKey(100); // Brief pause to show the image

    // Find contours in the thresholded image 
    // Using RETR_LIST instead of RETR_EXTERNAL to get all contours including inner ones
    // Using CHAIN_APPROX_NONE to get all contour points
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(edges, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_NONE);
    
    std::cout << "Initial contour count: " << contours.size() << std::endl;


    // Filter out very small contours (optional)
    std::vector<std::vector<cv::Point>> filtered_contours;
    for (const auto& contour : contours) {
        if (contour.size() > 5) {  // Filter contours with fewer than 5 points
            filtered_contours.push_back(contour);
        }
    }
    contours = filtered_contours;
    
    // Function to calculate proximity score between two contours
    auto calculateContourProximity = [this](const std::vector<cv::Point>& contour1, const std::vector<cv::Point>& contour2, 
                                       float threshold, int& c1_idx, int& c2_idx) -> float {
        // Check how many points in contour1 are close to any point in contour2
        int close_point_count = 0;
        int total_comparisons = 0;
        float min_distance = std::numeric_limits<float>::max();
        
        // Sample points from both contours for efficiency (use more points for better accuracy)
        int sample_rate = std::max(1, static_cast<int>(contour1.size() / 20)); // Check about 20 points from contour1
        int sample_rate2 = std::max(1, static_cast<int>(contour2.size() / 20)); // Check about 20 points from contour2
        
        for (size_t i = 0; i < contour1.size(); i += sample_rate) {
            for (size_t j = 0; j < contour2.size(); j += sample_rate2) {
                float dist = calculateDistance(contour1[i], contour2[j]);
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
    
    // First do endpoint-based merging (within 4 pixels)
    bool contours_merged = true;
    float endpoint_merge_threshold = 4.0f;
    
    std::cout << "Starting endpoint-based contour concatenation..." << std::endl;
    std::cout << "Initial number of contours: " << contours.size() << std::endl;
    
    // Keep merging until no more endpoint merges are possible
    while (contours_merged && contours.size() > 1) {
        contours_merged = false;
        
        // Find the closest pair of contours based on endpoints
        float min_distance = std::numeric_limits<float>::max();
        int contour1_idx = -1;
        int contour2_idx = -1;
        bool connect_end_to_start = true;  // Whether to connect end of contour1 to start of contour2
        
        for (size_t i = 0; i < contours.size(); ++i) {
            for (size_t j = i + 1; j < contours.size(); ++j) {
                // Check distance from end of contour i to start of contour j
                float dist_end_to_start = calculateDistance(contours[i].back(), contours[j].front());
                
                // Check distance from end of contour j to start of contour i
                float dist_end_to_start_reverse = calculateDistance(contours[j].back(), contours[i].front());
                
                // Check distance from start of contour i to start of contour j
                float dist_start_to_start = calculateDistance(contours[i].front(), contours[j].front());
                
                // Check distance from end of contour i to end of contour j
                float dist_end_to_end = calculateDistance(contours[i].back(), contours[j].back());
                
                // Find the minimum of these distances
                float min_dist_between_contours = std::min({dist_end_to_start, dist_end_to_start_reverse, 
                                                         dist_start_to_start, dist_end_to_end});
                
                if (min_dist_between_contours < min_distance) {
                    min_distance = min_dist_between_contours;
                    contour1_idx = i;
                    contour2_idx = j;
                    
                    // Determine how to connect these contours
                    if (min_dist_between_contours == dist_end_to_start) {
                        connect_end_to_start = true;  // Connect end of i to start of j
                    } else if (min_dist_between_contours == dist_end_to_start_reverse) {
                        connect_end_to_start = false; // Connect end of j to start of i
                    } else if (min_dist_between_contours == dist_start_to_start) {
                        // Need to reverse contour i
                        std::reverse(contours[i].begin(), contours[i].end());
                        connect_end_to_start = true;  // Now connect end of i to start of j
                    } else if (min_dist_between_contours == dist_end_to_end) {
                        // Need to reverse contour j
                        std::reverse(contours[j].begin(), contours[j].end());
                        connect_end_to_start = true;  // Now connect end of i to start of j
                    }
                }
            }
        }
        
        // If the closest contours are within the threshold, merge them
        if (min_distance <= endpoint_merge_threshold && contour1_idx != -1 && contour2_idx != -1) {
            std::cout << "Merging contours " << contour1_idx << " and " << contour2_idx 
                      << " (endpoint distance: " << min_distance << " pixels)" << std::endl;
            
            std::vector<cv::Point> merged_contour;
            
            if (connect_end_to_start) {
                // Connect end of contour1 to start of contour2
                merged_contour = contours[contour1_idx];
                merged_contour.insert(merged_contour.end(), contours[contour2_idx].begin(), contours[contour2_idx].end());
            } else {
                // Connect end of contour2 to start of contour1
                merged_contour = contours[contour2_idx];
                merged_contour.insert(merged_contour.end(), contours[contour1_idx].begin(), contours[contour1_idx].end());
            }
            
            // Replace contour1 with the merged contour and remove contour2
            contours[contour1_idx] = merged_contour;
            contours.erase(contours.begin() + contour2_idx);
            
            contours_merged = true;
        }
    }
    
    std::cout << "Number of contours after endpoint-based merging: " << contours.size() << std::endl;
    
    // Now perform proximity-based merging (contours with many close points)
    std::cout << "Starting proximity-based contour concatenation..." << std::endl;
    
    contours_merged = true;
    float proximity_threshold = 8.0f;  // Points closer than this are considered "close"
    float proximity_score_threshold = 0.15f;  // If 40% or more of sampled points are close, merge the contours
    
    // Keep merging until no more proximity-based merges are possible
    while (contours_merged && contours.size() > 1) {
        contours_merged = false;
        
        // Find the pair of contours with highest proximity score
        float max_proximity_score = 0.0f;
        int contour1_idx = -1;
        int contour2_idx = -1;
        int best_c1_idx = 0;
        int best_c2_idx = 0;
        
        for (size_t i = 0; i < contours.size(); ++i) {
            for (size_t j = i + 1; j < contours.size(); ++j) {
                int c1_idx = 0, c2_idx = 0;
                float proximity_score = calculateContourProximity(contours[i], contours[j], 
                                                                 proximity_threshold, c1_idx, c2_idx);
                
                if (proximity_score > max_proximity_score) {
                    max_proximity_score = proximity_score;
                    contour1_idx = i;
                    contour2_idx = j;
                    best_c1_idx = c1_idx;
                    best_c2_idx = c2_idx;
                }
            }
        }
        
        // If proximity score is high enough, merge the contours
        if (max_proximity_score >= proximity_score_threshold && contour1_idx != -1 && contour2_idx != -1) {
            std::cout << "Merging contours " << contour1_idx << " and " << contour2_idx 
                      << " (proximity score: " << max_proximity_score << ")" << std::endl;
            
            std::vector<cv::Point> merged_contour;
            
            // Rearrange contours for optimal connection
            // Split and reconnect the contours at the closest points
            std::vector<cv::Point> c1_first_part(contours[contour1_idx].begin(), 
                                               contours[contour1_idx].begin() + best_c1_idx);
            std::vector<cv::Point> c1_second_part(contours[contour1_idx].begin() + best_c1_idx, 
                                                contours[contour1_idx].end());
            
            std::vector<cv::Point> c2_first_part(contours[contour2_idx].begin(), 
                                               contours[contour2_idx].begin() + best_c2_idx);
            std::vector<cv::Point> c2_second_part(contours[contour2_idx].begin() + best_c2_idx, 
                                                contours[contour2_idx].end());
            
            // Create the merged contour by connecting at the closest points
            merged_contour = c1_first_part;
            merged_contour.insert(merged_contour.end(), c2_second_part.begin(), c2_second_part.end());
            merged_contour.insert(merged_contour.end(), c2_first_part.rbegin(), c2_first_part.rend());
            merged_contour.insert(merged_contour.end(), c1_second_part.rbegin(), c1_second_part.rend());
            
            // Replace contour1 with the merged contour and remove contour2
            contours[contour1_idx] = merged_contour;
            contours.erase(contours.begin() + contour2_idx);
            
            contours_merged = true;
        }
    }
    
    std::cout << "Final number of contours after concatenation: " << contours.size() << std::endl;

    std::cout << "Found " << contours.size() << " contours after filtering." << std::endl;
    
    // Create a structure to track which contours have been processed
    std::vector<bool> processed_contours(contours.size(), false);
    
    // Vector to store the optimal order of contours
    std::vector<int> contour_order;
    
    // Vector to store all waypoints
    std::vector<Waypoint> waypoints;
    
    // Variables to track jumps
    int jump_count = 0;
    float total_jump_distance = 0.0f;
    std::vector<std::pair<cv::Point, cv::Point>> jump_points;
    
    // Start with the first contour
    int current_contour_idx = 0;
    contour_order.push_back(current_contour_idx);
    processed_contours[current_contour_idx] = true;
    
    // Add first contour's points to waypoints
    const auto& first_contour = contours[current_contour_idx];
    if (!first_contour.empty()) {
        // Add the first point with zero distance
        waypoints.push_back(Waypoint(first_contour[0].x, first_contour[0].y, 0.0, 0.0));
        
        // Add remaining points from the first contour
        for (size_t i = 1; i < first_contour.size(); ++i) {
            float dist = calculateDistance(first_contour[i-1], first_contour[i]);
            waypoints.push_back(Waypoint(first_contour[i].x, first_contour[i].y, 0.0, dist));
        }
    }
    
    // Process all contours in optimal order to minimize jumps
    int contours_processed = 1;  // Already processed the first one
    while (contours_processed < contours.size()) {
        // Get the last point of the current contour (endpoint)
        cv::Point current_endpoint = contours[current_contour_idx].back();
        
        // Find the closest startpoint of an unprocessed contour
        float min_distance = std::numeric_limits<float>::max();
        int closest_contour_idx = -1;
        cv::Point closest_start_point;
        
        for (size_t i = 0; i < contours.size(); ++i) {
            if (!processed_contours[i] && !contours[i].empty()) {
                // Check distance to the start point of this contour
                float dist = calculateDistance(current_endpoint, contours[i][0]);
                if (dist < min_distance) {
                    min_distance = dist;
                    closest_contour_idx = i;
                    closest_start_point = contours[i][0];
                }
            }
        }
        
        if (closest_contour_idx != -1) {
            // Record the jump
            jump_count++;
            total_jump_distance += min_distance;
            jump_points.push_back(std::make_pair(current_endpoint, closest_start_point));
            
            std::cout << "Jump #" << jump_count << ": From contour " << current_contour_idx 
                      << " to contour " << closest_contour_idx
                      << " (distance: " << min_distance << " pixels)" << std::endl;
            
            // Add jump waypoint
            waypoints.push_back(Waypoint(closest_start_point.x, closest_start_point.y, 0.0, min_distance));
            
            // Add all points from the closest contour
            const auto& closest_contour = contours[closest_contour_idx];
            for (size_t i = 1; i < closest_contour.size(); ++i) {
                float dist = calculateDistance(closest_contour[i-1], closest_contour[i]);
                waypoints.push_back(Waypoint(closest_contour[i].x, closest_contour[i].y, 0.0, dist));
            }
            
            // Update current contour and mark as processed
            current_contour_idx = closest_contour_idx;
            contour_order.push_back(current_contour_idx);
            processed_contours[current_contour_idx] = true;
            contours_processed++;
        } else {
            // This shouldn't happen unless there's a logic error
            std::cerr << "Error: Could not find next contour." << std::endl;
            break;
        }
    }
    
    // Print jump statistics to console
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total jumps detected: " << jump_count << std::endl;
    std::cout << "Total jump distance: " << total_jump_distance << " pixels" << std::endl;
    std::cout << "Average jump distance: " << (jump_count > 0 ? total_jump_distance / jump_count : 0) << " pixels" << std::endl;
    std::cout << "Optimal contour order: ";
    for (int idx : contour_order) {
        std::cout << idx << " ";
    }
    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // Draw the robot's path on the original image
    cv::Mat pathImage;
    cv::cvtColor(image, pathImage, cv::COLOR_GRAY2BGR);

    // Draw contours in order with gradient color
    cv::Mat orderedContourImage = cv::Mat::zeros(image.size(), CV_8UC3);
    for (size_t i = 0; i < contour_order.size(); ++i) {
        int idx = contour_order[i];
        // Generate a color based on position in order (gradient from blue to red)
        int blue = 255 - static_cast<int>(255 * (i / static_cast<float>(contour_order.size())));
        int red = static_cast<int>(255 * (i / static_cast<float>(contour_order.size())));
        cv::Scalar color(blue, 0, red);
        
        cv::drawContours(orderedContourImage, contours, idx, color, 2);
        
        // Add contour number
        cv::Point center = contours[idx][0]; // Use first point for label
        cv::putText(orderedContourImage, std::to_string(i), center, 
                   cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    }
    
    // Draw jumps as green lines
    for (const auto& jump : jump_points) {
        cv::line(orderedContourImage, jump.first, jump.second, cv::Scalar(0, 255, 0), 1);
    }
    
    // Draw all waypoints as a continuous path
    for (size_t i = 1; i < waypoints.size(); ++i) {
        cv::Scalar line_color;
        
        // Check if this point was a jump
        bool is_jump = false;
        for (const auto& jump : jump_points) {
            if (waypoints[i].x == jump.second.x && waypoints[i].y == jump.second.y) {
                is_jump = true;
                break;
            }
        }
        
        if (is_jump) {
            line_color = cv::Scalar(0, 255, 0);  // Green for jumps
        } else {
            // Regular path points get a color gradient
            int color_value = static_cast<int>(255 * (i / static_cast<float>(waypoints.size())));
            line_color = cv::Scalar(color_value, 0, 255 - color_value);
        }
        
        cv::line(pathImage, 
                cv::Point(waypoints[i-1].x, waypoints[i-1].y),
                cv::Point(waypoints[i].x, waypoints[i].y),
                line_color, 1);
    }
    
    // Add jump statistics to the bottom of the image
    int img_height = pathImage.rows;
    std::stringstream ss;
    ss << "Jumps: " << jump_count << " | Total Distance: " << std::fixed << std::setprecision(1) << total_jump_distance 
       << " px | Avg Distance: " << std::fixed << std::setprecision(1) << (jump_count > 0 ? total_jump_distance / jump_count : 0) << " px";
    cv::putText(pathImage, ss.str(), cv::Point(10, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
    cv::putText(orderedContourImage, ss.str(), cv::Point(10, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);

    // Save waypoints to CSV
    saveWaypointsToFile(waypoints, csvDirectory_, csvFilename_);
    
    // Create windows for visualization
    // std::string pathWindowName = "Optimized Path";
    // cv::namedWindow(pathWindowName, cv::WINDOW_NORMAL);
    // std::string contourWindowName = "Contour Order";
    // cv::namedWindow(contourWindowName, cv::WINDOW_NORMAL);
    
    // Show the path and ordered contours
    // cv::imshow(pathWindowName, pathImage);
    // cv::imshow(contourWindowName, orderedContourImage);
    // cv::waitKey(0);
    
    // // Display each contour one at a time in the optimal order
    // std::cout << "Press any key to display each contour in optimal order." << std::endl;
    
    // for (size_t i = 0; i < contour_order.size(); ++i) {
    //     int idx = contour_order[i];
        
    //     // Create a clean image for this contour
    //     cv::Mat contourImage = cv::Mat::zeros(image.size(), CV_8UC3);
        
    //     // Draw previous contours in dark gray
    //     for (size_t j = 0; j < i; ++j) {
    //         cv::drawContours(contourImage, contours, contour_order[j], cv::Scalar(40, 40, 40), 1);
    //     }
        
    //     // Draw current contour in bright color
    //     cv::Scalar color(0, 255, 255); // Yellow
    //     cv::drawContours(contourImage, contours, idx, color, 2);
        
    //     // Draw the jump to this contour if it's not the first one
    //     if (i > 0) {
    //         cv::Point prev_end = contours[contour_order[i-1]].back();
    //         cv::Point curr_start = contours[idx][0];
    //         cv::line(contourImage, prev_end, curr_start, cv::Scalar(0, 255, 0), 1);
    //     }
        
    //     // Draw contour information
    //     cv::putText(contourImage, "Contour " + std::to_string(i+1) + "/" + std::to_string(contour_order.size()), 
    //                cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
    //     cv::putText(contourImage, "Points: " + std::to_string(contours[idx].size()), 
    //                cv::Point(10, 60), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
        
    //     if (i > 0) {
    //         float jump_dist = calculateDistance(contours[contour_order[i-1]].back(), contours[idx][0]);
    //         cv::putText(contourImage, "Jump from previous: " + std::to_string(jump_dist) + " px", 
    //                    cv::Point(10, 90), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    //     }
        
    //     // Add jump statistics to this image too
    //     cv::putText(contourImage, ss.str(), cv::Point(10, img_height - 10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
        
    //     // Display the image
    //     cv::imshow(contourWindowName, contourImage);
        
    //     // Wait for key press
    //     int key = cv::waitKey(0);
    //     if (key == 27) {  // ESC key
    //         break;
    //     }
    // }
    
    // // Final display with all contours in order
    // cv::Mat finalImage = orderedContourImage.clone();
    // cv::putText(finalImage, "All contours in optimal order! Press any key to exit.", 
    //            cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
    // cv::imshow(contourWindowName, finalImage);
    // cv::waitKey(0);
}

float pathPlanning::calculateDistance(const cv::Point& p1, const cv::Point& p2) {
    return std::sqrt(std::pow(p2.x - p1.x, 2) + std::pow(p2.y - p1.y, 2));
}

void pathPlanning::saveWaypointsToFile(const std::vector<Waypoint>& waypoints, const std::string& directory, const std::string& filename) {
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
        file << "x,y,z,distance_from_last\n";
        for (const auto& wp : waypoints) {
            file << wp.x << "," << wp.y << "," << wp.z << "," << wp.distance_from_last << "\n";
        }
        file.close();
        std::cout << "Waypoints saved to " << file_path << std::endl;
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