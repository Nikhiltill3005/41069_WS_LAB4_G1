#include "path_planning.h"
#include <filesystem>
#include <queue>
#include <set>

// Custom Line Contour Extractor Class
class LineContourExtractor {
public:
    struct Point {
        int x, y;
        Point(int x = 0, int y = 0) : x(x), y(y) {}
        
        bool operator<(const Point& other) const {
            return x < other.x || (x == other.x && y < other.y);
        }
        
        double distanceTo(const Point& other) const {
            return std::sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
        }
        
        // Conversion to cv::Point
        cv::Point toCvPoint() const {
            return cv::Point(x, y);
        }
    };
    
    using Contour = std::vector<Point>;
    using ContourList = std::vector<Contour>;

private:
    // 8-connectivity neighbors (including diagonals)
    const std::vector<Point> neighbors = {
        {-1, -1}, {-1, 0}, {-1, 1},
        {0, -1},           {0, 1},
        {1, -1},  {1, 0},  {1, 1}
    };
    
    int whiteThreshold;
    int minContourLength;
    
public:
    LineContourExtractor(int whiteThreshold = 200, int minContourLength = 10)
        : whiteThreshold(whiteThreshold), minContourLength(minContourLength) {}
    
    /**
     * Extract contours and convert to cv::Point format for compatibility
     */
    std::vector<std::vector<cv::Point>> extractContoursAsCvPoints(const cv::Mat& image) {
        ContourList contours = extractContours(image);
        
        // Convert to cv::Point format for compatibility with existing code
        std::vector<std::vector<cv::Point>> cvContours;
        for (const auto& contour : contours) {
            std::vector<cv::Point> cvContour;
            for (const auto& point : contour) {
                cvContour.push_back(point.toCvPoint());
            }
            cvContours.push_back(cvContour);
        }
        
        return cvContours;
    }

private:
    /**
     * Extract contours from white lines on black background
     */
    ContourList extractContours(const cv::Mat& image) {
        cv::Mat grayImage;
        
        // Convert to grayscale if needed
        if (image.channels() == 3) {
            cv::cvtColor(image, grayImage, cv::COLOR_BGR2GRAY);
        } else {
            grayImage = image.clone();
        }
        
        // Create binary mask for white pixels
        cv::Mat whiteMask;
        cv::threshold(grayImage, whiteMask, whiteThreshold, 255, cv::THRESH_BINARY);
        
        // Skeletonize the binary image to ensure 1-pixel thick lines
        cv::Mat skeletonized = skeletonize(whiteMask);
        
        // Track visited pixels
        cv::Mat visited = cv::Mat::zeros(skeletonized.size(), CV_8UC1);
        
        ContourList contours;
        
        // Scan through all pixels
        for (int y = 0; y < skeletonized.rows; ++y) {
            for (int x = 0; x < skeletonized.cols; ++x) {
                if (skeletonized.at<uchar>(y, x) > 0 && visited.at<uchar>(y, x) == 0) {
                    // Found unvisited white pixel
                    Contour component = extractConnectedComponent(Point(x, y), skeletonized, visited);
                    
                    if (component.size() >= minContourLength) {
                        // Break component into line segments to avoid gaps
                        std::vector<Contour> lineSegments = orderPixelsAsLineSegments(component);
                        
                        // Add each segment as a separate contour if it's long enough
                        for (const auto& segment : lineSegments) {
                            if (segment.size() >= minContourLength) {
                                contours.push_back(segment);
                            }
                        }
                    }
                }
            }
        }
        
        return contours;
    }
    
    /**
     * Skeletonize binary image using morphological thinning
     * Reduces all white lines to 1-pixel thickness while preserving connectivity
     */
    cv::Mat skeletonize(const cv::Mat& binaryImage) {
        cv::Mat skeleton = cv::Mat::zeros(binaryImage.size(), CV_8UC1);
        cv::Mat temp, eroded;
        
        cv::Mat element = cv::getStructuringElement(cv::MORPH_CROSS, cv::Size(3, 3));
        
        binaryImage.copyTo(temp);
        
        bool done = false;
        while (!done) {
            cv::erode(temp, eroded, element);
            cv::dilate(eroded, temp, element);
            cv::subtract(binaryImage, temp, temp);
            cv::bitwise_or(skeleton, temp, skeleton);
            eroded.copyTo(temp);
            
            done = (cv::countNonZero(temp) == 0);
        }
        
        // Alternative implementation using Zhang-Suen algorithm for better results
        return zhangSuenThinning(binaryImage);
    }
    
    /**
     * Zhang-Suen thinning algorithm for better skeletonization
     * This produces cleaner single-pixel skeletons than morphological operations
     */
    cv::Mat zhangSuenThinning(const cv::Mat& binaryImage) {
        cv::Mat img = binaryImage.clone();
        img /= 255; // Convert to 0 and 1
        
        cv::Mat prev = cv::Mat::zeros(img.size(), CV_8UC1);
        cv::Mat diff;
        
        do {
            zhangSuenThinningIteration(img, 0);
            zhangSuenThinningIteration(img, 1);
            cv::absdiff(img, prev, diff);
            img.copyTo(prev);
        } while (cv::countNonZero(diff) > 0);
        
        img *= 255; // Convert back to 0 and 255
        return img;
    }
    
    /**
     * Single iteration of Zhang-Suen thinning
     */
    void zhangSuenThinningIteration(cv::Mat& img, int iter) {
        cv::Mat marker = cv::Mat::zeros(img.size(), CV_8UC1);
        
        for (int i = 1; i < img.rows - 1; i++) {
            for (int j = 1; j < img.cols - 1; j++) {
                uchar p2 = img.at<uchar>(i-1, j);
                uchar p3 = img.at<uchar>(i-1, j+1);
                uchar p4 = img.at<uchar>(i, j+1);
                uchar p5 = img.at<uchar>(i+1, j+1);
                uchar p6 = img.at<uchar>(i+1, j);
                uchar p7 = img.at<uchar>(i+1, j-1);
                uchar p8 = img.at<uchar>(i, j-1);
                uchar p9 = img.at<uchar>(i-1, j-1);
                
                int A = (p2 == 0 && p3 == 1) + (p3 == 0 && p4 == 1) + 
                        (p4 == 0 && p5 == 1) + (p5 == 0 && p6 == 1) + 
                        (p6 == 0 && p7 == 1) + (p7 == 0 && p8 == 1) +
                        (p8 == 0 && p9 == 1) + (p9 == 0 && p2 == 1);
                int B = p2 + p3 + p4 + p5 + p6 + p7 + p8 + p9;
                int m1 = iter == 0 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = iter == 0 ? (p4 * p6 * p8) : (p2 * p6 * p8);
                
                if (A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0)
                    marker.at<uchar>(i,j) = 1;
            }
        }
        
        img &= ~marker;
    }
    
    /**
     * Get valid unvisited white neighbors for a pixel
     */
    std::vector<Point> getValidNeighbors(const Point& point, const cv::Mat& whiteMask, const cv::Mat& visited) {
        std::vector<Point> validNeighbors;
        
        for (const Point& neighbor : neighbors) {
            Point newPoint(point.x + neighbor.x, point.y + neighbor.y);
            
            if (newPoint.x >= 0 && newPoint.x < whiteMask.cols &&
                newPoint.y >= 0 && newPoint.y < whiteMask.rows &&
                whiteMask.at<uchar>(newPoint.y, newPoint.x) > 0 &&
                visited.at<uchar>(newPoint.y, newPoint.x) == 0) {
                validNeighbors.push_back(newPoint);
            }
        }
        
        return validNeighbors;
    }
    
    /**
     * Extract connected component using BFS
     */
    Contour extractConnectedComponent(const Point& startPoint, const cv::Mat& whiteMask, cv::Mat& visited) {
        Contour component;
        std::queue<Point> queue;
        
        queue.push(startPoint);
        visited.at<uchar>(startPoint.y, startPoint.x) = 255;
        
        while (!queue.empty()) {
            Point current = queue.front();
            queue.pop();
            component.push_back(current);
            
            std::vector<Point> validNeighbors = getValidNeighbors(current, whiteMask, visited);
            for (const Point& neighbor : validNeighbors) {
                visited.at<uchar>(neighbor.y, neighbor.x) = 255;
                queue.push(neighbor);
            }
        }
        
        return component;
    }
    
    /**
     * Count neighbors within distance 1.5 for endpoint detection
     */
    int countCloseNeighbors(const Point& point, const Contour& points) {
        int count = 0;
        for (const Point& other : points) {
            if (point.x != other.x || point.y != other.y) {  // Skip self
                double distance = point.distanceTo(other);
                if (distance <= 1.5) {
                    count++;
                }
            }
        }
        return count;
    }
    
    /**
     * Order pixels to form continuous line segments, breaking at large gaps
     */
    std::vector<Contour> orderPixelsAsLineSegments(const Contour& pixels) {
        if (pixels.size() <= 2) {
            return {pixels};
        }
        
        const double MAX_CONNECTION_DISTANCE = 2.5; // Maximum distance to connect pixels
        
        // Find potential endpoints (points with fewer neighbors)
        std::vector<std::pair<size_t, int>> endpointCandidates;
        for (size_t i = 0; i < pixels.size(); ++i) {
            int neighborCount = countCloseNeighbors(pixels[i], pixels);
            if (neighborCount <= 2) {
                endpointCandidates.push_back({i, neighborCount});
            }
        }
        
        // Choose starting point
        size_t startIdx = 0;
        if (!endpointCandidates.empty()) {
            // Start from point with fewest neighbors
            auto minElement = std::min_element(endpointCandidates.begin(), endpointCandidates.end(),
                [](const std::pair<size_t, int>& a, const std::pair<size_t, int>& b) {
                    return a.second < b.second;
                });
            startIdx = minElement->first;
        }
        
        // Greedy path construction with gap detection
        std::vector<Contour> lineSegments;
        Contour currentSegment;
        std::set<size_t> remainingIndices;
        for (size_t i = 0; i < pixels.size(); ++i) {
            remainingIndices.insert(i);
        }
        
        size_t currentIdx = startIdx;
        
        while (!remainingIndices.empty()) {
            currentSegment.push_back(pixels[currentIdx]);
            remainingIndices.erase(currentIdx);
            
            if (remainingIndices.empty()) {
                // Add the final segment
                if (!currentSegment.empty()) {
                    lineSegments.push_back(currentSegment);
                }
                break;
            }
            
            // Find closest remaining point
            Point currentPoint = pixels[currentIdx];
            double minDistance = std::numeric_limits<double>::max();
            size_t closestIdx = *remainingIndices.begin();
            
            for (size_t idx : remainingIndices) {
                double distance = currentPoint.distanceTo(pixels[idx]);
                if (distance < minDistance) {
                    minDistance = distance;
                    closestIdx = idx;
                }
            }
            
            // Check if the closest point is too far away
            if (minDistance > MAX_CONNECTION_DISTANCE) {
                // Gap detected - finish current segment and start a new one
                if (!currentSegment.empty()) {
                    lineSegments.push_back(currentSegment);
                    currentSegment.clear();
                }
                
                // Find the best starting point for the next segment
                // Prefer endpoints from the remaining candidates
                bool foundEndpoint = false;
                for (const auto& candidate : endpointCandidates) {
                    if (remainingIndices.count(candidate.first)) {
                        currentIdx = candidate.first;
                        foundEndpoint = true;
                        break;
                    }
                }
                
                if (!foundEndpoint) {
                    // No endpoint candidates left, just use the first remaining point
                    currentIdx = *remainingIndices.begin();
                }
            } else {
                // Continue with the closest point
                currentIdx = closestIdx;
            }
        }
        
        return lineSegments;
    }
};

void pathPlanning::visualizeContourOverlay(const cv::Mat& originalImage, 
                                         const std::vector<std::vector<cv::Point>>& contours,
                                         const std::vector<int>& contour_order) {
    // Create visualization images
    cv::Mat overlayImage, colorCodedImage;
    
    // Convert grayscale to BGR for colored overlay
    if (originalImage.channels() == 1) {
        cv::cvtColor(originalImage, overlayImage, cv::COLOR_GRAY2BGR);
        cv::cvtColor(originalImage, colorCodedImage, cv::COLOR_GRAY2BGR);
    } else {
        overlayImage = originalImage.clone();
        colorCodedImage = originalImage.clone();
    }
    
    // 1. Simple overlay - draw all contours in bright red on original image
    std::cout << "Creating simple contour overlay..." << std::endl;
    for (const auto& contour : contours) {
        for (size_t i = 1; i < contour.size(); ++i) {
            cv::line(overlayImage, contour[i-1], contour[i], cv::Scalar(0, 0, 255), 1); // Bright red
        }
        // Mark start points with green circles
        if (!contour.empty()) {
            cv::circle(overlayImage, contour[0], 2, cv::Scalar(0, 255, 0), -1); // Green start point
        }
        // Mark end points with blue circles
        if (contour.size() > 1) {
            cv::circle(overlayImage, contour.back(), 2, cv::Scalar(255, 0, 0), -1); // Blue end point
        }
    }
    
    // 2. Color-coded overlay - show drawing order with gradient colors
    std::cout << "Creating color-coded contour overlay..." << std::endl;
    for (size_t i = 0; i < contour_order.size(); ++i) {
        int idx = contour_order[i];
        
        // Generate color based on drawing order (gradient from blue to red)
        float progress = static_cast<float>(i) / std::max(1.0f, static_cast<float>(contour_order.size() - 1));
        int blue = static_cast<int>(255 * (1.0f - progress));
        int red = static_cast<int>(255 * progress);
        cv::Scalar color(blue, 0, red);
        
        const auto& contour = contours[idx];
        
        // Draw contour as connected lines
        for (size_t j = 1; j < contour.size(); ++j) {
            cv::line(colorCodedImage, contour[j-1], contour[j], color, 2);
        }
        
        // Add contour number label
        if (!contour.empty()) {
            cv::Point center = contour[0];
            cv::putText(colorCodedImage, std::to_string(i), center, 
                       cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
    }
    
    // 3. Side-by-side comparison
    cv::Mat comparison;
    cv::hconcat(overlayImage, colorCodedImage, comparison);
    
    // Add labels
    cv::putText(comparison, "Contours on Original (Red=Lines, Green=Start, Blue=End)", 
               cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    cv::putText(comparison, "Drawing Order (Blue->Red = First->Last)", 
               cv::Point(overlayImage.cols + 10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 255, 255), 2);
    
    // Save visualization images
    std::filesystem::path output_dir(csvDirectory_);
    
    cv::imwrite((output_dir / "contour_overlay_simple.png").string(), overlayImage);
    cv::imwrite((output_dir / "contour_overlay_ordered.png").string(), colorCodedImage);
    cv::imwrite((output_dir / "contour_comparison.png").string(), comparison);
    
    std::cout << "Visualization images saved:" << std::endl;
    std::cout << "  - Simple overlay: " << (output_dir / "contour_overlay_simple.png").string() << std::endl;
    std::cout << "  - Ordered overlay: " << (output_dir / "contour_overlay_ordered.png").string() << std::endl;
    std::cout << "  - Side-by-side comparison: " << (output_dir / "contour_comparison.png").string() << std::endl;
    
    // Optional: Display images (uncomment if you want to see them during execution)
    /*
    cv::imshow("Contour Verification", comparison);
    cv::waitKey(3000); // Show for 3 seconds
    cv::destroyAllWindows();
    */
    
    // Print verification statistics
    std::cout << "\n=== CONTOUR VERIFICATION STATS ===" << std::endl;
    std::cout << "Total contours extracted: " << contours.size() << std::endl;
    
    size_t total_points = 0;
    size_t min_points = std::numeric_limits<size_t>::max();
    size_t max_points = 0;
    
    for (const auto& contour : contours) {
        total_points += contour.size();
        min_points = std::min(min_points, contour.size());
        max_points = std::max(max_points, contour.size());
    }
    
    std::cout << "Total points across all contours: " << total_points << std::endl;
    std::cout << "Average points per contour: " << (contours.empty() ? 0 : total_points / contours.size()) << std::endl;
    std::cout << "Shortest contour: " << (contours.empty() ? 0 : min_points) << " points" << std::endl;
    std::cout << "Longest contour: " << (contours.empty() ? 0 : max_points) << " points" << std::endl;
    std::cout << "=================================" << std::endl;
};

pathPlanning::pathPlanning() : Node("path_planning") {
    RCLCPP_INFO(this->get_logger(), "Path Planning Node Started");

    // Create subscribers and publishers
    imageProcessorSub_ = this->create_subscription<std_msgs::msg::Bool>(
        "image_processed", 10, std::bind(&pathPlanning::imageProcessedCallback, this, std::placeholders::_1));
    pathPlanningPub_ = this->create_publisher<std_msgs::msg::Bool>("path_planned", 10);

    // Set the height to raise the pen
    penHeight_ = 640; // 640/4000 = 0.16m = 160mm
    canvasHeight_ = 23; // 24/4000 = 0.006m = 6mm WITHIN( 40-45) optimal range
    drawZ_ = canvasHeight_ + penHeight_; // 664/4000 = 0.166m = 166mm
    raiseZ_ = drawZ_ + 30; // 100/4000 = 0.025m = 25mm
}

void pathPlanning::imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    try {
        if (msg->data) {
            RCLCPP_INFO(this->get_logger(), "Image processing complete. Starting path planning...");
            std::this_thread::sleep_for(std::chrono::milliseconds(5000));
            tspSolver(); // Start Path Planning
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
    std::filesystem::path image_path = std::filesystem::path(getenv("HOME")) / "git/41069_WS_LAB4_G1/pablo/output/6_sketch.jpg";

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

    // REPLACED: Use custom line contour extractor instead of OpenCV's findContours
    LineContourExtractor extractor(128, 5);  // white_threshold=128, min_contour_length=5
    std::vector<std::vector<cv::Point>> contours = extractor.extractContoursAsCvPoints(image);
    
    std::cout << "Initial contour count (custom extractor with skeletonization): " << contours.size() << std::endl;

    // Filter out very small contours (keeping your existing filter)
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
        
        // Sample points from both contours for efficiency
        int sample_rate = std::max(1, static_cast<int>(contour1.size() / 20)); 
        int sample_rate2 = std::max(1, static_cast<int>(contour2.size() / 20)); 
        
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
    float proximity_score_threshold = 0.15f;  // If 15% or more of sampled points are close, merge the contours
    
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

    // Variables to track jumps
    int jump_count = 0;
    float total_jump_distance = 0.0f;
    std::vector<std::pair<cv::Point, cv::Point>> jump_points;
    
    // Process all contours in optimal order to minimize jumps
    std::vector<int> contour_order;
    std::vector<bool> processed_contours(contours.size(), false);
    std::vector<bool> contour_reversed(contours.size(), false); // Track if contour is reversed
    
    // Vector to store waypoints grouped by contour
    std::vector<std::vector<Waypoint>> contour_waypoints;
    
    // Start with the first contour
    int current_contour_idx = 0;
    contour_order.push_back(current_contour_idx);
    processed_contours[current_contour_idx] = true;
    
    // Add first contour's points to waypoints (never reversed)
    std::vector<Waypoint> first_contour_waypoints;
    const auto& first_contour = contours[current_contour_idx];
    if (!first_contour.empty()) {
        // Add the first point with zero distance
        first_contour_waypoints.push_back(Waypoint(first_contour[0].x, first_contour[0].y, 0.0, 0.0));
        
        // Add remaining points from the first contour
        for (size_t i = 1; i < first_contour.size(); ++i) {
            float dist = calculateDistance(first_contour[i-1], first_contour[i]);
            first_contour_waypoints.push_back(Waypoint(first_contour[i].x, first_contour[i].y, 0.0, dist));
        }
    }
    contour_waypoints.push_back(first_contour_waypoints);
    
    // Process remaining contours with intelligent ordering
    int contours_processed = 1;  // Already processed the first one
    while (contours_processed < contours.size()) {
        // Get the endpoint of the current contour (considering if it was reversed)
        cv::Point current_endpoint;
        if (contour_reversed[current_contour_idx]) {
            current_endpoint = contours[current_contour_idx].front(); // Reversed, so end is original start
        } else {
            current_endpoint = contours[current_contour_idx].back();  // Normal direction
        }
        
        // Find the optimal next contour and connection type
        float min_distance = std::numeric_limits<float>::max();
        int best_contour_idx = -1;
        bool should_reverse_next = false;
        cv::Point best_start_point;
        
        for (size_t i = 0; i < contours.size(); ++i) {
            if (!processed_contours[i] && !contours[i].empty()) {
                const auto& candidate_contour = contours[i];
                
                // Option 1: Connect to start of candidate (normal direction)
                float dist_to_start = calculateDistance(current_endpoint, candidate_contour.front());
                
                // Option 2: Connect to end of candidate (reverse direction)
                float dist_to_end = calculateDistance(current_endpoint, candidate_contour.back());
                
                // Choose the best option for this candidate
                if (dist_to_start <= dist_to_end) {
                    // Normal direction is better
                    if (dist_to_start < min_distance) {
                        min_distance = dist_to_start;
                        best_contour_idx = i;
                        should_reverse_next = false;
                        best_start_point = candidate_contour.front();
                    }
                } else {
                    // Reverse direction is better
                    if (dist_to_end < min_distance) {
                        min_distance = dist_to_end;
                        best_contour_idx = i;
                        should_reverse_next = true;
                        best_start_point = candidate_contour.back(); // This becomes the start when reversed
                    }
                }
            }
        }
        
        if (best_contour_idx != -1) {
            // Record the jump
            jump_count++;
            total_jump_distance += min_distance;
            jump_points.push_back(std::make_pair(current_endpoint, best_start_point));
            
            std::cout << "Jump #" << jump_count << ": From contour " << current_contour_idx 
                    << " to contour " << best_contour_idx
                    << " (distance: " << min_distance << " pixels)"
                    << (should_reverse_next ? " [REVERSED]" : " [NORMAL]") << std::endl;
            
            // Create waypoints for this contour (considering direction)
            std::vector<Waypoint> contour_points;
            const auto& best_contour = contours[best_contour_idx];
            
            if (should_reverse_next) {
                // Add points in reverse order
                contour_reversed[best_contour_idx] = true;
                
                // Add first point (which is the original last point)
                contour_points.push_back(Waypoint(best_contour.back().x, best_contour.back().y, 0.0, min_distance));
                
                // Add remaining points in reverse order
                for (int i = best_contour.size() - 2; i >= 0; --i) {
                    float dist = calculateDistance(best_contour[i+1], best_contour[i]);
                    contour_points.push_back(Waypoint(best_contour[i].x, best_contour[i].y, 0.0, dist));
                }
            } else {
                // Add points in normal order
                contour_reversed[best_contour_idx] = false;
                
                // Add first point
                contour_points.push_back(Waypoint(best_contour.front().x, best_contour.front().y, 0.0, min_distance));
                
                // Add remaining points
                for (size_t i = 1; i < best_contour.size(); ++i) {
                    float dist = calculateDistance(best_contour[i-1], best_contour[i]);
                    contour_points.push_back(Waypoint(best_contour[i].x, best_contour[i].y, 0.0, dist));
                }
            }
            
            // Add this contour's waypoints to the list
            contour_waypoints.push_back(contour_points);
            
            // Update current contour and mark as processed
            current_contour_idx = best_contour_idx;
            contour_order.push_back(current_contour_idx);
            processed_contours[current_contour_idx] = true;
            contours_processed++;
        } else {
            // This shouldn't happen unless there's a logic error
            std::cerr << "Error: Could not find next contour." << std::endl;
            break;
        }
    }
    
    // Print jump statistics to console with reversal info
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Total jumps detected: " << jump_count << std::endl;
    std::cout << "Total jump distance: " << total_jump_distance << " pixels" << std::endl;
    std::cout << "Average jump distance: " << (jump_count > 0 ? total_jump_distance / jump_count : 0) << " pixels" << std::endl;
    std::cout << "Optimal contour order: ";
    for (size_t i = 0; i < contour_order.size(); ++i) {
        int idx = contour_order[i];
        std::cout << idx;
        if (i > 0 && contour_reversed[idx]) {
            std::cout << "R"; // Mark reversed contours
        }
        std::cout << " ";
    }
    std::cout << std::endl;
    
    // Count how many contours were reversed for efficiency
    int reversed_count = 0;
    for (size_t i = 1; i < contour_order.size(); ++i) { // Skip first contour (never reversed)
        if (contour_reversed[contour_order[i]]) {
            reversed_count++;
        }
    }
    std::cout << "Contours reversed for efficiency: " << reversed_count << " out of " << (contour_order.size() - 1) << std::endl;
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
    
    // Save waypoints to CSV with pen up/down movements
    saveWaypointsToFile(contour_waypoints, csvDirectory_, csvFilename_);
    
    // Visualization: Create overlay to verify contours follow the white lines
    visualizeContourOverlay(image, contours, contour_order);
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