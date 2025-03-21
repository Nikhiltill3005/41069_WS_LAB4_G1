#include "ur3e_control.h"

ur3eControl::ur3eControl() 
: Node("ur3e_control", rclcpp::NodeOptions().automatically_declare_parameters_from_overrides(true)),
move_group_interface_(std::shared_ptr<rclcpp::Node>(this), "ur_manipulator") {
    RCLCPP_INFO(this->get_logger(), "UR3e Move Node Started");
    
    // Set parameters
    move_group_interface_.setMaxVelocityScalingFactor(0.9);  // Full velocity
    move_group_interface_.setMaxAccelerationScalingFactor(0.9);  // Full acceleration
    start_ = false;
    planningComplete_ = false;
    home_position();

    // Create both marker and pointcloud publishers
    marker_pub = this->create_publisher<visualization_msgs::msg::Marker>("visualization_marker", 1000);
    point_cloud_pub = this->create_publisher<sensor_msgs::msg::PointCloud2>("waypoints_cloud", 10);

    // Create subscribers
    pathPlanningSub_ = this->create_subscription<std_msgs::msg::Bool>(
        "path_planned", 10, std::bind(&ur3eControl::pathPlanningCallback, this, std::placeholders::_1));
    startDrawingSub_ = this->create_subscription<std_msgs::msg::Bool>(
        "starter", 10, std::bind(&ur3eControl::startDrawingCallback, this, std::placeholders::_1));

    // Start the drawing thread
    drawing_thread_ = new std::thread(&ur3eControl::executeDrawing, this);
    std::this_thread::sleep_for(std::chrono::seconds(1));
}

ur3eControl::~ur3eControl() {
    drawing_thread_->join();
    delete drawing_thread_;
}

void ur3eControl::pathPlanningCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    try {
        if (msg->data == true) {
            RCLCPP_INFO(this->get_logger(), "Path planning complete.");
            waypoints_ = readCSVposes();
            RCLCPP_INFO(this->get_logger(), "Waiting for start command...");
            planningComplete_ = true;
        }
    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in imageProcessedCallback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in imageProcessedCallback");
    }
}

void ur3eControl::startDrawingCallback(const std_msgs::msg::Bool::SharedPtr msg) {
    try {
        if (msg->data == true) {
            RCLCPP_INFO(this->get_logger(), "Starting drawing...");
            if(planningComplete_ = true){
                start_ = true;
            }
            else{
                RCLCPP_ERROR(this->get_logger(), "Path planning not complete!");
            }
        }
    } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(), "Exception in startDrawingCallback: %s", e.what());
    } catch (...) {
        RCLCPP_ERROR(this->get_logger(), "Unknown exception in startDrawingCallback");
    }
}

void ur3eControl::executeDrawing() {
    // While thread is running
    while (rclcpp::ok()) {
        // Move robot while start_ is true
        while(!start_) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        RCLCPP_INFO(this->get_logger(), "Moving Robot...");
        publishPointCloud(waypoints_);
        move_along_cartesian_path(waypoints_);
    }
}

void ur3eControl::home_position() {
    // Set the joint positions for the home position
    std::vector<double> joint_positions = {-1.5708, -1.62316, -2.46091, -0.628319, -4.71239, 0.0};  

    // Plan movement
    move_group_interface_.setJointValueTarget(joint_positions);
    moveit::planning_interface::MoveGroupInterface::Plan joint_plan;
    auto success = static_cast<bool>(move_group_interface_.plan(joint_plan));

    // Execute the plan
    if (success) {
        move_group_interface_.execute(joint_plan);
        RCLCPP_INFO(this->get_logger(), "Robot moved to initial joint positions!");
    } else {
        RCLCPP_ERROR(this->get_logger(), "Planning failed for initial joint positions!");
    }
}

void ur3eControl::move_along_cartesian_path(std::vector<geometry_msgs::msg::Pose> points) {
    // Use CSV poses instead of hardcoded ones
    std::cout << "reading csv files" << '\n';
    std::cout << "waypoints size is " << points.size() << endl;
    // std::vector<geometry_msgs::msg::Pose> waypoints = generate_target_poses();
    moveit_msgs::msg::RobotTrajectory trajectory;
    const double eef_step = 0.01;  // Step size in meters (small values = smoother motion)
    const double jump_threshold = 0.0; // 0 to disable jump detection

    // Compute Cartesian Path
    double fraction = move_group_interface_.computeCartesianPath(points, eef_step, jump_threshold, trajectory);

    if (fraction > 0.9) { 
        // Apply velocity scaling to the trajectory
        robot_trajectory::RobotTrajectory rt(move_group_interface_.getRobotModel(), "ur_manipulator");
        rt.setRobotTrajectoryMsg(*move_group_interface_.getCurrentState(), trajectory);
        
        trajectory_processing::IterativeParabolicTimeParameterization iptp;
        iptp.computeTimeStamps(rt, 0.05, 0.05); // velocity and acceleration scaling
        
        rt.getRobotTrajectoryMsg(trajectory);
        
        RCLCPP_INFO(this->get_logger(), "Cartesian path planned successfully! Executing...");
        move_group_interface_.execute(trajectory);
    } else {
        RCLCPP_ERROR(this->get_logger(), "Cartesian path planning failed! Only %.2f%% succeeded.", fraction * 100);
    }
    
    // Move back to home position
    start_ = false;
    home_position();
    RCLCPP_INFO(this->get_logger(), "Drawing complete. Robot moved back to home position.");
}

std::vector<geometry_msgs::msg::Pose> generate_target_poses() {
    std::vector<geometry_msgs::msg::Pose> poses;

    geometry_msgs::msg::Pose pose1;
    
    pose1.position.x = 0.25;
    pose1.position.y = 0.25;
    pose1.position.z = 0.1;
    pose1.orientation.w = 0.0;  // Ensure a valid orientation
    pose1.orientation.x = 1.0;
    pose1.orientation.y = 0.0;
    pose1.orientation.z = 0.0;  // Slightly above surface to avoid collision
    poses.push_back(pose1);
    
    geometry_msgs::msg::Pose pose2 = pose1;
    pose2.position.x = 0.0;
    pose2.position.y = 0.25;
    pose2.position.z = 0.1;
    pose2.orientation.x = 1.0;
    pose2.orientation.y = 0.0;
    pose2.orientation.z = 0.0;
    poses.push_back(pose2);

    geometry_msgs::msg::Pose pose3 = pose1;
    pose3.position.x = -0.25;
    pose3.position.y = 0.25;
    pose3.position.z = 0.1;
    pose3.orientation.x = 1.0;
    pose3.orientation.y = 0.0;
    pose3.orientation.z = 0.0;
    poses.push_back(pose3);

    geometry_msgs::msg::Pose pose4 = pose1;
    pose4.position.x = 0.25;
    pose4.position.y = 0.30;
    pose4.position.z = 0.1;  // Slightly above surface to avoid collision
    pose4.orientation.w = 0.0;  // Ensure a valid orientation
    pose4.orientation.x = 1.0;
    pose4.orientation.y = 0.0;
    pose4.orientation.z = 0.0;
    poses.push_back(pose4);

    geometry_msgs::msg::Pose pose5 = pose1;
    pose5.position.x = 0.0;
    pose5.position.y = 0.45;
    pose5.position.z = 0.1;
    pose5.orientation.x = 1.0;
    pose5.orientation.y = 0.0;
    pose5.orientation.z = 0.0;
    poses.push_back(pose5);

    geometry_msgs::msg::Pose pose6 = pose1;
    pose6.position.x = -0.25;
    pose6.position.y = 0.45;
    pose6.position.z = 0.1;
    pose6.orientation.x = 1.0;
    pose6.orientation.y = 0.0;
    pose6.orientation.z = 0.0;
    poses.push_back(pose6);

    return poses;
}

void ur3eControl::printCurrentPosition() {
    geometry_msgs::msg::PoseStamped current_pos = move_group_interface_.getCurrentPose();
    RCLCPP_INFO(this->get_logger(), "Current Position -> X: %.3f, Y: %.3f, Z: %.3f",
                current_pos.pose.position.x, 
                current_pos.pose.position.y, 
                current_pos.pose.position.z);
}

std::vector<geometry_msgs::msg::Pose> ur3eControl::readCSVposes() {
    std::vector<geometry_msgs::msg::Pose> poses;
    std::ifstream file("/home/edan/git/41069_WS_LAB4_G1/pablo/output/waypoints.csv");
    std::string line;
    int row_count = 0;

    if (!file.is_open()) {
        std::cout << "Failed to open waypoints.csv file" << std::endl;
        return poses;
    }

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
        geometry_msgs::msg::Pose targetPoint;
        std::stringstream ss(line);
        std::string x_str, y_str, z_str;

        // Extract values from the line
        if (std::getline(ss, x_str, ',') &&
            std::getline(ss, y_str, ',') &&
            std::getline(ss, z_str, ','))
        {
            row_count++;
            float x = std::stof(x_str);
            float y = std::stof(y_str);
            float z = std::stof(z_str);

            targetPoint.position.x = (x/4000+0.1);
            targetPoint.position.y = (y/4000)+0.25;
            targetPoint.position.z = z/4000;

            // Set a default orientation
            targetPoint.orientation.x = 1.0;
            targetPoint.orientation.y = 0.0;
            targetPoint.orientation.z = 0.0;
            targetPoint.orientation.w = 0.0;

            poses.push_back(targetPoint);
        } 
        else 
        {
            std::cout << "Error parsing line: " << line << std::endl;
        }
    }
    
    std::cout<< "The size of points is: " <<poses.size() << '\n';
    return poses;
}

void ur3eControl::publishMarkers(const std::vector<geometry_msgs::msg::Pose> &poses) {
    int id = 0;
    for (const auto &pose : poses)
    {
        visualization_msgs::msg::Marker marker;
        marker.header.frame_id = "world";  // Ensure this frame matches your TF setup
        marker.header.stamp = this->get_clock()->now();
        marker.ns = "waypoints";
        marker.id = id++;
        marker.type = visualization_msgs::msg::Marker::CYLINDER;
        marker.action = visualization_msgs::msg::Marker::ADD;

        marker.pose.position.x = pose.position.x;
        marker.pose.position.y = pose.position.y;
        marker.pose.position.z = pose.position.z;

        // Set the scale (size) of the sphere
        marker.scale.x = 0.5;
        marker.scale.y = 0.5;
        marker.scale.z = 0.5;

        // Set the color (green)
        marker.color.r = 0.0f;
        marker.color.g = 1.0f;
        marker.color.b = 0.0f;
        marker.color.a = 1.0;  // Fully opaque

        marker.lifetime = rclcpp::Duration::from_seconds(0);

        marker_pub->publish(marker);
    }

    std::cout << "The amount of markers are: " << id << endl;
}

void ur3eControl::publishPointCloud(const std::vector<geometry_msgs::msg::Pose> &poses) {
    // Create a PointCloud2 message
    sensor_msgs::msg::PointCloud2 cloud;
    
    // Set up the header
    cloud.header.stamp = this->get_clock()->now();
    cloud.header.frame_id = "world";  // Use the same frame as your markers
    
    // Set up basic point cloud parameters
    cloud.height = 1;  // Unorganized point cloud
    cloud.width = poses.size();  // Number of points
    cloud.is_dense = true;
    cloud.is_bigendian = false;
    
    // Set up the fields
    sensor_msgs::PointCloud2Modifier modifier(cloud);
    modifier.setPointCloud2Fields(4,
        "x", 1, sensor_msgs::msg::PointField::FLOAT32,
        "y", 1, sensor_msgs::msg::PointField::FLOAT32,
        "z", 1, sensor_msgs::msg::PointField::FLOAT32,
        "rgb", 1, sensor_msgs::msg::PointField::FLOAT32);  // For color information
        
    // Resize the point cloud
    modifier.resize(poses.size());
    
    // Create iterators for the point cloud data
    sensor_msgs::PointCloud2Iterator<float> iter_x(cloud, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(cloud, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(cloud, "z");
    sensor_msgs::PointCloud2Iterator<float> iter_rgb(cloud, "rgb");
    
    // Pack the RGB color: Green (0, 255, 0) - same as your markers
    const uint8_t r = 0;    // Red
    const uint8_t g = 255;  // Green
    const uint8_t b = 0;    // Blue
    const uint32_t rgb_value = ((uint32_t)r << 16 | (uint32_t)g << 8 | (uint32_t)b);
    float rgb_float;
    memcpy(&rgb_float, &rgb_value, sizeof(float));
    
    // Fill the point cloud with the pose positions
    for (const auto &pose : poses) {
        *iter_x = pose.position.x;
        *iter_y = pose.position.y;
        *iter_z = pose.position.z;
        *iter_rgb = rgb_float;
        
        ++iter_x;
        ++iter_y;
        ++iter_z;
        ++iter_rgb;
    }
    
    // Publish the point cloud
    point_cloud_pub->publish(cloud);
    std::cout << "Published point cloud with " << poses.size() << " points" << std::endl;
}

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ur3eControl>());
    rclcpp::shutdown();
    return 0;
}
