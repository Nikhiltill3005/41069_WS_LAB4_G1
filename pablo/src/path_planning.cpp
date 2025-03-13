#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"
#include <thread>

class pathPlanning : public rclcpp::Node {
public:
    pathPlanning() : Node("path_planning") {
        RCLCPP_INFO(this->get_logger(), "Path Planning Node Started");

        // Create a subscriber
        imageProcessorSub_ = this->create_subscription<std_msgs::msg::Bool>(
            "image_processed", 10, std::bind(&pathPlanning::imageProcessedCallback, this, std::placeholders::_1));

        // Create a publisher
        pathPlanningPub_ = this->create_publisher<std_msgs::msg::Bool>("path_planned", 10);
    }

private:
    void imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        try {
            if (msg->data == true) {
                RCLCPP_INFO(this->get_logger(), "Image processing complete. Starting path planning...");
                std::this_thread::sleep_for(std::chrono::milliseconds(5000));
                RCLCPP_INFO(this->get_logger(), "Path planning complete. Publishing path_planned message...");
                auto pathPlannedMsg = std_msgs::msg::Bool();
                pathPlannedMsg.data = true;
                pathPlanningPub_->publish(pathPlannedMsg);

                RCLCPP_INFO(this->get_logger(), "Path planned message published");
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in imageProcessedCallback: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception in imageProcessedCallback");
        }
    }

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr imageProcessorSub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr pathPlanningPub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<pathPlanning>());
    rclcpp::shutdown();
    return 0;
}