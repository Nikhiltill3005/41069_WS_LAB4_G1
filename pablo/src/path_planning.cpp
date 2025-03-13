#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

class pathPlanning : public rclcpp::Node {
public:
    pathPlanning() : Node("path_planning") {
        RCLCPP_INFO(this->get_logger(), "C++ Node Started");

        // Create a subscriber
        subscription_ = this->create_subscription<std_msgs::msg::Bool>(
            "image_processed", 10, std::bind(&pathPlanning::imageProcessedCallback, this, std::placeholders::_1));
    }

private:
    void imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        try {
            if (msg->data == true) {
                RCLCPP_INFO(this->get_logger(), "Image processing complete. Starting path planning...");
                // Add your path planning logic here
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in imageProcessedCallback: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception in imageProcessedCallback");
        }
    }

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr subscription_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<pathPlanning>());
    rclcpp::shutdown();
    return 0;
}