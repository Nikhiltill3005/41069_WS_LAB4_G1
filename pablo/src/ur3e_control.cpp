#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

class ur3eControl : public rclcpp::Node {
public:
    ur3eControl() : Node("ur3e_control") {
        RCLCPP_INFO(this->get_logger(), "UR3e Move Node Started");

        // Create a subscriber
        pathPlanningSub_ = this->create_subscription<std_msgs::msg::Bool>(
            "path_planned", 10, std::bind(&ur3eControl::imageProcessedCallback, this, std::placeholders::_1));

        // Create a subscriber
        startDrawingSub_ = this->create_subscription<std_msgs::msg::Bool>(
            "starter", 10, std::bind(&ur3eControl::startDrawingCallback, this, std::placeholders::_1));
    }

private:
    void imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        try {
            if (msg->data == true) {
                RCLCPP_INFO(this->get_logger(), "Image processing complete. Waiting for start...");
                // Add your path planning logic here
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in imageProcessedCallback: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception in imageProcessedCallback");
        }
    }

    void startDrawingCallback(const std_msgs::msg::Bool::SharedPtr msg) {
        try {
            if (msg->data == true) {
                RCLCPP_INFO(this->get_logger(), "Starting drawing...");
                // Add your drawing logic here
            }
        } catch (const std::exception &e) {
            RCLCPP_ERROR(this->get_logger(), "Exception in startDrawingCallback: %s", e.what());
        } catch (...) {
            RCLCPP_ERROR(this->get_logger(), "Unknown exception in startDrawingCallback");
        }
    }

    
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pathPlanningSub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr startDrawingSub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ur3eControl>());
    rclcpp::shutdown();
    return 0;
}
