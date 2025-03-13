#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

class ur3eMove : public rclcpp::Node {
public:
    ur3eMove() : Node("ur3e_move") {
        RCLCPP_INFO(this->get_logger(), "UR3e Move Node Started");

        // Create a subscriber
        pathPlanningSub_ = this->create_subscription<std_msgs::msg::Bool>(
            "image_processed", 10, std::bind(&ur3eMove::imageProcessedCallback, this, std::placeholders::_1));
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

    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pathPlanningSub_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ur3eMove>());
    rclcpp::shutdown();
    return 0;
}
