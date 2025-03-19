#ifndef UR3E_CONTROL_H
#define UR3E_CONTROL_H

#include "rclcpp/rclcpp.hpp"
#include "std_msgs/msg/bool.hpp"

class ur3eControl : public rclcpp::Node {
    public:
        ur3eControl();
    
    private:
        void imageProcessedCallback(const std_msgs::msg::Bool::SharedPtr msg);
        void startDrawingCallback(const std_msgs::msg::Bool::SharedPtr msg);
    
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr pathPlanningSub_;
        rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr startDrawingSub_;
    };

#endif // UR3E_CONTROL_H