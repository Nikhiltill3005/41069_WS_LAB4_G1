#include "rclcpp/rclcpp.hpp"

class pathPlanning : public rclcpp::Node {
public:
    pathPlanning() : Node("path_planning") {
        RCLCPP_INFO(this->get_logger(), "C++ Node Started");
    }

private:

};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<pathPlanning>());
    rclcpp::shutdown();
    return 0;
}
