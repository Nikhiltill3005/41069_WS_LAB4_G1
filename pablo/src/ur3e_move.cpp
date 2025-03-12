#include "rclcpp/rclcpp.hpp"

class ur3eMove : public rclcpp::Node {
public:
    ur3eMove() : Node("ur3e_move") {
        RCLCPP_INFO(this->get_logger(), "C++ Node Started");
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ur3eMove>());
    rclcpp::shutdown();
    return 0;
}
