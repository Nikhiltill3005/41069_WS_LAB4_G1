#include "rclcpp/rclcpp.hpp"

class MyCppNode : public rclcpp::Node {
public:
    MyCppNode() : Node("my_cpp_node") {
        RCLCPP_INFO(this->get_logger(), "C++ Node Started");
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<MyCppNode>());
    rclcpp::shutdown();
    return 0;
}
