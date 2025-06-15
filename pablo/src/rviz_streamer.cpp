#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <gst/gst.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <thread>
#include <chrono>

class RVizStreamerNode : public rclcpp::Node {
private:
    GstElement *pipeline;
    GstElement *source, *convert, *encoder, *payloader, *sink;
    Window rviz_window;
    bool streaming_active = false;
    std::thread pipeline_thread;
    
    // ROS2 publisher to announce streaming status
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_publisher;
    rclcpp::TimerBase::SharedPtr status_timer;

public:
    RVizStreamerNode() : Node("rviz_streamer") {
        // Initialize GStreamer
        gst_init(nullptr, nullptr);
        
        // Create ROS2 publisher
        status_publisher = this->create_publisher<std_msgs::msg::String>("rviz_stream_status", 10);
        
        // Create timer to publish status
        status_timer = this->create_wall_timer(
            std::chrono::seconds(5),
            std::bind(&RVizStreamerNode::publishStatus, this)
        );
        
        // Initialize streaming
        if (initializeStreaming()) {
            startStreaming();
        } else {
            RCLCPP_ERROR(this->get_logger(), "Failed to initialize RViz streaming");
        }
    }
    
    ~RVizStreamerNode() {
        stopStreaming();
    }
    
private:
    bool initializeStreaming() {
        // Find RViz window
        if (!findRVizWindow()) {
            RCLCPP_ERROR(this->get_logger(), "Could not find RViz window");
            return false;
        }
        
        RCLCPP_INFO(this->get_logger(), "Found RViz window: %lu", rviz_window);
        
        // Create pipeline elements
        pipeline = gst_pipeline_new("rviz-streamer");
        source = gst_element_factory_make("ximagesrc", "source");
        convert = gst_element_factory_make("videoconvert", "convert");
        encoder = gst_element_factory_make("x264enc", "encoder");
        payloader = gst_element_factory_make("h264parse", "parser");
        sink = gst_element_factory_make("hlssink", "sink");
        
        if (!pipeline || !source || !convert || !encoder || !payloader || !sink) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create GStreamer elements");
            return false;
        }
        
        // Configure elements
        g_object_set(G_OBJECT(source), "xid", rviz_window, nullptr);
        g_object_set(G_OBJECT(encoder), 
                    "tune", 0x00000004, // zerolatency
                    "bitrate", 2000,
                    "speed-preset", 1, // ultrafast
                    nullptr);
        g_object_set(G_OBJECT(sink), 
                    "location", "/tmp/rviz_stream%05d.ts",
                    "playlist-location", "/tmp/rviz_stream.m3u8",
                    "max-files", 5,
                    "target-duration", 2,
                    nullptr);
        
        // Add elements to pipeline
        gst_bin_add_many(GST_BIN(pipeline), source, convert, encoder, payloader, sink, nullptr);
        
        // Link elements
        if (!gst_element_link_many(source, convert, encoder, payloader, sink, nullptr)) {
            RCLCPP_ERROR(this->get_logger(), "Failed to link GStreamer elements");
            return false;
        }
        
        return true;
    }
    
    void startStreaming() {
        GstStateChangeReturn ret = gst_element_set_state(pipeline, GST_STATE_PLAYING);
        if (ret == GST_STATE_CHANGE_FAILURE) {
            RCLCPP_ERROR(this->get_logger(), "Failed to start GStreamer pipeline");
            return;
        }
        
        streaming_active = true;
        RCLCPP_INFO(this->get_logger(), "Started RViz HLS streaming to /tmp/rviz_stream.m3u8");
        
        // Start pipeline monitoring thread
        pipeline_thread = std::thread(&RVizStreamerNode::monitorPipeline, this);
    }
    
    void stopStreaming() {
        streaming_active = false;
        
        if (pipeline) {
            gst_element_set_state(pipeline, GST_STATE_NULL);
            gst_object_unref(pipeline);
            pipeline = nullptr;
        }
        
        if (pipeline_thread.joinable()) {
            pipeline_thread.join();
        }
        
        RCLCPP_INFO(this->get_logger(), "Stopped RViz streaming");
    }
    
    void monitorPipeline() {
        GstBus *bus = gst_element_get_bus(pipeline);
        
        while (streaming_active) {
            GstMessage *msg = gst_bus_timed_pop_filtered(bus, 100 * GST_MSECOND,
                static_cast<GstMessageType>(GST_MESSAGE_STATE_CHANGED | GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
            
            if (msg != nullptr) {
                switch (GST_MESSAGE_TYPE(msg)) {
                    case GST_MESSAGE_ERROR: {
                        GError *err;
                        gchar *debug_info;
                        gst_message_parse_error(msg, &err, &debug_info);
                        RCLCPP_ERROR(this->get_logger(), "GStreamer error: %s", err->message);
                        g_clear_error(&err);
                        g_free(debug_info);
                        break;
                    }
                    case GST_MESSAGE_EOS:
                        RCLCPP_INFO(this->get_logger(), "End-Of-Stream reached");
                        break;
                    default:
                        break;
                }
                gst_message_unref(msg);
            }
        }
        
        gst_object_unref(bus);
    }
    
    bool findRVizWindow() {
        Display* display = XOpenDisplay(nullptr);
        if (!display) {
            RCLCPP_ERROR(this->get_logger(), "Cannot open X display");
            return false;
        }
        
        Window root = DefaultRootWindow(display);
        Window parent, *children;
        unsigned int nchildren;
        
        if (XQueryTree(display, root, &root, &parent, &children, &nchildren)) {
            for (unsigned int i = 0; i < nchildren; i++) {
                char* window_name;
                if (XFetchName(display, children[i], &window_name) && window_name) {
                    std::string name_str(window_name);
                    if (name_str.find("RViz") != std::string::npos || 
                        name_str.find("rviz") != std::string::npos) {
                        rviz_window = children[i];
                        XFree(window_name);
                        XFree(children);
                        XCloseDisplay(display);
                        return true;
                    }
                    XFree(window_name);
                }
            }
            XFree(children);
        }
        
        XCloseDisplay(display);
        return false;
    }
    
    void publishStatus() {
        auto message = std_msgs::msg::String();
        if (streaming_active) {
            message.data = "RViz streaming active - http://localhost:8080/rviz_stream.m3u8";
        } else {
            message.data = "RViz streaming inactive";
        }
        status_publisher->publish(message);
    }
};

int main(int argc, char * argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<RVizStreamerNode>());
    rclcpp::shutdown();
    return 0;
}