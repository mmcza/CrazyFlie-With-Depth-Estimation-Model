#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>

class ImageSaver : public rclcpp::Node
{
    public:
        ImageSaver() : Node("image_saver")
        {
            subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
                "/crazyflie/depth_camera", 10,
                std::bind(&ImageSaver::image_callback, this, std::placeholders::_1));
            
            count_ = 0;
        }

    private:
        void image_callback(const sensor_msgs::msg::Image::SharedPtr msg)
        {
            RCLCPP_INFO(this->get_logger(), "Image encoding: %s", msg->encoding.c_str());

            if (count_ < 1){
                cv_bridge::CvImagePtr cv_ptr;

                try
                {
                    // Check if the image encoding is 32FC1 (single-channel depth image)
                    if (msg->encoding == "32FC1") {
                        cv_ptr = cv_bridge::toCvCopy(msg, msg->encoding);
                    } else {
                        // If the image is in BGR8 or another format, convert accordingly
                        cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
                    }
                }
                catch (cv_bridge::Exception &e)
                {
                    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
                    return;
                }

                // Save the image based on its encoding
                std::string filename = "saved_image.png";
                cv::Mat display_image;

                if (msg->encoding == "32FC1") {
                    // Normalize depth data for saving and displaying as an 8-bit image
                    cv::normalize(cv_ptr->image, display_image, 0, 255, cv::NORM_MINMAX);
                    display_image.convertTo(display_image, CV_8U);
                    cv::imwrite(filename, display_image);
                } else {
                    display_image = cv_ptr->image;
                    cv::imwrite(filename, display_image);
                }
                
                RCLCPP_INFO(this->get_logger(), "Saved image to %s", filename.c_str());

                // Display the image in a window
                cv::imshow("Saved Image", display_image);
                cv::waitKey(0); // Wait for a key press to close the window

                count_++;
            }
        }

        rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subscription_;
        int count_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ImageSaver>());
    rclcpp::shutdown();
    return 0;
}
