#include <ros/ros.h>
#include <ros/console.h>
#include <cv_bridge/cv_bridge.h>

#include <nav_msgs/Odometry.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/ximgproc.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

ros::Publisher pointcloud_pub;
nav_msgs::Odometry odom;

void odomCallback(const nav_msgs::Odometry::ConstPtr &odom_msg) {
  odom.header = odom_msg->header;
  odom.child_frame_id = odom_msg->child_frame_id;
  odom.pose = odom_msg->pose;
  odom.twist = odom_msg->twist;
}

void leftCallback(const sensor_msgs::Image::ConstPtr &left_msg) {
  cv_bridge::CvImagePtr left_ptr;
  try {
    left_ptr = cv_bridge::toCvCopy(left_msg, sensor_msgs::image_encodings::BGR8);
  } catch (cv_bridge::Exception& e) {
    ROS_ERROR("cv_bridge exception: %s", e.what());
    return;
  }

  cv::Mat hsv, thresholded, eroded;
  cv::Mat dilated1, dilated2, dilated3;
  cv::Mat thinned1, thinned2;
  cv::Mat output;

  cv::Mat4i lines;

  cv::cvtColor(left_ptr->image, hsv, cv::COLOR_BGR2HSV);
  cv::inRange(hsv, cv::Scalar(70, 30, 180), cv::Scalar(100, 255, 255), thresholded);
  cv::ximgproc::thinning(thresholded, thinned1, cv::ximgproc::THINNING_ZHANGSUEN);
  cv::dilate(thinned1, dilated1, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
  cv::dilate(dilated1, dilated2, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
  cv::dilate(dilated2, dilated3, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5)));
  cv::ximgproc::thinning(dilated3, thinned2, cv::ximgproc::THINNING_ZHANGSUEN);
  
  cv::Ptr<cv::LineSegmentDetector> lsd = cv::createLineSegmentDetector();
  lsd->detect(thinned2, lines);
  lsd->drawSegments(output, lines);

  cv::imshow("Image1", output);
  cv::waitKey(3);
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "line_striping_node");
  ros::NodeHandle nh;

  ros::Subscriber odom_sub = nh.subscribe("/odom", 1, odomCallback);
  ros::Subscriber left_sub = nh.subscribe("/mipi/cam0", 1, leftCallback);
  //ros::Subscriber right_sub = nh.subscribe("/mipi/cam1", 1, rightCallback);

  //output_cloud->header.frame_id = "base_link";
  //pointcloud_pub = nh.advertise<PointCloud<PointXYZ>>("points", 1);

  ros::spin();
  return 0;
}
