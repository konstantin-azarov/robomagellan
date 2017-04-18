#include <Eigen/Geometry>

#include "ros/ros.h"
#include "geometry_msgs/PoseWithCovarianceStamped.h"
#include "geometry_msgs/Twist.h"

#include "trex_dmc01/SetMotors.h"

#include "controller/Parameters.h"

#include <iostream>

namespace e = Eigen;

geometry_msgs::Vector3 vec3(double x, double y, double z) {
  geometry_msgs::Vector3 res;
  res.x = x;
  res.y = y;
  res.z = z;

  return res;
}

class Controller {
  public:
    Controller(ros::NodeHandle& node) 
        : smoothing_(node.param("smoothing", 0.6)),
          s_(
              node.param("k_l", 2.1),
              node.param("k_r", 2.1),
              node.param("b", 3.1)),
          p_(e::Vector3f(
                node.param("k_v", 0.5),
                node.param("k_v", 0.5),
                node.param("b_v", 0.5)).asDiagonal()),
          odometry_sub_(node.subscribe(
                "/odometry", 50, &Controller::receiveOdometry, this)),
          control_sub_(node.subscribe(
                "/control", 50, &Controller::receiveCommand, this)),
          twist_pub_(node.advertise<geometry_msgs::Twist>("/twist", 10)),
          command_pub_(
              node.advertise<trex_dmc01::SetMotors>("/motors", 10)),
          params_pub_(
              node.advertise<controller::Parameters>("/parameters", 10)) {
    }

    void receiveOdometry(
        const geometry_msgs::PoseWithCovarianceStamped::ConstPtr& pose_msg) {
      const auto& pose = pose_msg->pose.pose;

      e::Vector3d pos(pose.position.x, pose.position.y, pose.position.z);
      auto t = pose_msg->header.stamp;
      double dt = (t - t_).toSec();

      if (t_.isValid()) {
        double speed = pos.norm() / dt;

        double angle = 2*acos(pose.orientation.w);
        double s = sqrt(1 - pose.orientation.w * pose.orientation.w);

        double yaw_r =  
          s > 1E-5 ? (angle * pose.orientation.y / s) / dt : 0;

        updateFilter_(speed, 0.1, yaw_r, 0.02);
        updateControl_();
        sendControl_();

        yaw_rate_ = yaw_rate_ * (1 - smoothing_) + yaw_r * smoothing_;
        speed_ = speed_ * (1 - smoothing_) + speed * smoothing_;

        geometry_msgs::Twist msg;
        msg.linear = vec3(0.0, 0.0, speed_);
        msg.angular = vec3(0, yaw_rate_, 0);
        twist_pub_.publish(msg);

        controller::Parameters p;
        p.k_l = s_.x();
        p.k_r = s_.y();
        p.b = s_.z();
        params_pub_.publish(p);
      }
      
      t_ = t;
    }

    void receiveCommand(const geometry_msgs::Twist::ConstPtr& msg) {
      target_v_ = msg->linear.z;
      target_w_ = msg->angular.y;

      updateControl_();
      sendControl_();

      /* ROS_INFO("K: %f %f %f", s_.x(), s_.y(), s_.z()); */
      /* ROS_INFO("Received control: %f %f => (%f %f)", */ 
      /*     target_v_, target_w_, cmd_left_, cmd_right_); */
    }

  private:  
    void updateControl_() {
      cmd_left_ = clampCmd_(
          (2 * target_v_ + target_w_ * s_.z()) / (2.0 * s_.x()));
      cmd_right_ = clampCmd_(
          (2 * target_v_ - target_w_ * s_.z()) / (2.0 * s_.y()));
    }

    void sendControl_() {
      trex_dmc01::SetMotors cmd;
      cmd.left = (int)(cmd_left_ * 127 + 0.5);
      cmd.right = (int)(cmd_right_ * 127 + 0.5);
      command_pub_.publish(cmd);
    }

    float clampCmd_(float x) {
      if (x > 1) return 1;
      else if (x < -1) return -1;
      else return x;
    }

    void updateFilter_(
        double speed, 
        double speed_var, 
        double yaw_rate, 
        double yaw_var) {
      // Linearized measurement function 
      e::Matrix<float, 2, 3> m;
      m << cmd_left_/2, cmd_right_/2, 0,
           cmd_left_/s_.z(), -cmd_right_/s_.z(), 
           (s_.y() * cmd_right_ - s_.x() * cmd_left_) / (s_.z() * s_.z());

      // Measurement variance
      e::Matrix2f l = e::Vector2f(speed_var, yaw_var).asDiagonal();
      auto k = p_ * m.transpose() * (m * p_ * m.transpose() + l).inverse(); 

      e::Vector2f f(
          (cmd_left_ * s_.x() + cmd_right_ * s_.y())/2 - speed,
          (cmd_left_ * s_.x() - cmd_right_ * s_.y())/s_.z() - yaw_rate);

      s_ = s_ - k * f;
      p_ = (e::Matrix3f::Identity() - k * m) * p_;
    }

  private:
    double smoothing_ = 0;

    // Kl, Kr, B
    e::Vector3f s_;
    e::Matrix3f p_;

    // Current command
    float cmd_left_ = 0, cmd_right_ = 0;

    float target_v_ = 0, target_w_ = 0;

    float yaw_rate_ = 0, speed_ = 0;
    ros::Time t_;

    ros::Subscriber odometry_sub_, control_sub_;
    ros::Publisher twist_pub_, command_pub_, params_pub_;
};

int main(int argc, char** argv) {

  ros::init(argc, argv, "controller");

  ros::NodeHandle node("controller");
  
  Controller controller(node);

  ros::spin();
}
