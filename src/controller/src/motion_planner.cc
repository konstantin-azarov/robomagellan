#include <cmath>

#include <Eigen/Geometry>

#include "ros/ros.h"
#include "tf/transform_broadcaster.h"

#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/Twist.h"
#include "nav_msgs/Path.h"
#include "std_msgs/Empty.h" 

namespace e = Eigen;

const double kHeartbeatTimeout = 0.10;
const double kLookAheadSecs = 5;
const double kLookAheadMin = 2;
const double kSpeedMax = 1;
const double kTurnRateMax = 30 * M_PI/180;
const double kAccelerationMax = 0.5;
const double kFinalDistanceThreshold = 0.1;

class MotionPlanner {
  public:
    MotionPlanner(ros::NodeHandle& node) 
      : heartbeat_sub_(node.subscribe(
            "heartbeat", 10, &MotionPlanner::receiveHeartbeat, this)),
        path_sub_(node.subscribe(
            "path", 10, &MotionPlanner::receivePath, this)),
        odometry_sub_(node.subscribe(
            "odometry", 10, &MotionPlanner::receiveOdometry, this)),
        cmd_pub_(
            node.advertise<geometry_msgs::Twist>("control", 10)),
        robot_pose_pub_(
            node.advertise<geometry_msgs::PoseStamped>("estimated_pose", 10)),
        target_point_pub_(
            node.advertise<geometry_msgs::PoseStamped>("target_point", 10)),
        pos_(0, 0, 0),
        heartbeat_time_(0) {
      map_transform_.setOrigin(tf::Vector3(0, 0, 0));
      map_transform_.setRotation(tf::Quaternion(1, 0, 0, 0));

      rot_ = e::AngleAxisd(-M_PI/2, e::Vector3d(1, 0, 0));
      camera_to_robot_ = 
        e::AngleAxisd(-M_PI/2, e::Vector3d(0, 1, 0)) * 
        e::AngleAxisd(M_PI/2, e::Vector3d(1, 0, 0)); 
    }

    void receivePath(const nav_msgs::Path::ConstPtr& path_msg) {
      path_.clear();
      for (const auto& p : path_msg->poses) {
        path_.push_back(e::Vector2d(p.pose.position.x, p.pose.position.y)); 
      }
      current_segment_ = 0;
    }
    
    void receiveOdometry(const geometry_msgs::PoseStamped::ConstPtr& pose_msg) {
      auto time = ros::Time::now();
      double dt = (pose_msg->header.stamp - last_update_).toSec();
      last_update_ = pose_msg->header.stamp;

      const auto& pose = pose_msg->pose;
      e::Vector3d dp(pose.position.x, pose.position.y, pose.position.z);

      if (pose_msg->header.frame_id != "invalid") {
        e::Quaterniond dr(
            pose.orientation.w,
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z);

        pos_ += rot_ * dp;
        rot_ = rot_ * dr;
        rot_.normalize();
      } else {
        if (dt > 0) {
          // xcxc
        }
      }

      double v = dt > 0 ?  dp.norm() / dt : 0;

      geometry_msgs::Twist cmd_msg;
      double cmd_vel = 0, cmd_turn = 0;
      e::Vector2d target_point;

      if (path_.size() > 0 && current_segment_ <= path_.size() - 1 && dt > 0) {
         e::Vector2d target_point, target_dir;

        if (current_segment_ < path_.size() - 1) {
          double t;
          pointSegmentDistance(
              e::Vector2d(pos_.x(), pos_.y()),
              path_[current_segment_],
              path_[current_segment_ + 1],
              t);

          double lookahead = std::max(kLookAheadMin, kLookAheadSecs * v);
          target_point = lookaheadPoint(
              current_segment_, t, lookahead, target_dir);
        } else {
          target_point = path_.back();
          target_dir = (path_.back() - path_[path_.size() - 2]).normalized();
        }

        publishPose_(
            target_point_pub_, time, 
            e::Vector3d(target_point.x(), target_point.y(), 0), 
            e::Quaterniond(e::AngleAxisd(
              atan2(target_dir.y(), target_dir.x()), e::Vector3d(0, 0, 1))));

        navigateToPoint(
            target_point, current_segment_ == path_.size() - 1, 
            cmd_vel, cmd_turn);

        if (cmd_vel > v) {
          cmd_vel = std::min(v + kAccelerationMax*dt, cmd_vel);
        } else {
          //cmd_vel = std::max(v - kAccelerationMax*dt, cmd_vel);
        }
      }

      if ((ros::Time::now() - heartbeat_time_).toSec() < kHeartbeatTimeout) {
        cmd_msg.linear.z = cmd_vel;
        cmd_msg.angular.y = cmd_turn;
      } else {
        cmd_msg.linear.z = 0;
        cmd_msg.angular.y = 0;
      }

      cmd_pub_.publish(cmd_msg);

      publishPose_(robot_pose_pub_, time, pos_, rot_ * camera_to_robot_);

      tf_broadcaster_.sendTransform(
          tf::StampedTransform(map_transform_, ros::Time::now(), "world", "map"));
    }

    void publishPose_(
        ros::Publisher& pub,
        ros::Time time,
        const e::Vector3d& pos,
        const e::Quaterniond& rot) {
      geometry_msgs::PoseStamped msg;
      msg.header.stamp = time;
      msg.header.seq = 0;
      msg.header.frame_id = "map";
      msg.pose.position.x = pos.x();
      msg.pose.position.y = pos.y();
      msg.pose.position.z = pos.z();
      e::Quaterniond robot_rot = rot_ * camera_to_robot_;
      msg.pose.orientation.w = rot.w();
      msg.pose.orientation.x = rot.x();
      msg.pose.orientation.y = rot.y();
      msg.pose.orientation.z = rot.z();

      pub.publish(msg);
    }

    void receiveHeartbeat(const std_msgs::Empty::ConstPtr&) {
      heartbeat_time_ = ros::Time::now();
    }

    void navigateToPoint(
        const e::Vector2d& pt, 
        bool final_point,
        double& v, 
        double& w) {
      e::Vector2d p0(pos_.x(), pos_.y());
      e::Vector3d d_3d = rot_ * e::Vector3d(0, 0, 1);
      e::Vector2d d(d_3d.x(), d_3d.y());
      e::Vector2d cr(d.y(), -d.x());
      
      e::Vector2d p12 = pt - p0;
      double r_inv = (2 * cr.dot(p12)) / (cr.norm() * p12.dot(p12));

      v = std::min(kTurnRateMax / std::max(0.00001, fabs(r_inv)), kSpeedMax);


      if (final_point) {
        double dist = p12.norm();
      
        if (dist > kFinalDistanceThreshold) {
          v = std::min(v, sqrt(2 * dist * kAccelerationMax));
        } else {
          v = 0;
        }
      }

      w = v * r_inv;
    }

    double pointSegmentDistance(
        const e::Vector2d& pt,
        const e::Vector2d& r1,
        const e::Vector2d& r2,
        double& t) {
      double l = (r2 - r1).norm();
      t = (pt - r1).dot(r2 - r1) / l; 
      if (t <= 0) {
        t = 0;
        return (r1 - pt).norm();
      } else if (t >= l) {
        t = l;
        return (r2 - pt).norm();
      } else {
        return (r1 + (r2 - r1)*t/l - pt).norm();
      }
    }

    e::Vector2d lookaheadPoint(
        int& i,
        double t,
        double lookahead,
        e::Vector2d& dir) {

      t += lookahead;

      e::Vector2d res;

      while (t > 0 && i < path_.size() - 1) {
        dir = (path_[i+1] - path_[i]);
        double l = dir.norm();
        if (l >= t) {
          res = path_[i] + dir*t/l;
          dir.normalize();
          t = 0;
        } else {
          t -= l;
          i++;
        }
      } 

      if (t > 0) {
        res = path_.back();
      }

      return res;
    }

  private:
    ros::Subscriber path_sub_, odometry_sub_, heartbeat_sub_;   
    ros::Publisher cmd_pub_, robot_pose_pub_, target_point_pub_;
    std::vector<e::Vector2d> path_;
  
    // Current position and orientation
    e::Vector3d pos_;
    e::Quaterniond rot_, camera_to_robot_;

    int current_segment_;

    ros::Time heartbeat_time_, last_update_;

    tf::TransformBroadcaster tf_broadcaster_;
    tf::Transform map_transform_;
};

int main(int argc, char** argv) {
  ros::init(argc, argv, "motion_planner");

  ros::NodeHandle node;

  MotionPlanner planner(node);

  ros::spin();
}
