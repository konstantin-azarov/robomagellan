#include "ros/ros.h"
#include "trex_dmc01/Status.h"
#include "trex_dmc01/SetMotors.h"

#include <boost/asio.hpp>
#include <boost/format.hpp>
#include <stdexcept>

#include <string>
#include <iostream>

namespace asio = boost::asio;
using boost::format;

using namespace std;

struct TrexException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TrexDriver {
  public:
    TrexDriver(const std::string& port_name, int timeout_ms) 
        : port_(io_), timeout_(boost::posix_time::milliseconds(timeout_ms)) {
      port_.open(port_name);
      port_.set_option(asio::serial_port_base::baud_rate(19200));
      /* port_.set_option( */
      /*     asio::serial_port_base::stop_bits( */
      /*       asio::serial_port_base::stop_bits::one)); */
      /* port_.set_option( */
      /*     asio::serial_port_base::parity(asio::serial_port_base::parity::none)); */
      /* port_.set_option( */
      /*     asio::serial_port_base::flow_control( */
      /*       asio::serial_port_base::flow_control::none)); */
      /* port_.set_option( */
      /*     asio::serial_port_base::character_size(8)); */

      ROS_INFO("Is open: %d", port_.is_open());

      std::string signature(reinterpret_cast<char*>(command<7>(0x81).data()));
      
      if (signature.substr(0, 4) != "TReX") {
        throw TrexException(
            str(format("Unexpected signature: %1%") % signature));
      }

      ROS_INFO_STREAM("Found TReX controller: " << signature);
    }

    trex_dmc01::Status status() {
      trex_dmc01::Status res;

      auto status = command<1>(0x84)[0];
      if ((status & (1 << 6)) || (status & (1 << 4)) || (status & 1)) {
        ROS_ERROR_STREAM("TReX failure: " << status);
        res.state = trex_dmc01::Status::STATE_ERROR;
      } else {
        auto control_mode = command<1>(0x83);
        switch (control_mode[0]) {
          case 0:
            res.state = trex_dmc01::Status::STATE_RC;
            break;
          case 1:
            res.state = trex_dmc01::Status::STATE_SERIAL;
            break;
          default:
            throw TrexException("Unexpected response to the 0x83 command");
        }

        auto channels = command<5>(0x87, 0x1F);

        for (int i = 0; i < 5; ++i) {
          auto v = channels[i];
          res.channels[i] = (v & 0x3F) * ((v >> 7) ? -1 : 1);
        }
      }

      return res;
    }

    void setMotors(int8_t left, int8_t right) {
      int dir1 = right > 0 ? 2 : 1;
      int v1 = std::abs(right);
      int dir2 = left > 0 ? 2 : 1;
      int v2 = std::abs(left);

      command<0>(0xE0 | (dir2*4) | dir1, v1, v2);
    }

  private:

    template <int N>
    boost::array<uint8_t, N> command(uint8_t cmd) {
      return command<N, 1>(boost::array<uint8_t, 1>{ cmd });
    }

    template <int N>
    boost::array<uint8_t, N> command(uint8_t cmd, uint8_t arg) {
      return command<N, 2>(boost::array<uint8_t, 2>{ cmd, arg });
    }

    template <int N>
    boost::array<uint8_t, N> command(uint8_t cmd, uint8_t arg1, uint8_t arg2) {
      return command<N, 3>(boost::array<uint8_t, 3>{ cmd, arg1, arg2 });
    }

    template <int N, int M>
    boost::array<uint8_t, N> command(boost::array<uint8_t, M> cmd) {
 //     boost::array<uint8_t, M> echo;
      boost::array<uint8_t, N> response;

      boost::system::error_code error_code;

      io_.reset();

      /* asio::deadline_timer timer(io_); */

      boost::asio::write(port_, boost::asio::buffer(cmd));
      timespec ts = { 0, 1000*1000*20 } ;
      nanosleep(&ts, nullptr);
      /* boost::asio::read(port_, boost::asio::buffer(response)); */
      boost::asio::async_read(
          port_, 
          boost::asio::buffer(response),
          [&](
            const boost::system::error_code& error, 
            std::size_t n_bytes) {
              error_code.assign(error.value(), error.category());
              /* timer.cancel(); */
            });

      /* timer.expires_from_now(timeout_); */
      /* timer.async_wait( */
      /*     [&](const boost::system::error_code& error) { */
      /*       port_.cancel(); */
      /*     }); */

      io_.run();

      if (error_code.value() != boost::system::errc::success) {
        throw TrexException(str(format(
                "Read from controller failed: %1%") % error_code.message()));
      }

      /* if (echo != cmd) { */
      /*   throw TrexException("Invalid echo received from the controller"); */
      /* } */

      return response;
    }


  private:
    asio::io_service io_;
    asio::serial_port port_; 
    boost::posix_time::time_duration timeout_;
};

template <class T>
T getParam(ros::NodeHandle& node, const std::string& param_name) {
  T value;
  if (!node.getParam(param_name, value)) {
    throw TrexException(str(format("Parameter not found: %1%") % param_name));
  }

  return value;
}

void run() {
  ros::NodeHandle node("trex_dmc01");

  ROS_INFO_STREAM("Started node " << node.getNamespace());

  TrexDriver driver(
      getParam<std::string>(node, "serial_port"),
      node.param("timeout_ms", 300));

  auto pub = node.advertise<trex_dmc01::Status>("status", 10);

  boost::function<void(const trex_dmc01::SetMotorsConstPtr&)> handle_command = 
    [&](const trex_dmc01::SetMotors::ConstPtr& command) {
      driver.setMotors(command->left, command->right);
    };

  auto sub = node.subscribe("motors_command", 10, handle_command);

  auto timer = node.createTimer(
      ros::Duration(node.param("update_period_ms", 100)/1000.0),
      [&](const ros::TimerEvent& event) {
        pub.publish(driver.status());
      });



  ros::spin();
}

int main(int argc, char** argv) {
  ros::init(argc, argv, "trex_dmc01");
  
  try {
    run();
  } catch (const std::exception& e) {
    std::cerr << "[ ERROR] [" << ros::Time::now() << "]: " << e.what() 
      << std::endl;
    return 1;
  } 
  
  return 0;
}
