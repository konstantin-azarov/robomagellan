#include "ros/ros.h"
#include "trex_dmc01/Status.h"
#include "trex_dmc01/SetMotors.h"

#include <boost/format.hpp>
#include <stdexcept>

#include <fcntl.h>
#include <termios.h>

#include <string>
#include <iostream>

using boost::format;

using namespace std;

struct TrexException : public std::runtime_error {
  using std::runtime_error::runtime_error;
};

class TrexDriver {
  public:
    TrexDriver(const std::string& port_name, int timeout_ms) {

      port_fd_ = open(port_name.c_str(), O_RDWR | O_NOCTTY | O_SYNC);
      if (port_fd_ < 0) {
        throw TrexException("Failed to open serial port");
      }

      termios tty;
      if (tcgetattr(port_fd_, &tty) != 0) {
        throw TrexException("tcgetattr failed");
      }

      cfsetospeed (&tty, B19200);
      cfsetispeed (&tty, B19200);

      tty.c_cflag = (tty.c_cflag & ~CSIZE) | CS8;     // 8-bit chars
      tty.c_iflag &= ~IGNBRK;         // disable break processing
      tty.c_lflag = 0;                // no signaling chars, no echo,
                                      // no canonical processing
      tty.c_oflag = 0;                // no remapping, no delays
      tty.c_cc[VMIN]  = 0;            // read doesn't block
      tty.c_cc[VTIME] = std::min(1, timeout_ms / 100);
      tty.c_iflag &= ~(IXON | IXOFF | IXANY); // shut off xon/xoff ctrl
      tty.c_cflag |= (CLOCAL | CREAD); // ignore modem controls,
                                       // enable reading
      tty.c_cflag &= ~(PARENB | PARODD);      // no parity
      tty.c_cflag |= 0;
      tty.c_cflag &= ~CSTOPB;   // one stop bit
      tty.c_cflag &= ~CRTSCTS;  // no hardware flow control

      if (tcsetattr(port_fd_, TCSANOW, &tty) != 0) {
        throw TrexException("tcsetattr failed");
      }

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

        auto channels = command<10>(0x86, 0x1F);

        for (int i = 0; i < 5; ++i) {
          res.channels[i] = channels[i*2] + (channels[i*2 + 1] << 8);
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
      boost::array<uint8_t, N> response;

      if (write(port_fd_, cmd.data(), cmd.size()) != cmd.size()) {
        throw TrexException("Write failed");
      }

      int nread = 0, cnt;
      
      while (nread < N &&
             (cnt = read(port_fd_, response.data() + nread, N - nread)) > 0) {
        nread += cnt;
      }

      if (nread != response.size()) {
        throw TrexException(str(format("Read failed: %1%") % nread));
      }

      return response;
    }


  private:
    int port_fd_;
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
      node.param("timeout_ms", 100));

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
