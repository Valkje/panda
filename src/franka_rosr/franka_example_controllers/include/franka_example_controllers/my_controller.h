// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <typeinfo>

#include <controller_interface/multi_interface_controller.h>
#include <dynamic_reconfigure/server.h>
#include <geometry_msgs/PoseStamped.h>
#include <hardware_interface/joint_command_interface.h>
#include <hardware_interface/robot_hw.h>
#include <ros/node_handle.h>
#include <ros/time.h>
#include <Eigen/Dense>
#include <Eigen/Core>

#include <franka_example_controllers/my_paramConfig.h>
#include <franka_hw/franka_model_interface.h>
#include <franka_hw/franka_state_interface.h>
#include <franka_hw/franka_cartesian_command_interface.h>

//Force:
#include <franka_example_controllers/desired_mass_paramConfig.h>

namespace franka_example_controllers {

class MyController : public controller_interface::MultiInterfaceController<
                                                franka_hw::FrankaModelInterface,
                                                hardware_interface::EffortJointInterface,
                                                franka_hw::FrankaStateInterface> {
 public:
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& node_handle) override;
  void starting(const ros::Time&) override;
  void update(const ros::Time&, const ros::Duration& period) override;

 private:
  int counter{0};
  int counter2 = 5000;
  std::string state = "Free";
  double lambda1 = 0.1;
  double lambda2 = 5;
  double height_threshold = 0.15;
  double x_max = 0.94;
  double x_min = 0.45;
  double y_max = 0.3;
  double y_min = -0.3;
  double free_vel_max_ = 0.25;
  double free_acceleration_time_ = 0.0005;
  double free_vel_divider_ = 5.0;
  bool change_to_docked_ = false;

  int figure_type_ = 0;
  int figure_type_target_ = 0;
  enum Figure {Circle, Lissajous, Spiral, Square};

  // Saturation
  Eigen::Matrix<double, 7, 1> saturateTorqueRate(
      const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
      const Eigen::Matrix<double, 7, 1>& tau_J_d);  // NOLINT (readability-identifier-naming)

  std::unique_ptr<franka_hw::FrankaStateHandle> state_handle_;
  std::unique_ptr<franka_hw::FrankaModelHandle> model_handle_;
  std::vector<hardware_interface::JointHandle> joint_handles_;

  double filter_params_{0.005};
  double nullspace_stiffness_{0.0};
  double nullspace_stiffness_target_{0.0};
  const double delta_tau_max_{1.0};
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_;
  Eigen::Matrix<double, 6, 6> cartesian_stiffness_target_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_;
  Eigen::Matrix<double, 6, 6> cartesian_damping_target_;
  Eigen::Matrix<double, 7, 1> q_d_nullspace_;
  Eigen::Vector3d position_d_;
  Eigen::Quaterniond orientation_d_;
  Eigen::Vector3d position_d_target_;
  Eigen::Quaterniond orientation_d_target_;
  Eigen::Vector3d start_position_;

  //=================================== //
  // Force control:
  double desired_mass_{0.25};
  double target_mass_{0.25};
  double k_p_{0.45};
  double k_i_{0.42};
  double k_d_{0.3};
  double target_k_p_{0.45};
  double target_k_i_{0.42};
  double target_k_d_{0.3};
  double filter_gain_{0.001};
  Eigen::Matrix<double, 7, 1> tau_ext_initial_;
  Eigen::Matrix<double, 7, 1> tau_error_;
  static constexpr double kDeltaTauMax{1.0};
  double setpoint_ = -5.0;
  double setpoint_target_ = setpoint_;
  double setpoint_switch_ = 1.0;

  // Cartesian pose:
  std::array<double, 16> initial_pose_;

  double radius_{0.000001};
  double radius_target_ = radius_;
  double acceleration_time_{2.0};
  double vel_max_{0.01};
  double vel_max_target_ = vel_max_;
  double angle_{0.0};
  double vel_current_{0.0};
  double figure_time_{0.0};
  double delta_x_{0.0};
  double delta_y_{0.0};

  Eigen::Vector3d initial_docked_pos_;
  double force_error_{0.0};
  double force_error_integral_{0.0};
  double force_error_derivative_{0.0};


  // Dynamic reconfigure
  std::unique_ptr<dynamic_reconfigure::Server<franka_example_controllers::my_paramConfig>>
      dynamic_server_my_param_;
  ros::NodeHandle dynamic_reconfigure_my_param_node_;
  void myParamCallback(franka_example_controllers::my_paramConfig& config,
                               uint32_t level);
  void initFigure();
};

}  // namespace franka_example_controllers
