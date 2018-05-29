// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/my_controller.h>

#include <cmath>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include "pseudo_inversion.h"

namespace franka_example_controllers {

bool MyController::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  sub_equilibrium_pose_ = node_handle.subscribe(
      "/equilibrium_pose", 20, &MyController::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("MyController: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "MyController: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  franka_hw::FrankaModelInterface* model_interface =
      robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MyController: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_.reset(
        new franka_hw::FrankaModelHandle(model_interface->getHandle(arm_id + "_model")));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "MyController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  franka_hw::FrankaStateInterface* state_interface =
      robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MyController: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_.reset(
        new franka_hw::FrankaStateHandle(state_interface->getHandle(arm_id + "_robot")));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "MyController: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  hardware_interface::EffortJointInterface* effort_joint_interface =
      robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MyController: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "MyController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle("dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_.reset(
      new dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>(
          dynamic_reconfigure_compliance_param_node_));
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&MyController::complianceParamCallback, this, _1, _2));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  // Cartesian pose
  try {
    model_handle_.reset(
        new franka_hw::FrankaModelHandle(model_interface->getHandle(arm_id + "_model")));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  franka_hw::FrankaPoseCartesianInterface* cartesian_pose_interface =
      robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface == nullptr) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Error getting cartesian pose interface from hardware");
    return false;
  }
  try {
    cartesian_pose_handle_.reset(new franka_hw::FrankaCartesianPoseHandle(
        cartesian_pose_interface->getHandle(arm_id + "_robot")));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "JointImpedanceExampleController: Exception getting cartesian pose handle from interface: "
        << ex.what());
    return false;
  }

  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "JointImpedanceExampleController: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  if (!node_handle.getParam("radius", radius_)) {
    ROS_INFO_STREAM(
        "JointImpedanceExampleController: No parameter radius, defaulting to: " << radius_);
  }
  if (std::fabs(radius_) < 0.005) {
    ROS_INFO_STREAM("JointImpedanceExampleController: Set radius to small, defaulting to: " << 0.1);
    radius_ = 0.1;
  }

  if (!node_handle.getParam("vel_max", vel_max_)) {
    ROS_INFO_STREAM(
        "JointImpedanceExampleController: No parameter vel_max, defaulting to: " << vel_max_);
  }
  if (!node_handle.getParam("acceleration_time", acceleration_time_)) {
    ROS_INFO_STREAM(
        "JointImpedanceExampleController: No parameter acceleration_time, defaulting to: "
        << acceleration_time_);
  }

  return true;
}

void MyController::starting(const ros::Time& /*time*/) {
  // compute initial velocity with jacobian and set x_attractor and q_d_nullspace
  // to initial configuration
  franka::RobotState initial_state = state_handle_->getRobotState();
  // get jacobian
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);
  // convert to eigen
  Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > dq_initial(initial_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > q_initial(initial_state.q.data());
  Eigen::Affine3d initial_transform(Eigen::Matrix4d::Map(initial_state.O_T_EE.data()));

  // set equilibrium point to current state
  // Translation into Cartesian space?
  position_d_ = initial_transform.translation();
  orientation_d_ = Eigen::Quaterniond(initial_transform.linear());
  position_d_target_ = initial_transform.translation();
  orientation_d_target_ = Eigen::Quaterniond(initial_transform.linear());
  
  // ======================================================================= //
  // Force control
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_measured(robot_state.tau_J.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > gravity(gravity_array.data());
  // Bias correction for the current external torque
  tau_ext_initial_ = tau_measured - gravity;
  tau_error_.setZero();

  initial_docked_pos_ << position_d_;

  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  cartesian_pose_handle_->setCommand(initial_pose_);

  start_position_ << (
    0.305501,
    -0.000100327,
    0.483903
  );
}

void MyController::update(const ros::Time& /*time*/,
                                                 const ros::Duration& period) {
  // get state variables
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> coriolis_array = model_handle_->getCoriolis();
  std::array<double, 42> jacobian_array =
      model_handle_->getZeroJacobian(franka::Frame::kEndEffector);

  // convert to Eigen
  Eigen::Map<Eigen::Matrix<double, 7, 1> > coriolis(coriolis_array.data());
  Eigen::Map<Eigen::Matrix<double, 6, 7> > jacobian(jacobian_array.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > q(robot_state.q.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > dq(robot_state.dq.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_J_d(  // NOLINT (readability-identifier-naming)
      robot_state.tau_J_d.data());
  Eigen::Affine3d transform(Eigen::Matrix4d::Map(robot_state.O_T_EE.data()));
  Eigen::Vector3d position(transform.translation());
  Eigen::Quaterniond orientation(transform.linear());
  
  Eigen::Matrix<double, 6, 1> error;


  double force_EE_z = robot_state.K_F_ext_hat_K[2];
  
  if (force_EE_z > lambda2 && change_to_docked_){ // && position.z() < height_threshold
    if (state.compare("Free") == 0) {
      initial_docked_pos_ << position_d_;
      vel_current_ = 0.0;
    }
    state = "Docked";
    counter2 = 5000;
    cartesian_stiffness_target_(2, 2) = z_stiffness_docked;
  }
  
  if (force_EE_z < lambda1){ // || position.z() >= height_threshold
    state = "Free";
    if (state.compare("Docked") == 0) {
      position_d_target_(2) += 0.3;
      vel_current_ = 0.0;
    }
    cartesian_stiffness_target_(2, 2) = z_stiffness;
    force_error_integral_ = 0.0;
    force_error_ = 0.0;
    // angle_ = 0.0;
    vel_current_ = 0.0;
    change_to_docked_ = false;
  }

  if (counter % 500 == 0) {
    std::cout << "\033[2J\033[1;1H";
    std::cout << "State: " << state << "\n";
    std::cout << "delta_x: " << (radius_ * std::cos(2.0 * figure_time_)) << "\n";
    std::cout << "delta_y: " << (radius_ * std::sin(2.0 * figure_time_)) << "\n";
    std::cout << "delta_x (old): " << (radius_ * std::sin(angle_)) << "\n";
    std::cout << "delta_y (old): " << (radius_ * (1 - std::cos(angle_))) << "\n";
  }
  counter = (counter + 1) % 500;

  if (change_to_docked_) {
    double filter = 0.9999999999999999;
    k_p_ = filter * k_p_ + (1 - filter) * 0.29;
    k_i_ = filter * k_i_ + (1 - filter) * 5.0;
  }

  if (state.compare("Free") == 0){
    if (vel_current_ < free_vel_max_) {
      vel_current_ += period.toSec() * std::fabs(free_vel_max_ / free_acceleration_time_);
    }
    vel_current_ = std::fmin(vel_current_, free_vel_max_);

    bool in_x_bounds = (position(0) < x_max && position(0) > x_min);
    bool in_y_bounds = (position(1) < y_max && position(1) > y_min);

    if (!in_x_bounds) {
      position_d_target_(0) += std::copysign(vel_current_ * period.toSec(), x_max - position(0));
    }

    if (!in_y_bounds) {
      position_d_target_(1) += std::copysign(vel_current_ * period.toSec(), y_max - position(1));
    }

    if (in_x_bounds && in_y_bounds) {
      if (force_EE_z < lambda2) {
        position_d_target_(2) -= vel_current_ * period.toSec() / free_vel_divider_;
      } else if (force_EE_z >= (lambda2 + 10)) {
        position_d_target_(2) += vel_current_ * period.toSec() / free_vel_divider_ / 10.0;
      } else {
        counter2 = (counter2 + 1) % 250;
        if (counter2 % 250 == 0) {
          change_to_docked_ = true;
          k_p_ = k_i_ = k_d_ = 0;
        }
      }
    }

    // compute error to desired pose
    // position error
    error.head(3) << position - position_d_;

    // orientation error
    if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
      orientation.coeffs() << -orientation.coeffs();
    }

    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation * orientation_d_.inverse());
    // convert to axis angle
    Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
    // compute "orientation error"
    error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

    // compute control
    // allocate variables
    Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d(7);

    // pseudoinverse for nullspace handling
    // kinematic pseuoinverse
    Eigen::MatrixXd jacobian_transpose_pinv;
    pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

    // Cartesian PD control with damping ratio = 1
    tau_task << jacobian.transpose() *
                    (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));
    // nullspace PD control with damping ratio = 1
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                      jacobian.transpose() * jacobian_transpose_pinv) *
                         (0.0 * (q_d_nullspace_ - q) -
                          (2.0 * sqrt(0.0)) * dq); //nullspace_stiffness_ = 0.0
    // Desired torque
    tau_d << tau_task + coriolis + tau_nullspace;
    // Saturate torque rate to avoid discontinuities
    tau_d << saturateTorqueRate(tau_d, tau_J_d);

    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(tau_d(i));
    }
  }
  
  // Force control
  if (state.compare("Docked") == 0){
    if (force_EE_z > 10.0) {
      angle_ = 0.0;
      state = "Free";
    }

    //Cartesian pose
    if (vel_current_ < vel_max_) {
      vel_current_ += period.toSec() * std::fabs(vel_max_ / acceleration_time_);
    }
    vel_current_ = std::fmin(vel_current_, vel_max_);

    angle_ += period.toSec() * vel_current_ / std::fabs(radius_);
    if (angle_ > 2 * M_PI) {
      angle_ -= 2 * M_PI;
    }

    figure_time_ += period.toSec() * vel_current_;

    double delta_x = radius_ * std::sin(angle_);
    double delta_y = radius_ * (1 - std::cos(angle_));

    position_d_(0) = initial_docked_pos_(0) - delta_x;
    position_d_(1) = initial_docked_pos_(1) + delta_y;
    
    //=======================================
    // compute error to desired pose - copied from impedance control
    // position error
    error.head(3) << position - position_d_;

    // orientation error
    if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
      orientation.coeffs() << -orientation.coeffs();
    }

    // "difference" quaternion
    Eigen::Quaterniond error_quaternion(orientation * orientation_d_.inverse());
    // convert to axis angle
    Eigen::AngleAxisd error_quaternion_angle_axis(error_quaternion);
    // compute "orientation error"
    error.tail(3) << error_quaternion_angle_axis.axis() * error_quaternion_angle_axis.angle();

    // compute control
    // allocate variables
    Eigen::VectorXd tau_task(7), tau_nullspace(7), tau_d_impedance(7), forces(7);

    // pseudoinverse for nullspace handling
    // kinematic pseuoinverse
    Eigen::MatrixXd jacobian_transpose_pinv;
    pseudoInverse(jacobian.transpose(), jacobian_transpose_pinv);

    forces << (-cartesian_stiffness_ * error - cartesian_damping_ * (jacobian * dq));

    double old_error = force_error_;
    force_error_ = setpoint_ + force_EE_z; // force_EE_z is positive
    force_error_integral_ += period.toSec() * force_error_;
    force_error_derivative_ = (old_error - force_error_) / period.toSec(); 

    forces(2) = setpoint_ + k_p_ * force_error_ + k_i_ * force_error_integral_ + k_d_ * force_error_derivative_;

    // Cartesian PD control with damping ratio = 1
    tau_task << jacobian.transpose() *
                    (forces); // <- force
    // nullspace PD control with damping ratio = 1
    tau_nullspace << (Eigen::MatrixXd::Identity(7, 7) -
                      jacobian.transpose() * jacobian_transpose_pinv) *
                         (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                          (2.0 * sqrt(nullspace_stiffness_)) * dq);
    // Desired torque
    tau_d_impedance << tau_task + coriolis + tau_nullspace;
    // Saturate torque rate to avoid discontinuities
    tau_d_impedance << saturateTorqueRate(tau_d_impedance, tau_J_d);
    
    //=======================================
    
    // std::array<double, 7> gravity_array = model_handle_->getGravity();
    // Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_measured(robot_state.tau_J.data());
    // Eigen::Map<Eigen::Matrix<double, 7, 1> > gravity(gravity_array.data());

    // Eigen::VectorXd tau_d(7), desired_force_torque(6), tau_cmd(7), tau_ext(7);
    // desired_force_torque.setZero();
    // desired_force_torque(2) = desired_mass_ * -9.81;
    // tau_ext = tau_measured - gravity - tau_ext_initial_;
    // tau_d << jacobian.transpose() * desired_force_torque;
    // tau_error_ = tau_error_ + period.toSec() * (tau_d - tau_ext);
    // // FF + PI control (PI gains are initially all 0)
    // tau_cmd = tau_d + k_p_ * (tau_d - tau_ext) + k_i_ * tau_error_;
    // tau_cmd << saturateTorqueRate(tau_cmd+tau_nullspace, tau_J_d);

    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(tau_d_impedance(i));
    }

    // Update signals changed online through dynamic reconfigure
    desired_mass_ = filter_gain_ * target_mass_ + (1 - filter_gain_) * desired_mass_;
    k_p_ = filter_gain_ * target_k_p_ + (1 - filter_gain_) * k_p_;
    k_i_ = filter_gain_ * target_k_i_ + (1 - filter_gain_) * k_i_;
    k_d_ = filter_gain_ * target_k_d_ + (1 - filter_gain_) * k_d_;
}
  
  // update parameters changed online either through dynamic reconfigure or through the interactive
    // target by filtering
    cartesian_stiffness_ =
        filter_params_ * cartesian_stiffness_target_ + (1.0 - filter_params_) * cartesian_stiffness_;
    cartesian_damping_ =
        filter_params_ * cartesian_damping_target_ + (1.0 - filter_params_) * cartesian_damping_;
    nullspace_stiffness_ =
        filter_params_ * nullspace_stiffness_target_ + (1.0 - filter_params_) * nullspace_stiffness_;
    position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
    Eigen::AngleAxisd aa_orientation_d(orientation_d_);
    Eigen::AngleAxisd aa_orientation_d_target(orientation_d_target_);
    aa_orientation_d.axis() = filter_params_ * aa_orientation_d_target.axis() +
                              (1.0 - filter_params_) * aa_orientation_d.axis();
    aa_orientation_d.angle() = filter_params_ * aa_orientation_d_target.angle() +
                               (1.0 - filter_params_) * aa_orientation_d.angle();
    orientation_d_ = Eigen::Quaterniond(aa_orientation_d);


    radius_ = (1 - 0.0001) * radius_ + 0.0001 * radius_target_;
    vel_max_ = filter_params_ * vel_max_target_ + (1.0 - filter_params_) * vel_max_;
    setpoint_ = filter_params_ * setpoint_target_ + (1.0 - filter_params_) * setpoint_;
}

Eigen::Matrix<double, 7, 1> MyController::saturateTorqueRate(
    const Eigen::Matrix<double, 7, 1>& tau_d_calculated,
    const Eigen::Matrix<double, 7, 1>& tau_J_d) {  // NOLINT (readability-identifier-naming)
  Eigen::Matrix<double, 7, 1> tau_d_saturated{};
  for (size_t i = 0; i < 7; i++) {
    double difference = tau_d_calculated[i] - tau_J_d[i];
    tau_d_saturated[i] =
        tau_J_d[i] + std::max(std::min(difference, delta_tau_max_), -delta_tau_max_);
  }
  return tau_d_saturated;
}

void MyController::complianceParamCallback(
    franka_example_controllers::compliance_paramConfig& config,
    uint32_t /*level*/) {
  cartesian_stiffness_target_.setIdentity();
  cartesian_stiffness_target_.topLeftCorner(3, 3)
      << config.translational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_stiffness_target_.bottomRightCorner(3, 3)
      << config.rotational_stiffness * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.setIdentity();
  // Damping ratio = 1
  cartesian_damping_target_.topLeftCorner(3, 3)
      << 2.0 * sqrt(config.translational_stiffness) * Eigen::Matrix3d::Identity();
  cartesian_damping_target_.bottomRightCorner(3, 3)
      << 2.0 * sqrt(config.rotational_stiffness) * Eigen::Matrix3d::Identity();
  nullspace_stiffness_target_ = config.nullspace_stiffness;

  target_mass_ = config.desired_mass;
  target_k_p_ = config.k_p;
  target_k_i_ = config.k_i;
  target_k_d_ = config.k_d;

  z_stiffness = config.translational_stiffness;
  z_stiffness_docked = config.z_stiffness;

  radius_target_ = config.radius;
  vel_max_target_ = config.velocity;
  setpoint_target_ = -config.setpoint;
}

void MyController::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  // position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  // Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  // orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
  //     -0.00866312, msg->pose.orientation.w;
  // if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
  //   orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  // }
}

void MyController::desiredMassParamCallback(
    franka_example_controllers::desired_mass_paramConfig& config,
    uint32_t /*level*/) {
  target_mass_ = config.desired_mass;
  target_k_p_ = config.k_p;
  target_k_i_ = config.k_i;
}


}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::MyController,
                       controller_interface::ControllerBase)
