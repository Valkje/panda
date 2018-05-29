// Copyright (c) 2017 Franka Emika GmbH
// Use of this source code is governed by the Apache-2.0 license, see LICENSE
#include <franka_example_controllers/my_controller_v2.h>

#include <cmath>
#include <stdexcept>
#include <string>

#include <controller_interface/controller_base.h>
#include <franka/robot_state.h>
#include <pluginlib/class_list_macros.h>
#include <ros/ros.h>

#include "pseudo_inversion.h"

#include <franka_hw/franka_cartesian_command_interface.h>
#include <hardware_interface/hardware_interface.h>

namespace franka_example_controllers {

bool MyControllerV2::init(hardware_interface::RobotHW* robot_hw,
                                               ros::NodeHandle& node_handle) {
  // Impedance part:  
  std::vector<double> cartesian_stiffness_vector;
  std::vector<double> cartesian_damping_vector;

  sub_equilibrium_pose_ = node_handle.subscribe(
      "/equilibrium_pose", 20, &MyControllerV2::equilibriumPoseCallback, this,
      ros::TransportHints().reliable().tcpNoDelay());

  std::string arm_id;
  if (!node_handle.getParam("arm_id", arm_id)) {
    ROS_ERROR_STREAM("MyControllerV2: Could not read parameter arm_id");
    return false;
  }
  std::vector<std::string> joint_names;
  if (!node_handle.getParam("joint_names", joint_names) || joint_names.size() != 7) {
    ROS_ERROR(
        "MyControllerV2: Invalid or no joint_names parameters provided, "
        "aborting controller init!");
    return false;
  }

  franka_hw::FrankaModelInterface* model_interface =
      robot_hw->get<franka_hw::FrankaModelInterface>();
  if (model_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MyControllerV2: Error getting model interface from hardware");
    return false;
  }
  try {
    model_handle_.reset(
        new franka_hw::FrankaModelHandle(model_interface->getHandle(arm_id + "_model")));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "MyControllerV2: Exception getting model handle from interface: "
        << ex.what());
    return false;
  }

  franka_hw::FrankaStateInterface* state_interface =
      robot_hw->get<franka_hw::FrankaStateInterface>();
  if (state_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MyControllerV2: Error getting state interface from hardware");
    return false;
  }
  try {
    state_handle_.reset(
        new franka_hw::FrankaStateHandle(state_interface->getHandle(arm_id + "_robot")));
  } catch (hardware_interface::HardwareInterfaceException& ex) {
    ROS_ERROR_STREAM(
        "MyControllerV2: Exception getting state handle from interface: "
        << ex.what());
    return false;
  }

  hardware_interface::EffortJointInterface* effort_joint_interface =
      robot_hw->get<hardware_interface::EffortJointInterface>();
  if (effort_joint_interface == nullptr) {
    ROS_ERROR_STREAM(
        "MyControllerV2: Error getting effort joint interface from hardware");
    return false;
  }
  for (size_t i = 0; i < 7; ++i) {
    try {
      joint_handles_.push_back(effort_joint_interface->getHandle(joint_names[i]));
    } catch (const hardware_interface::HardwareInterfaceException& ex) {
      ROS_ERROR_STREAM(
          "MyControllerV2: Exception getting joint handles: " << ex.what());
      return false;
    }
  }

  dynamic_reconfigure_compliance_param_node_ =
      ros::NodeHandle("dynamic_reconfigure_compliance_param_node");

  dynamic_server_compliance_param_.reset(
      new dynamic_reconfigure::Server<franka_example_controllers::compliance_paramConfig>(
          dynamic_reconfigure_compliance_param_node_));
  dynamic_server_compliance_param_->setCallback(
      boost::bind(&MyControllerV2::complianceParamCallback, this, _1, _2));

  position_d_.setZero();
  orientation_d_.coeffs() << 0.0, 0.0, 0.0, 1.0;
  position_d_target_.setZero();
  orientation_d_target_.coeffs() << 0.0, 0.0, 0.0, 1.0;

  cartesian_stiffness_.setZero();
  cartesian_damping_.setZero();

  // Cartesian pose control:


  cartesian_pose_interface_ = robot_hw->get<franka_hw::FrankaPoseCartesianInterface>();
  if (cartesian_pose_interface_ == nullptr) {
    ROS_ERROR(
        "CartesianPoseExampleController: Could not get Cartesian Pose "
        "interface from hardware");
    return false;
  }
  
  try {
    cartesian_pose_handle_.reset(new franka_hw::FrankaCartesianPoseHandle(
        cartesian_pose_interface_->getHandle(arm_id + "_robot")));
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseExampleController: Exception getting Cartesian handle: " << e.what());
    return false;
  }

 /*
  try {
    auto state_handle = state_interface->getHandle(arm_id + "_robot");

    std::array<double, 7> q_start{{0, -M_PI_4, 0, -3 * M_PI_4, 0, M_PI_2, M_PI_4}};
    for (size_t i = 0; i < q_start.size(); i++) {
      if (std::abs(state_handle.getRobotState().q_d[i] - q_start[i]) > 0.1) {
        ROS_ERROR_STREAM(
            "CartesianPoseExampleController: Robot is not in the expected starting position for "
            "running this example. Run `roslaunch franka_example_controllers move_to_start.launch "
            "robot_ip:=<robot-ip> load_gripper:=<has-attached-gripper>` first.");
        return false;
      }
    }
  } catch (const hardware_interface::HardwareInterfaceException& e) {
    ROS_ERROR_STREAM(
        "CartesianPoseExampleController: Exception getting state handle: " << e.what());
    return false;
  }
  */

  return true;
}

void MyControllerV2::starting(const ros::Time& /*time*/) {
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
  
  //std::cout << "D: \n" << position_d_ << "\n";

  // set nullspace equilibrium configuration to initial q
  q_d_nullspace_ << 
    -0.269985,
    0.272253,
    -0.117828,
    -2.35781,
    0.110962,
    2.65076,
    0.484513;
    
  //std::cout << "Ours: \n" << q_initial << "\n End of Ours. \n";
  
  // ======================================================================= //
  // Force control
  franka::RobotState robot_state = state_handle_->getRobotState();
  std::array<double, 7> gravity_array = model_handle_->getGravity();
  Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_measured(robot_state.tau_J.data());
  Eigen::Map<Eigen::Matrix<double, 7, 1> > gravity(gravity_array.data());
  // Bias correction for the current external torque
  tau_ext_initial_ = tau_measured - gravity;
  tau_error_.setZero();

  // Cartesian pose control:
  initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  cartesian_pose_handle_->setCommand(initial_pose_);
  elapsed_time_ = ros::Duration(0.0);

}

void MyControllerV2::update(const ros::Time& /*time*/,
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


  double force_EE_z = robot_state.K_F_ext_hat_K[2];
  
  if (force_EE_z > lambda2 && position.z() < height_threshold){
    state = "Docked";
    initial_pose_ = cartesian_pose_handle_->getRobotState().O_T_EE_d;
  }
  
  if (force_EE_z < lambda1 || position.z() >= height_threshold){
    state = "Free";
  }
  //std::cout << state << "\n";
  //std::cout << force_EE_z << "\t" << position.z() << "\n";


  if (state.compare("Free") == 0){
    // compute error to desired pose
    // position error
    Eigen::Matrix<double, 6, 1> error;
    error.head(3) << position - position_d_;

    // orientation error
    if (orientation_d_.coeffs().dot(orientation.coeffs()) < 0.0) {
      orientation.coeffs() << -orientation.coeffs();
    }
    //std::cout << "Orientation:\n" << orientation.coeffs() << "\n";

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
                         (nullspace_stiffness_ * (q_d_nullspace_ - q) -
                          (2.0 * sqrt(nullspace_stiffness_)) * dq);
    // Desired torque
    tau_d << tau_task + tau_nullspace + coriolis;
    // Saturate torque rate to avoid discontinuities
    tau_d << saturateTorqueRate(tau_d, tau_J_d);

    for (size_t i = 0; i < 7; ++i) {
      joint_handles_[i].setCommand(tau_d(i));
    }

  //  position_d_.z() = 0.08;

    if (counter % 1000 == 0) {
      //std::cout << "\033[2J\033[1;1H";
      std::array<double, 7> tau_j = robot_state.tau_J; 
      for (size_t i = 0; i < 7; ++i) {   
        //std::cout << "tau_d(" << i << ") " << tau_d(i) << "\n";
        //std::cout << "tau_J(" << i << ") " << tau_j[i] << "\n";
      }
      //std::cout << "Position_d: " << position_d_ << "\n";
      //std::cout << "Position: " << position << "\n";
    }
    counter = (counter + 1) % 1000;

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
  }
  
  // Force control
  if (state.compare("Docked") == 0){
    
    //======================================= Cartesian pose control:
    
    elapsed_time_ += period;

    double radius = 0.1;
    double angle = M_PI / 4 * (1 - std::cos(M_PI / 5.0 * elapsed_time_.toSec()));
    double delta_x = radius * std::sin(angle);
    double delta_y = radius * (std::cos(angle) - 1);
    std::array<double, 16> new_pose = initial_pose_;
    new_pose[12] -= delta_x;
    new_pose[13] -= delta_y;
    cartesian_pose_handle_->setCommand(new_pose);

    //=======================================
    
    std::array<double, 7> gravity_array = model_handle_->getGravity();
    Eigen::Map<Eigen::Matrix<double, 7, 1> > tau_measured(robot_state.tau_J.data());
    Eigen::Map<Eigen::Matrix<double, 7, 1> > gravity(gravity_array.data());

    Eigen::VectorXd tau_d(7), desired_force_torque(6), tau_cmd(7), tau_ext(7);
    desired_force_torque.setZero();
    desired_force_torque(2) = desired_mass_ * -9.81;
    tau_ext = tau_measured - gravity - tau_ext_initial_;
    tau_d << jacobian.transpose() * desired_force_torque;
    tau_error_ = tau_error_ + period.toSec() * (tau_d - tau_ext);
    // FF + PI control (PI gains are initially all 0)
    tau_cmd = tau_d + k_p_ * (tau_d - tau_ext) + k_i_ * tau_error_;
    tau_cmd << saturateTorqueRate(tau_cmd, tau_J_d);

    for (size_t i = 0; i < 7; ++i) {
      //joint_handles_[i].setCommand(tau_cmd(i));
    }

    

    // Update signals changed online through dynamic reconfigure
    desired_mass_ = filter_gain_ * target_mass_ + (1 - filter_gain_) * desired_mass_;
    k_p_ = filter_gain_ * target_k_p_ + (1 - filter_gain_) * k_p_;
    k_i_ = filter_gain_ * target_k_i_ + (1 - filter_gain_) * k_i_;

    position_d_ = filter_params_ * position_d_target_ + (1.0 - filter_params_) * position_d_;
    Eigen::AngleAxisd aa_orientation_d(orientation_d_);
    Eigen::AngleAxisd aa_orientation_d_target(orientation_d_target_);
    aa_orientation_d.axis() = filter_params_ * aa_orientation_d_target.axis() +
                              (1.0 - filter_params_) * aa_orientation_d.axis();
    aa_orientation_d.angle() = filter_params_ * aa_orientation_d_target.angle() +
                               (1.0 - filter_params_) * aa_orientation_d.angle();
    orientation_d_ = Eigen::Quaterniond(aa_orientation_d);
    

    if (counter % 1000 == 0) {
      //std::cout << "\033[2J\033[1;1H";
      std::array<double, 7> tau_j = robot_state.tau_J; 
      for (size_t i = 0; i < 7; ++i) {   
        //std::cout << "tau_d(" << i << ") " << tau_d(i) << "\n";
        //std::cout << "tau_ext(" << i << ") " << tau_ext(i) << "\n";
      }
      //std::cout << "Position_d: " << position_d_ << "\n";
      //std::cout << "Position: " << position << "\n";
    }
    counter = (counter + 1) % 1000;
  }
}

Eigen::Matrix<double, 7, 1> MyControllerV2::saturateTorqueRate(
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

void MyControllerV2::complianceParamCallback(
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
}

void MyControllerV2::equilibriumPoseCallback(
    const geometry_msgs::PoseStampedConstPtr& msg) {
  position_d_target_ << msg->pose.position.x, msg->pose.position.y, msg->pose.position.z;
  Eigen::Quaterniond last_orientation_d_target(orientation_d_target_);
  orientation_d_target_.coeffs() << msg->pose.orientation.x, msg->pose.orientation.y,
      -0.00866312, msg->pose.orientation.w;
  if (last_orientation_d_target.coeffs().dot(orientation_d_target_.coeffs()) < 0.0) {
    orientation_d_target_.coeffs() << -orientation_d_target_.coeffs();
  }

  //std::cout << position_d_target_;
}

void MyControllerV2::desiredMassParamCallback(
    franka_example_controllers::desired_mass_paramConfig& config,
    uint32_t /*level*/) {
  target_mass_ = config.desired_mass;
  target_k_p_ = config.k_p;
  target_k_i_ = config.k_i;
}


}  // namespace franka_example_controllers

PLUGINLIB_EXPORT_CLASS(franka_example_controllers::MyControllerV2,
                       controller_interface::ControllerBase)
