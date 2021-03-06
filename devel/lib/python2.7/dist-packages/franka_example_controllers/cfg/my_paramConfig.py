## *********************************************************
##
## File autogenerated for the franka_example_controllers package
## by the dynamic_reconfigure package.
## Please do not edit.
##
## ********************************************************/

from dynamic_reconfigure.encoding import extract_params

inf = float('inf')

config_description = {'upper': 'DEFAULT', 'lower': 'groups', 'srcline': 245, 'name': 'Default', 'parent': 0, 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'cstate': 'true', 'parentname': 'Default', 'class': 'DEFAULT', 'field': 'default', 'state': True, 'parentclass': '', 'groups': [], 'parameters': [{'srcline': 292, 'description': 'Cartesian translational stiffness', 'max': 400.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'translational_stiffness', 'edit_method': '', 'default': 250.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'Cartesian rotational stiffness', 'max': 30.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'rotational_stiffness', 'edit_method': '', 'default': 20.4, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'Stiffness of the joint space nullspace controller (the desired configuration is the one at startup)', 'max': 100.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'nullspace_stiffness', 'edit_method': '', 'default': 0.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'force P gain', 'max': 2.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'k_p', 'edit_method': '', 'default': 0.3, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'force I gain', 'max': 20.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'k_i', 'edit_method': '', 'default': 1.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'force D gain', 'max': 0.1, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'k_d', 'edit_method': '', 'default': 0.01, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'radius', 'max': 0.3, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'radius', 'edit_method': '', 'default': 0.1, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'velocity', 'max': 0.5, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'velocity', 'edit_method': '', 'default': 0.04, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'setpoint', 'max': 5.0, 'cconsttype': 'const double', 'ctype': 'double', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'setpoint', 'edit_method': '', 'default': 4.0, 'level': 0, 'min': 0.0, 'type': 'double'}, {'srcline': 292, 'description': 'Figure parameter', 'max': 3, 'cconsttype': 'const int', 'ctype': 'int', 'srcfile': '/opt/ros/kinetic/lib/python2.7/dist-packages/dynamic_reconfigure/parameter_generator_catkin.py', 'name': 'figure', 'edit_method': "{'enum_description': 'An enum to select a figure', 'enum': [{'srcline': 18, 'description': 'Draws a circle', 'srcfile': '/home/panda1/catkin_ws/src/franka_ros/franka_example_controllers/cfg/my_param.cfg', 'cconsttype': 'const int', 'value': 0, 'ctype': 'int', 'type': 'int', 'name': 'Circle'}, {'srcline': 19, 'description': 'Draws an intricate Lissajous figure', 'srcfile': '/home/panda1/catkin_ws/src/franka_ros/franka_example_controllers/cfg/my_param.cfg', 'cconsttype': 'const int', 'value': 1, 'ctype': 'int', 'type': 'int', 'name': 'Lissajous'}, {'srcline': 20, 'description': 'Makes a beautiful spiral', 'srcfile': '/home/panda1/catkin_ws/src/franka_ros/franka_example_controllers/cfg/my_param.cfg', 'cconsttype': 'const int', 'value': 2, 'ctype': 'int', 'type': 'int', 'name': 'Spiral'}, {'srcline': 21, 'description': 'Not implemented', 'srcfile': '/home/panda1/catkin_ws/src/franka_ros/franka_example_controllers/cfg/my_param.cfg', 'cconsttype': 'const int', 'value': 3, 'ctype': 'int', 'type': 'int', 'name': 'Square'}]}", 'default': 1, 'level': 0, 'min': 0, 'type': 'int'}], 'type': '', 'id': 0}

min = {}
max = {}
defaults = {}
level = {}
type = {}
all_level = 0

#def extract_params(config):
#    params = []
#    params.extend(config['parameters'])
#    for group in config['groups']:
#        params.extend(extract_params(group))
#    return params

for param in extract_params(config_description):
    min[param['name']] = param['min']
    max[param['name']] = param['max']
    defaults[param['name']] = param['default']
    level[param['name']] = param['level']
    type[param['name']] = param['type']
    all_level = all_level | param['level']

my_param_Circle = 0
my_param_Lissajous = 1
my_param_Spiral = 2
my_param_Square = 3
