from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, ExecuteProcess
from launch.substitutions import LaunchConfiguration, TextSubstitution, EnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.conditions import IfCondition
from ament_index_python.packages import get_package_share_directory
from launch.launch_description_sources import AnyLaunchDescriptionSource
import os

def generate_launch_description():
    # Launch arguments
    robot_ip_arg = DeclareLaunchArgument(
        'robot_ip',
        default_value=EnvironmentVariable('ROBOT_IP', default_value='yyy.yyy.yyy.yyy'),
        description='IP address of the UR3e robot. Set via command line or ROBOT_IP environment variable'
    )
    
    use_fake_hardware_arg = DeclareLaunchArgument(
        'use_fake_hardware',
        default_value=EnvironmentVariable('FAKE_HARDWARE', default_value='true'),
        description='Use fake hardware if true'
    )
    
    launch_rviz_arg = DeclareLaunchArgument(
        'launch_rviz',
        default_value='false',
        description='Launch RViz if true'
    )
    
    initial_controller_arg = DeclareLaunchArgument(
        'initial_joint_controller',
        default_value='scaled_joint_trajectory_controller',
        description='Initial joint controller to start'
    )
    
    launch_website_arg = DeclareLaunchArgument(
        'launch_website',
        default_value='true',
        description='Launch Pablo website if true'
    )
    
    # Create launch configurations
    robot_ip = LaunchConfiguration('robot_ip')
    use_fake_hardware = LaunchConfiguration('use_fake_hardware')
    launch_rviz = LaunchConfiguration('launch_rviz')
    initial_joint_controller = LaunchConfiguration('initial_joint_controller')
    launch_website = LaunchConfiguration('launch_website')
    
    # Get package share directories
    ur_robot_driver_dir = get_package_share_directory('ur_robot_driver')
    ur_moveit_config_dir = get_package_share_directory('ur_moveit_config')
    
    # Include UR3e Controller launch file
    ur_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_robot_driver_dir, 'launch', 'ur_control.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur3e',
            'robot_ip': robot_ip,
            'initial_joint_controller': initial_joint_controller,
            'use_fake_hardware': use_fake_hardware,
            'launch_rviz': 'true'  # We'll launch RViz from the MoveIt config
        }.items()
    )
    
    # Include UR3e MoveIt launch file
    ur_moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(ur_moveit_config_dir, 'launch', 'ur_moveit.launch.py')
        ),
        launch_arguments={
            'ur_type': 'ur3e',
            'launch_rviz': launch_rviz
        }.items()
    )
    
    # Define Pablo nodes
    image_processor_node = Node(
        package='pablo',
        executable='image_processor.py',
        name='image_processor',
        output='screen'
    )
    
    path_planning_node = Node(
        package='pablo',
        executable='path_planning',
        name='path_planning',
        output='screen'
    )
    
    ur3e_control_node = Node(
        package='pablo',
        executable='ur3e_control',
        name='ur3e_control',
        output='screen'
    )

    # Launch ROSBridge Server
    rosbridge_server = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(get_package_share_directory('rosbridge_server'), 'launch', 'rosbridge_websocket.launch.xml')
        )
    )

    # Start Python Server
    python_server = ExecuteProcess(
        cmd=['python3', '-m', 'http.server', '8080'],
        output='screen',
        cwd=os.path.join(get_package_share_directory('pablo'), 'web')
    )

    # Load Webpage GUI
    launch_pablo_website = ExecuteProcess(
        # cmd=['xdg-open', '/home/edan/git/41069_WS_LAB4_G1/pablo/web/pablo3.html'],
        cmd=['xdg-open', os.path.join(get_package_share_directory('pablo'), 'web', 'pabloUI.html')],
        output='screen',
        # condition=IfCondition(launch_website)
    )

    # Log info message
    log_info = LogInfo(
        msg=['Launching UR3e with Pablo nodes and website']
    )
    
    # Return the launch description
    return LaunchDescription([
        # Launch arguments
        robot_ip_arg,
        use_fake_hardware_arg,
        launch_rviz_arg,
        initial_controller_arg,
        launch_website_arg,
        # Log info
        log_info,
        # Launch files
        ur_moveit_launch,
        ur_control_launch,
        # Pablo nodes
        image_processor_node,
        path_planning_node,
        # ur3e_control_node # (uncommented if needed)
        # Launch website
        # rosbridge_server,
        python_server
        # launch_pablo_website
    ])