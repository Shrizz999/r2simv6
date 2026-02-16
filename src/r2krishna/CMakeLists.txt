cmake_minimum_required(VERSION 3.8)
project(r2krishna)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(ros_gz_sim REQUIRED)
find_package(ros_gz_bridge REQUIRED)
find_package(ros_gz_image REQUIRED)

# Install resource directories
# Added check to ensure directories exist before installing
install(DIRECTORY
  launch
  config
  meshes
  urdf
  DESTINATION share/${PROJECT_NAME}
)

# Only install rviz directory if it exists to avoid build failure
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/rviz")
  install(DIRECTORY rviz DESTINATION share/${PROJECT_NAME})
endif()

# Install Python scripts
# These are copied to the lib folder where 'ros2 run' looks for them
install(PROGRAMS
  scripts/spawn_block.py
  scripts/vision_tracker.py
  scripts/ramp_detector.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
