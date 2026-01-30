cmake_minimum_required(VERSION 3.10.2 FATAL_ERROR)
project(gz_mecanum_drive_controller)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${GAZEBO_CXX_FLAGS}")

# find dependencies
find_package(ament_cmake REQUIRED)


find_package(gz-cmake3 REQUIRED)

find_package(gz-plugin2 REQUIRED COMPONENTS register)
set(GZ_PLUGIN_VER ${gz-plugin2_VERSION_MAJOR})

find_package(gz-sim8 REQUIRED)


include_directories(include/mecanum_drive_controller)
add_library(gz_mecanum_drive_controller SHARED
  src/MecanumDrive.cc
)
set_property(TARGET gz_mecanum_drive_controller PROPERTY CXX_STANDARD 17)
target_link_libraries(gz_mecanum_drive_controller
  PRIVATE gz-plugin${GZ_PLUGIN_VER}::gz-plugin${GZ_PLUGIN_VER}
  PRIVATE gz-sim8::gz-sim8)



#find_package(ament_cmake_python REQUIRED)
#find_package(rclcpp REQUIRED)
#find_package(rclpy REQUIRED)
#find_package(gazebo REQUIRED)
#find_package(gazebo_ros REQUIRED)
#find_package(std_msgs REQUIRED)
#find_package(robot_state_publisher REQUIRED)
#find_package(xacro REQUIRED)

# uncomment the following section in order to fill in
# further dependencies manually.
# find_package(<dependency> REQUIRED)

#gz_add_system(mecanum-drive
#  SOURCES
#    MecanumDrive.cc
#  PUBLIC_LINK_LIBS
#    gz-common${GZ_COMMON_VER}::gz-common${GZ_COMMON_VER}
#)

install(TARGETS
  gz_mecanum_drive_controller
  DESTINATION lib/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
