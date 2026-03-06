cmake_minimum_required(VERSION 3.8)
project(pointcloud_utils)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(sensor_msgs REQUIRED)

macro(add_component LIB_NAME SOURCE_FILE PLUGIN_CLASS)
  add_library(${LIB_NAME} SHARED src/${SOURCE_FILE})
  target_include_directories(${LIB_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
  )
  ament_target_dependencies(${LIB_NAME}
    rclcpp rclcpp_components sensor_msgs
  )
  rclcpp_components_register_node(${LIB_NAME}
    PLUGIN "${PLUGIN_CLASS}"
    EXECUTABLE ${LIB_NAME}_node
  )
endmacro()

add_component(pointcloud_field_stripper
  pointcloud_field_stripper.cpp
  "pointcloud_utils::PointCloudFieldStripper"
)
add_component(laserscan_interpolator
  laserscan_interpolator.cpp
  "pointcloud_utils::LaserScanInterpolator"
)

install(TARGETS pointcloud_field_stripper laserscan_interpolator
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
