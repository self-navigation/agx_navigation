cmake_minimum_required(VERSION 3.8)
project(rviz_toggles)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rviz_common REQUIRED)
find_package(rviz2 REQUIRED)
find_package(std_srvs REQUIRED)
find_package(pluginlib REQUIRED)
find_package(Qt5 REQUIRED COMPONENTS Widgets Core)

# Qt requires this for MOC (Meta-Object Compiler) processing
set(CMAKE_AUTOMOC ON)

add_library(${PROJECT_NAME} SHARED
  src/my_panel.cpp
  include/rviz_toggles/my_panel.hpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

ament_target_dependencies(${PROJECT_NAME}
  rclcpp
  rviz_common
  std_srvs
  pluginlib
)

target_link_libraries(${PROJECT_NAME}
  Qt5::Widgets
  Qt5::Core
)

# This is what makes RViz2 discover your plugin
pluginlib_export_plugin_description_file(rviz_common plugin_description.xml)

install(TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

ament_package()