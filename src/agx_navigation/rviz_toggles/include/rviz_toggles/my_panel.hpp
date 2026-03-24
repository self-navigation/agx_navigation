#pragma once

#include <rclcpp/rclcpp.hpp>
#include <rviz_common/panel.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/set_bool.hpp>

// Qt forward declarations
class QPushButton;
class QLineEdit;
class QLabel;

namespace rviz_toggles
{

class MyCommandPanel : public rviz_common::Panel
{
  // RViz2 panels use Qt — Q_OBJECT is mandatory
  Q_OBJECT

public:
  // This constructor signature is required by pluginlib
  explicit MyCommandPanel(QWidget * parent = nullptr);
  ~MyCommandPanel() override = default;

  // Called when the panel is initialized by RViz2
  // This is where you get access to the RViz2 node
  void onInitialize() override;

  // Optional: save/load panel config from .rviz file
  void save(rviz_common::Config config) const override;
  void load(const rviz_common::Config & config) override;

private Q_SLOTS:
  // Qt slots — called when buttons are clicked
  void onToggleExplorationClicked();
  void onClearMapClicked();

private:
  // Reuse the node that RViz2 already has — don't create a new one
  rclcpp::Node::SharedPtr node_;

  // Service clients
  rclcpp::Client<std_srvs::srv::SetBool>::SharedPtr toggle_exploration_client_;
  rclcpp::Client<std_srvs::srv::Empty>::SharedPtr clear_map_client_;

  // UI elements
  QPushButton * toggle_exploration_btn_;
  QPushButton * clear_map_btn_;
  QLabel * status_label_;

  // State tracking
  bool exploration_active_;
};

}