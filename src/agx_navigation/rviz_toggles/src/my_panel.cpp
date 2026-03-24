#include "rviz_toggles/my_panel.hpp"

#include <QPushButton>
#include <QVBoxLayout>
#include <QHBoxLayout>
#include <QLabel>
#include <QTimer>
#include <QString>
#include <memory>

#include <rviz_common/display_context.hpp>
#include <std_srvs/srv/set_bool.hpp>
#include <std_srvs/srv/empty.hpp>

namespace rviz_toggles
{

MyCommandPanel::MyCommandPanel(QWidget * parent)
: rviz_common::Panel(parent), exploration_active_(false)
{
  // --- Build the UI ---
  auto * layout = new QVBoxLayout(this);

  // Status label at the top
  status_label_ = new QLabel("Ready");
  status_label_->setAlignment(Qt::AlignCenter);
  layout->addWidget(status_label_);

  // Exploration section
  auto * exploration_label = new QLabel("<b>Exploration</b>");
  layout->addWidget(exploration_label);

  toggle_exploration_btn_ = new QPushButton("Start Exploration");
  layout->addWidget(toggle_exploration_btn_);

  // RTABMap section
  auto * rtab_label = new QLabel("<b>RTAB-Map Commands</b>");
  layout->addWidget(rtab_label);

  clear_map_btn_ = new QPushButton("Clear RTAB-Map");
  layout->addWidget(clear_map_btn_);

  layout->addStretch();  // push widgets to top
  setLayout(layout);

  // Connect Qt signals to slots
  // These are connected here but service clients are created in onInitialize()
  connect(toggle_exploration_btn_, &QPushButton::clicked,
    this, &MyCommandPanel::onToggleExplorationClicked);
  connect(clear_map_btn_, &QPushButton::clicked,
    this, &MyCommandPanel::onClearMapClicked);
}

void MyCommandPanel::onInitialize()
{
  // IMPORTANT: get the node from RViz2's context — don't spin up your own
  // getRosNodeAbstraction() returns a wrapper; lock() gives the actual node
  node_ = getDisplayContext()->getRosNodeAbstraction().lock()->get_raw_node();

  // Create service clients
  toggle_exploration_client_ = node_->create_client<std_srvs::srv::SetBool>("/toggle_exploration");
  clear_map_client_ = node_->create_client<std_srvs::srv::Empty>("/rtabmap/reset");
}

void MyCommandPanel::onToggleExplorationClicked()
{
  if (!toggle_exploration_client_->service_is_ready()) {
    status_label_->setText("Error: Toggle service not available");
    return;
  }

  exploration_active_ = !exploration_active_;
  auto request = std::make_shared<std_srvs::srv::SetBool::Request>();
  request->data = exploration_active_;

  toggle_exploration_client_->async_send_request(request,
    [this](rclcpp::Client<std_srvs::srv::SetBool>::SharedFuture future) {
      auto result = future.get();
      status_label_->setText(QString::fromStdString(result->message));
      if (result->success) {
        toggle_exploration_btn_->setText(exploration_active_ ? "Stop Exploration" : "Start Exploration");
      }
    });
}

void MyCommandPanel::onClearMapClicked()
{
  if (!clear_map_client_->service_is_ready()) {
    status_label_->setText("Error: Clear map service not available");
    return;
  }

  auto request = std::make_shared<std_srvs::srv::Empty::Request>();

  clear_map_client_->async_send_request(request,
    [this](rclcpp::Client<std_srvs::srv::Empty>::SharedFuture future) {
      try {
        future.get();
        status_label_->setText("RTABMap cleared");
      } catch (const std::exception & e) {
        status_label_->setText(QString::fromStdString("Error: " + std::string(e.what())));
      }
    });
}

void MyCommandPanel::save(rviz_common::Config config) const
{
  // Save any panel state to the .rviz config file
  Panel::save(config);
  // e.g.: config.mapSetValue("SomeParameter", some_value_);
}

void MyCommandPanel::load(const rviz_common::Config & config)
{
  // Load panel state from the .rviz config file
  Panel::load(config);
}

} // namespace my_rviz_panels

// This macro registers the class with pluginlib
// It must be in the .cpp file, not the header
#include <pluginlib/class_list_macros.hpp>
PLUGINLIB_EXPORT_CLASS(rviz_toggles::MyCommandPanel, rviz_common::Panel)