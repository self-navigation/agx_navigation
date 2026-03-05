import rclpy
from rclpy.node import Node

from std_msgs.msg import String
from scout_msgs import msg

msg.ScoutActuatorState

class MinimalSubscriber(Node):

    def __init__(self):
        super().__init__('minimal_subscriber')
        self.subscription = self.create_subscription(
            msg.ScoutStatus,
            '/scout_status',
            self.status_callback,
            10)
        self.subscription  # prevent unused variable warning

    def status_callback(self, msg: msg.ScoutStatus):
        # print(f'I heard: {msg}')
        pass


def main(args=None):
    rclpy.init(args=args)

    minimal_subscriber = MinimalSubscriber()

    rclpy.spin(minimal_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    minimal_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()