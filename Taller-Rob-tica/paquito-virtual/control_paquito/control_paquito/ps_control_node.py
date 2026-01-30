import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Joy
from std_msgs.msg import String
from geometry_msgs.msg import Twist

class JoyControl(Node):
    EPSILON = 0.01
    MAX_LINE_SPEED = 0.5  # m/s
    MAX_ANGLE_SPEED = 3.1416 / 2 # rad/s

    def __init__(self):
        super().__init__('ps_control_node')
        self.joy_subscription = self.create_subscription(
            Joy,
            '/joy',
            self.joy_listener_callback,
            10
        )
        self.joy_subscription

        self.string_command_publisher = self.create_publisher(
            String,
            '/command_for_paquito',
            10
        )

        self.vel_command_publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.get_logger().info(f"Escuchando al control de mando")

    def joy_listener_callback(self, msg):
        #self.get_logger().info(f"Mensaje: {msg}")
        axes = msg.axes
        buttons = msg.buttons

        

        # if 1.0 - axes[7] < JoyControl.EPSILON:
        if axes[7] == 1.0:
            # Flecha arriba: FL
            msg = String()
            self.get_logger().debug(f"FL")
            msg.data = 'FL'
            self.string_command_publisher.publish(msg)
        if axes[6] == -1.0:
            # Flecha a la derecha: FR
            msg = String()
            self.get_logger().debug(f"FR")
            msg.data = 'FR'
            self.string_command_publisher.publish(msg)
        if axes[7] == -1.0:
            # Flecha arriba: RR
            msg = String()
            self.get_logger().debug(f"RR")
            msg.data = 'RR'
            self.string_command_publisher.publish(msg)
        if axes[6] == 1.0:
            # Flecha arriba: RL
            msg = String()
            self.get_logger().debug(f"RL")
            msg.data = 'RL'
            self.string_command_publisher.publish(msg)

        if buttons[2] == 1:
            # Triángulo
            msg = String()
            self.get_logger().debug(f"speak")
            msg.data = 'speak'
            self.string_command_publisher.publish(msg)
        else:
            ax_y = axes[0]
            ax_x = axes[1]
            ax_wz = axes[3]

            vel_msg = Twist()
            vel_msg.linear.x = ax_x * JoyControl.MAX_LINE_SPEED
            vel_msg.linear.y = ax_y * JoyControl.MAX_LINE_SPEED
            vel_msg.angular.z = ax_wz * JoyControl.MAX_ANGLE_SPEED
            self.vel_command_publisher.publish(vel_msg)



def main(args=None):
    rclpy.init(args=args)
    node = JoyControl()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Deteniendo nodo...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
