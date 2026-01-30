import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import numpy as np

class YOLOTracker(Node):
    def __init__(self):
        super().__init__('yolo_tracker_node')

        # Cargar modelo YOLO11
        self.model = YOLO('/home/taller/Escritorio/Taller-Rob-tica/paquito-virtual/models')  # Ruta a tu modelo entrenado

        # Bridge ROS-OpenCV
        self.bridge = CvBridge()

        # Subscribers y Publishers
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image',
            self.image_callback,
            10
        )

        self.vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        # Parámetros de control
        self.image_width = 640
        self.image_height = 480
        self.target_area_min = 5000  # Área mínima para detección válida
        self.kp_linear = 0.001  # Ganancia proporcional lineal
        self.kp_angular = 0.005  # Ganancia proporcional angular

        self.get_logger().info('YOLO Tracker initialized')

    def image_callback(self, msg):
        try:
            # Convertir ROS Image a OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Detección YOLO
            results = self.model(cv_image, conf=0.5, verbose=False)

            # Procesar detecciones
            self.process_detections(results, cv_image)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')

    def process_detections(self, results, image):
        vel_msg = Twist()

        if len(results[0].boxes) > 0:
            # Obtener la detección con mayor confianza
            boxes = results[0].boxes
            best_box = boxes[0]  # Ya están ordenadas por confianza

            # Extraer coordenadas
            x1, y1, x2, y2 = best_box.xyxy[0].cpu().numpy()

            # Calcular centro del objeto
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            area = (x2 - x1) * (y2 - y1)

            # Centro de la imagen
            image_center_x = self.image_width / 2

            # Calcular errores
            error_x = center_x - image_center_x
            error_area = self.target_area_min - area

            # Control proporcional
            # Girar para centrar objeto
            vel_msg.angular.z = -self.kp_angular * error_x

            # Avanzar/retroceder según tamaño
            if area < self.target_area_min * 0.8:
                vel_msg.linear.x = 0.3  # Acercarse
            elif area > self.target_area_min * 1.2:
                vel_msg.linear.x = -0.2  # Alejarse
            else:
                vel_msg.linear.x = 0.0  # Mantener distancia

            # Limitar velocidades
            vel_msg.linear.x = np.clip(vel_msg.linear.x, -0.5, 0.5)
            vel_msg.angular.z = np.clip(vel_msg.angular.z, -1.0, 1.0)

            self.get_logger().info(f'Object detected at ({center_x:.0f}, {center_y:.0f}), area: {area:.0f}')

        else:
            # Sin detección: girar para buscar
            vel_msg.angular.z = 0.3
            self.get_logger().info('Searching for object...')

        self.vel_pub.publish(vel_msg)

def main(args=None):
    rclpy.init(args=args)
    node = YOLOTracker()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Shutting down YOLO tracker...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
