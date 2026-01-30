#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/joy.hpp"
#include <iostream>
#include <vector>

#include <geometry_msgs/msg/twist.hpp>

//#include "paquito.hpp"

#define PS2
/// Control PS2
#if defined(PS2)
const unsigned int X_AXIS = 1;
const unsigned int Y_AXIS = 0;
const unsigned int Z_AXIS = 3;  // ?

const unsigned int STOP = 3;
const unsigned int SPEED_UP = 2;
const unsigned int SPEED_DOWN = 0;
/// Control ZhiXu
#elif defined(ZHI_XU)
const unsigned int X_AXIS = 1;      // Joy izquierdo
const unsigned int Y_AXIS = 0;
const unsigned int Z_AXIS = 2;      // Joy derecho horizontal

const unsigned int STOP = 9;        // Atrás inferior derecha
const unsigned int SPEED_UP = 4;    // X (Botón superior)
const unsigned int SPEED_DOWN = 3;  // Y (Botón izquierdo)
#endif

// Generado con ayuda de Gemini 2.5 Pro y adaptado
using std::placeholders::_1;

// La clase hereda de rclcpp::Node, la estructura base de un nodo en ROS 2
class JoySubscriber : public rclcpp::Node
{
public:
    // Constructor
    JoySubscriber() : Node("ps_controller_reader")
    {
        // Creamos la suscripción.
        // El tipo de mensaje es sensor_msgs::msg::Joy.
        // El '10' es la profundidad de la cola (Quality of Service).
        // El callback se enlaza usando std::bind.
        _subscription = this->create_subscription<sensor_msgs::msg::Joy>(
            "joy", 10, std::bind(&JoySubscriber::joy_callback, this, _1));

        // Publicará las velocidades solicitadas según el uso del control PS2
        _vel_publisher = this->create_publisher<geometry_msgs::msg::Twist>("cmd_vel", 10);


        RCLCPP_INFO(this->get_logger(), "Nodo de control de PlayStation iniciado. Esperando mensajes en /joy...  Publicando en /cmd_vel...");
    }

private:
    void log_state(const sensor_msgs::msg::Joy::SharedPtr msg)
    {
        std::stringstream ss;

        // Imprime el estado de botones, ejes y variables de interés
        // msg->axes es un vector de floats, generalmente entre -1.0 y 1.0
        RCLCPP_INFO(this->get_logger(), "Botones:");
        ss << " Eje x " << X_AXIS << ": " << msg->axes[X_AXIS] << std::endl;
        ss << " Eje y " << Y_AXIS << ": " << msg->axes[Y_AXIS] << std::endl;
        ss << " Eje z (giro) " << Z_AXIS << ": " << msg->axes[Z_AXIS] << std::endl;

        ss << " Botón acelera " << SPEED_UP << ": " << (msg->buttons[SPEED_UP] ? "Presionado" : "Suelto") << std::endl;
        ss << " Botón frena " << SPEED_DOWN << ": " << (msg->buttons[SPEED_DOWN] ? "Presionado" : "Suelto") << std::endl;
        ss << " Botón alto " << STOP << ": " << (msg->buttons[STOP] ? "Presionado" : "Suelto") << std::endl;

        ss << " Rapidez = " << speed << std::endl;

        RCLCPP_INFO(this->get_logger(), "%s", ss.str().c_str());
    }

    // Función de callback que se ejecuta al recibir un mensaje
    void joy_callback(const sensor_msgs::msg::Joy::SharedPtr msg)
    {
        geometry_msgs::msg::Twist twist;

        if (msg->buttons[SPEED_DOWN]) {
            if (speed > MIN_SPEED) {
                speed -= SPEED_STEP;
            }
        } else if(msg->buttons[SPEED_UP]) {
            if (speed < MAX_SPEED) {
                speed += SPEED_STEP;
            }
        } else if(msg->buttons[STOP]) {
            speed = 0;
        }

        twist.linear.x = msg->axes[X_AXIS] * speed;
        twist.linear.y = msg->axes[Y_AXIS] * speed;
        twist.angular.z = msg->axes[Z_AXIS] * speed;

        //log_state(msg);

        _vel_publisher->publish(twist);
    }

    rclcpp::Subscription<sensor_msgs::msg::Joy>::SharedPtr _subscription;
    rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr _vel_publisher;
    double speed = 0;               // m/s
    const double SPEED_STEP = 0.1;
    const double MAX_SPEED = 1.0;   // m/s
    const double MIN_SPEED = -1.0;  // m/s
};

int main(int argc, char * argv[])
{
    // Inicializa ROS 2
    rclcpp::init(argc, argv);

    // Crea una instancia del nodo y lo mantiene vivo hasta que se detenga (Ctrl+C)
    rclcpp::spin(std::make_shared<JoySubscriber>());

    // Libera los recursos de ROS 2
    rclcpp::shutdown();
    return 0;
}
