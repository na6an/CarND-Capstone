from pid import PID
from lowpass import LowPassFilter
from yaw_controller import YawController
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass, fuel_capacity, brake_deadband, decel_limit,
        accel_limit, wheel_radius, wheel_base, steer_ratio, max_lat_accel,
        max_steer_angle):

        self.yaw_controller = YawController(
            wheel_base, steer_ratio, ONE_MPH, max_lat_accel, max_steer_angle)

        kp = 0.5
        ki = 0.0001
        kd = 0.3
        self.throttle_controller = PID(kp, ki, kd, decel_limit, accel_limit)                

        self.acc_filter = LowPassFilter(3., 1.)
        self.steer_filter = LowPassFilter(1., 1.)
        
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.total_mass = vehicle_mass + fuel_capacity*GAS_DENSITY
        self.brake_deadband = brake_deadband

    def control(self, target_velocity, current_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        
        ang_vel_threshold = 0.4
        ang_vel_ratio_min = 0.3
        if abs(target_velocity.angular.z) > ang_vel_threshold:
            target_velocity.linear.x = max(ang_vel_ratio_min * target_velocity.linear.x, 
                min(1.0, ang_vel_threshold / abs(target_velocity.angular.z)) * target_velocity.linear.x)
        
        acceleration = self.throttle_controller.step(
            target_velocity.linear.x,
            current_velocity.linear.x,
            rospy.get_time())
        
        acceleration = self.acc_filter.filt(acceleration)

        torque = (acceleration * self.total_mass) * self.wheel_radius
        max_torque = 1000.

        # there is a constant bias of throttle we need to correct for
        torque -= 0.02 * max_torque

        # for idle state, we apply a small constant brake :
        if target_velocity.linear.x < 0.05:
            throttle = 0.0
            brake = 100.            
        elif acceleration > 0.0:
            throttle = min(.25, max(torque, 0.) / max_torque)
            brake = 0.
        else: 
            throttle = 0.0
            brake = max(0., -torque)
 
          # Steering
        steering = self.yaw_controller.get_steering(
            target_velocity.linear.x,
            target_velocity.angular.z,
            current_velocity.linear.x)
        steering = self.steer_filter.filt(steering)

        return throttle, brake, steering
    
    def reset(self):
        self.throttle_controller.reset()