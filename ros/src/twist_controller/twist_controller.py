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
            wheel_base, steer_ratio, 0.5, max_lat_accel, max_steer_angle)

        kp = 0.5
        ki = 0.0001
        kd = 0.
        self.throttle_controller = PID(kp, ki, kd)
        
        max_speed = 40
        self.max_abs_u = (abs(kp) + abs(ki) + abs(kd)) * abs(max_speed)

        tau = 0.5 # 1/(2pi * tau) = cutoff frequency
        ts = 0.02 # sample time
        self.vel_lpf = LowPassFilter(tau, ts)
        
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        total_mass = vehicle_mass + fuel_capacity*GAS_DENSITY
        self.min_brake = -1.0 * brake_deadband
        self.max_brake_torque = total_mass * decel_limit * wheel_radius
        

    def control(self, target_velocity, current_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        u = self.throttle_controller.step(
            target_velocity.linear.x,
            current_velocity.linear.x,
            rospy.get_time())

        # Clamp
        if u > 0:
            # Positive
            throttle = max(0.0, min(1.0, u))
            brake = 0
        else:
            throttle = 0
            brake = self.max_brake_torque * min(self.min_brake, u/self.max_abs_u)      

        # Steering
        steering = self.yaw_controller.get_steering(
            target_velocity.linear.x,
            target_velocity.angular.z,
            current_velocity.linear.x)
        steering = self.vel_lpf.filt(steering)

        return throttle, brake, steering

def reset(self):
    self.pid.reset()
