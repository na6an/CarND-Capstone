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
        kd = 0.0001
        self.throttle_controller = PID(kp, ki, kd, decel_limit, accel_limit)                

        tau = 0.3 # 1/(2pi * tau) = cutoff frequency
        ts = 0.02 # sample time        
        self.steer_lpf = LowPassFilter(tau, ts)
        
        self.decel_limit = decel_limit
        self.accel_limit = accel_limit
        self.wheel_radius = wheel_radius

        self.total_mass = vehicle_mass + fuel_capacity*GAS_DENSITY
        self.brake_deadband = brake_deadband

    def control(self, target_velocity, current_velocity):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        acceleration = self.throttle_controller.step(
            target_velocity.linear.x,
            current_velocity.linear.x,
            rospy.get_time())
      
        if acceleration > 0.0:
            # Positive
            throttle = max(0.0, min(self.accel_limit, acceleration))
            brake = 0.
        else:
            throttle = 0.
            brake = -1.0*max(self.decel_limit, acceleration)*self.total_mass

        # Steering
        steering = self.yaw_controller.get_steering(
            target_velocity.linear.x,
            target_velocity.angular.z,
            current_velocity.linear.x)
        steering = self.steer_lpf.filt(steering)

        return throttle, brake, steering
    
    def reset(self):
        self.throttle_controller.reset()
