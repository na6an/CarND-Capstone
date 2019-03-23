
MIN_NUM = float('-inf')
MAX_NUM = float('inf')


class PID(object):
    def __init__(self, kp, ki, kd, mn=MIN_NUM, mx=MAX_NUM):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.min = mn
        self.max = mx

        # Controller state
        self.t = None
        self.last_error = 0.0
        self.integral = 0.0        

    def reset(self):
        self.integral = 0.0

    def step(self, target, current, t):
        # PID controller requires delta t and derivative of error. 
        # i.e. at least one timestep must have elapsed.
        if self.t == None:
            self.t = t            
            return 0.0

        delta_t = t - self.t

        # calculate error
        error = target - current    
        
        integral = self.integral + error * delta_t        
        integral = max(MIN_NUM, min(MAX_NUM, integral)) # do clamping
        derivative = (error - self.last_error) / delta_t
        
        control = self.kp * error + self.ki * integral + self.kd * derivative
        control = max(self.min, min(self.max, control)) # do clamping
        
        self.integral = integral
        self.last_error = error

        return control
