import numpy

class CartPole:
    """Cart Pole environment. This implementation alows multiple poles,
    noisy action, and random starts. It has been checked repeatedly for
    'correctness', specifically the direction of gravity. Some implementations of
    cart pole on the internet have the gravity constant inverted. The way to check is to
    limit the force to be zero, start from a valid random start state and watch how long
    it takes for the pole to fall. If the pole falls almost immediately, you're all set. If it takes
    tens or hundreds of steps then you have gravity inverted. It will tend to still fall because
    of round off errors that cause the oscillations to grow until it eventually falls.
    """
    name = "Cart Pole"

    def __init__(self, mode='easy', pole_scales=[1.], noise=0.0, reward_noise=0.0, random_start=True):
        self.noise = noise
        self.reward_noise = reward_noise
        self.random_start = random_start
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle = numpy.zeros((len(pole_scales),))
        self.pole_velocity = numpy.zeros((len(pole_scales),))

        # Setup pole lengths and masses based on scale of each pole
        # (Papers using multi-poles tend to have them either same lengths/masses
        #   or they vary by some scalar from the other poles)
        pole_scales = numpy.array(pole_scales)
        self.pole_length = numpy.ones((len(pole_scales),))*0.5 * pole_scales
        self.pole_mass = numpy.ones((len(pole_scales),))*0.1 * pole_scales

        self.domain_name = "Cart Pole"

        self.mode = mode
        if mode == 'hard':
            self.state_range = numpy.array([[-3., 3.],                                   # Cart location bound
                                            [-5., 5.],                                    # Cart velocity bound
                                            [-numpy.pi * 45./180., numpy.pi * 45./180.], # Pole angle bounds
                                            [-2.5*numpy.pi, 2.5*numpy.pi]])              # Pole velocity bound
            self.mu_c = 0.0005
            self.mu_p = 0.000002
            self.sim_steps = 10
            self.discount_factor = 0.999
        elif mode == 'swingup':
            self.state_range = numpy.array([[-3., 3.],                                   # Cart location bound
                                            [-5., 5.],                                   # Cart velocity bound
                                            [-numpy.pi, numpy.pi],                       # Pole angle bounds
                                            [-2.5*numpy.pi, 2.5*numpy.pi]])              # Pole velocity bound
            self.mu_c = 0.0005
            self.mu_p = 0.000002
            self.sim_steps = 10
            self.discount_factor = 1.
        else:
            if mode != 'easy':
                print "Error: CartPole does not recognize mode", mode
                print "Defaulting to 'easy'"
            self.state_range = numpy.array([[-2.4, 2.4],                                 # Cart location bound
                                            [-6., 6.],                                   # Cart velocity bound
                                            [-numpy.pi * 12./180., numpy.pi * 12./180.], # Pole angle bounds
                                            [-6., 6.]])                                  # Pole velocity bound
            self.mu_c = 0.
            self.mu_p = 0.
            self.sim_steps = 1
            self.discount_factor = 0.999

        self.reward_range = (-1000., 1.*len(pole_scales)) if self.mode == "swingup" else (-1., 1.)
        self.delta_time = 0.02
        self.max_force = 10.
        self.gravity = -9.8
        self.cart_mass = 1.

    def initialState(self):
        self.cart_location = 0.0
        self.cart_velocity = 0.0
        self.pole_angle.fill(0.0)
        self.pole_velocity.fill(0.0)
        if self.random_start:
            self.pole_angle = (numpy.random.random(self.pole_angle.shape)-0.5)/5.
        
    def getState(self):   
        s_now = numpy.array([self.cart_location, self.cart_velocity, self.pole_angle, self.pole_velocity])
        return s_now, self.terminate()
        
    def oneStep(self, intAction):
         """
          Returns the current situation.
          A situation can be the current perceptual inputs, a random problem instance ...
         """
         s, reward, terminal = self.takeAction(intAction)
         return s, reward, terminal

    def __gravity_on_pole(self):
        pull = self.mu_p * self.pole_velocity/(self.pole_mass * self.pole_length)
        pull += self.gravity * numpy.sin(self.pole_angle)
        return pull

    def __effective_force(self):
        F = self.pole_mass * self.pole_length * self.pole_velocity**2 * numpy.sin(self.pole_angle)
        F += .75 * self.pole_mass * numpy.cos(self.pole_angle) * self.__gravity_on_pole()
        return F.sum()

    def __effective_mass(self):
        return (self.pole_mass * (1. - .75 * numpy.cos(self.pole_angle)**2)).sum()

    def takeAction(self, intAction):
        #print ' ACTION  %d ' % (intAction)
        if intAction[0] == 1:
            force = self.max_force
        elif intAction[1] == 1:
            force = -self.max_force
        else:
            force = 0.0
            
        force += self.max_force*numpy.random.normal(scale=self.noise) if self.noise > 0 else 0.0 # Compute noise

        for step in range(self.sim_steps):
            cart_accel = force - self.mu_c * numpy.sign(self.cart_velocity) + self.__effective_force()
            cart_accel /= self.cart_mass + self.__effective_mass()
            pole_accel = (-.75/self.pole_length) * (cart_accel * numpy.cos(self.pole_angle) + self.__gravity_on_pole())

            # Update state variables
            df = (self.delta_time / float(self.sim_steps))
            self.cart_location += df * self.cart_velocity
            self.cart_velocity += df * cart_accel
            self.pole_angle += df * self.pole_velocity
            self.pole_velocity += df * pole_accel

        # If theta (state[2]) has gone past our conceptual limits of [-pi,pi]
        # map it onto the equivalent angle that is in the accepted range (by adding or subtracting 2pi)
        for i in range(len(self.pole_angle)):
            while self.pole_angle[i] < -numpy.pi:
                self.pole_angle[i] += 2. * numpy.pi

            while self.pole_angle[i] > numpy.pi:
                self.pole_angle[i] -= 2. * numpy.pi

        s = numpy.array([self.cart_location, self.cart_velocity, self.pole_angle, self.pole_velocity])
        
        if self.mode == 'swingup':
            return numpy.cos(numpy.abs(self.pole_angle)).sum()
        else:
            if self.terminate():
                return s, -1., self.terminate()            
            else:
                return s, 1., self.terminate()

    def terminate(self):
        """Indicates whether or not the episode should terminate.

        Returns:
            A boolean, true indicating the end of an episode and false indicating the episode should continue.
            False is returned if either the cart location or
            the pole angle is beyond the allowed range.
        """
        return numpy.abs(self.cart_location) > self.state_range[0,1] or (numpy.abs(self.pole_angle) > self.state_range[2,1]).any()

