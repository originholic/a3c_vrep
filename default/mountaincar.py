import numpy

class MountainCarND(object):
    """A generalized Mountain Car domain, which allows N-dimensional
    movement. When dimension=2 this behaves exactly as the classical
    Mountain Car domain. For dimension=3 it behaves as given in the
    paper:

    Autonomous Transfer for Reinforcement Learning. 2008.
    Matthew Taylor, Gregory Kuhlmann, and Peter Stone.

    However, this class also allows for even greater dimensions.
    """
    name = "3D Mountain Car"
    def __init__(self, noise=0.0, random_start=False, dim=2):
        self.noise = noise
        self.reward_noise = 0.0
        self.random_start = random_start
        self.state = numpy.zeros((dim-1,2))
        self.state_range = numpy.array([[[-1.2, 0.6], [-0.07, 0.07]] for i in range(dim-1)])
        self.goalPos = 0.5
        self.acc = 0.001
        self.gravity = -0.0025
        self.hillFreq = 3.0
        self.delta_time = 1.0

    def _reset(self):
        if self.random_start:
            self.state = numpy.random.random(self.state.shape)
            self.state *= (self.state_range[:,:,1] - self.state_range[:,:,0])
            self.state += self.state_range[:,:,0]
        else:
            self.state = numpy.zeros(self.state.shape)
            self.state[:,0] = -0.5

    def _state(self):
        return self.state.flatten().tolist(), self.isAtGoal

    def isAtGoal(self):
        return (self.state[:,0] >= self.goalPos).all()

    def takeAction(self, intAction):
        # Translate action into a (possibly) multi-dimensional direction
        intAction -= 1
        direction = numpy.zeros((self.state.shape[0],)) # Zero is Neutral
        if intAction >= 0:
            direction[int(intAction)/2] = ((intAction % 2) - 0.5)*2.0
        if self.noise > 0:
            direction += self.acc * numpy.random.normal(scale=self.noise, size=direction.shape)

        self.state[:,1] += self.acc*(direction) + self.gravity*numpy.cos(self.hillFreq*self.state[:,0])
        self.state[:,1] = self.state[:,1].clip(min=self.state_range[:,1,0], max=self.state_range[:,1,1])
        self.state[:,0] += self.delta_time * self.state[:,1]
        self.state[:,0] = self.state[:,0].clip(min=self.state_range[:,0,0], max=self.state_range[:,0,1])

    def _step(self,intAction):
        terminal = 0
        theReward = -1.0

        self.takeAction(intAction)

        if self.isAtGoal():
            theReward = 0.0
            terminal = 1

        if self.reward_noise > 0:
            theReward += numpy.random.normal(scale=self.reward_noise)

        return self.state.flatten().tolist(), theReward, terminal

