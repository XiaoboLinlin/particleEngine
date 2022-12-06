
import numpy as np

class Particle:
    """Particle class

    Properties include position and velocities. Methods include an
    advance function as well as convenience function to calculate
    momentum and energies.
    """
    g = np.array( [ 0.0, -9.8 ] )
    
    def __init__(self, state = [0.0, 0.0, 0.0, -1.0], size = 0.05, M = 0.05):
        self.state = state
        self.size = size
        self.mass = M

    def advance(self, dt):
        self.state[:2] += dt*self.state[2:] + 0.5*Particle.g*dt*dt
        self.state[2:] += Particle.g*dt

    def getMomentum( self, dir ):
        if dir==0:
            return (1-dir)*self.mass*np.sqrt((self.state[2])**2)
        if dir==1:
            return (dir)*self.mass*np.sqrt((self.state[3])**2)

    def getKE( self ):
        return 1/2 * self.mass * ((self.state[2])**2 + (self.state[3])**2)

    def getPE( self, ref ):
        return self.mass*Particle.g*(self.state[1]-ref[2])
