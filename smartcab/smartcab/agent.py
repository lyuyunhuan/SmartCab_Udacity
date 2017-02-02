import random
import math
import operator
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
from collections import namedtuple
from math import exp


class LearningAgent(Agent):
    """ An agent that learns to drive in the Smartcab world.
        This is the object you will be modifying. """ 

    def __init__(self, env, learning=True, epsilon=1.0, alpha=0.5):
        super(LearningAgent, self).__init__(env)     # Set the agent in the evironment 
        self.planner = RoutePlanner(self.env, self)  # Create a route planner
        self.valid_actions = self.env.valid_actions  # The set of valid actions

        # Set parameters of the learning agent
        self.learning = learning # Whether the agent is expected to learn
        self.Q = dict()          # Create a Q-table which will be a dictionary of tuples
        ###St = namedtuple('State','direction receiving')
        ###self.Q = {St:None, {}}
        self.epsilon = epsilon   # Random exploration factor
        self.alpha = alpha       # Learning factor

        ###########
        ## TO DO ##
        ###########
        # Set any additional class parameters as needed
        self.env = env


    def reset(self, destination=None, testing=False):
        """ The reset function is called at the beginning of each trial.
            'testing' is set to True if testing trials are being used
            once training trials have completed. """

        # Select the destination as the new location to route to
        self.planner.route_to(destination)
        
        ########### 
        ## TO DO ##
        ###########
        # Update epsilon using a decay function of your choice
        self.epsilon = self.epsilon- 0.05
        
        self.alpha = self.alpha - 0.00007
        ##self.epsilon = exp(-0.0005 * (float(0.5-self.alpha)/0.00007))
        # Update additional class parameters as needed
        # If 'testing' is True, set epsilon and alpha to 0
        if testing == True:
            self.epsilon = 0

        return None

    def build_state(self):
        """ The build_state function is called when the agent requests data from the 
            environment. The next waypoint, the intersection inputs, and the deadline 
            are all features available to the agent. """

        # Collect data about the environment
        waypoint = self.planner.next_waypoint() # The next waypoint 
        inputs = self.env.sense(self)           # Visual input - intersection light and traffic
        deadline = self.env.get_deadline(self)  # Remaining deadline

        ########### 
        ## TO DO ##
        ###########
        # Set 'state' as a tuple of relevant data for the agent 
        # https://pymotw.com/2/collections/namedtuple.html
        
        ###St = namedtuple('State','direction receiving')
        ###nowState = St(direction = waypoint, receiving =inputs)#None
        u = ''.join(waypoint)
        v = ''.join(inputs)
        return (waypoint, inputs)
    
    def re_left(self, state5):
        a,b = state5
        u = ''.join(a)
        v = ''.join(b)
        s = u+v
        
        stats = self.Q[s]
        return stats['left']


    def get_maxQ(self, state4):
        """ The get_max_Q function is called when the agent is asked to find the
            maximum Q-value of all actions based on the 'state' the smartcab is in. """

        ########### 
        ## TO DO ##
        ###########
        # Calculate the maximum Q-value of all actions for a given state
        # reference link: http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        # reference link: http://yehnan.blogspot.tw/2015/06/pythonoperatoritemgetter.html
        
        a,b = state4
        u = ''.join(a)
        #v1 = ''.join(b['oncoming'])
        if b['oncoming'] == None:
            v1 = ''.join('None')
        else:
            v1 = ''.join(b['oncoming'])
        
        if b['left'] == None:
            v2 = ''.join('None')
        else:
            v2 = ''.join(b['left'])    
       
        if b['right'] == None:
            v3 = ''.join('None')
        else:
            v3 = ''.join(b['right']) 
        
        v4 = ''.join(b['light'])
        ###s = ''.join(state2)
        s = u+v1+v2+v3+v4
        
        stats = self.Q[s]
        inputst = self.env.sense(self)  
        
        #if stats[self.planner.next_waypoint()] > 0 - 5 - self.epsilon * 10 or (inputst['light'] == 'green' and inputst['oncoming'] != 'left'):
        ####if (inputst['light'] == 'green' and inputst['oncoming'] == None ):
        ####    maxQ = a
        ##elif (inputst['light'] == 'green' and inputst['oncoming'] != 'left' and self.planner.next_waypoint() == 'left' and stats[self.planner.next_waypoint()] > -3):
            #maxQ = self.planner.next_waypoint()
        ##    maxQ = 'left'
        ##elif (inputst['light'] == 'green' and inputst['oncoming'] != 'left' and self.planner.next_waypoint() != 'left' and stats[self.planner.next_waypoint()] > -3):
        ##    maxQ = self.planner.next_waypoint()
        if a == None:
            b = 'None'
        else:
            b = a
        
        if stats[b] >= -4.2:
            maxQ = a
        else:
            maxQ = max(stats.iteritems(), key=operator.itemgetter(1))[0]#None
            if maxQ == 'None':
                maxQ = None
        
        return maxQ 


    def createQ(self, state2):
        """ The createQ function is called when a state is generated by the agent. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, check if the 'state' is not in the Q-table
        # reference links: TypeError: unhashable type: 'dict' 
        # http://stackoverflow.com/questions/13264511/typeerror-unhashable-type-dict
        # http://stackoverflow.com/questions/4531941/typeerror-unhashable-type-dict-when-dict-used-as-a-key-for-another-dict?noredirect=1&lq=1
        # http://stackoverflow.com/questions/27435798/unhashable-type-dict-type-error
        # http://stackoverflow.com/questions/4878881/python-tuples-dictionaries-as-keys-select-sort
        
        ###namedState = namedtuple("namedState", "state1")
        ###f = namedState(state1 = state)
        ###dicc = self.Q
        
        ###exist = dicc.get(f,0)
        # http://stackoverflow.com/questions/4878881/python-tuples-dictionaries-as-keys-select-sort
        #print state.direction
        a,b = state2
        u = ''.join(a)
        if b['oncoming'] == None:
            v1 = ''.join('None')
        else:
            v1 = ''.join(b['oncoming'])
        
        if b['left'] == None:
            v2 = ''.join('None')
        else:
            v2 = ''.join(b['left'])    
       
        if b['right'] == None:
            v3 = ''.join('None')
        else:
            v3 = ''.join(b['right']) 
        v4 = ''.join(b['light'])
        
        ###s = ''.join(state2)
        s = u+v1+v2+v3+v4
        if s not in self.Q:
        ####    pass
        ####else:
            self.Q[s]={}
            self.Q[s]['right'] = 0
            self.Q[s]['left'] = 0
            self.Q[s]['forward'] = 0
            self.Q[s]['None'] = 0
        # If it is not, create a new dictionary for that state
        ##if not exsit:
        #   Then, for each action available, set the initial Q-value to 0.0
            ##self.Q[state]=dict()
            ##self.Q[state]['right'] = 0
            ##self.Q[state]['left'] = 0
            ##self.Q[state]['forward'] = 0
            ##self.Q[state][None] = 0

        return 


    def choose_action(self, state):
        """ The choose_action function is called when the agent is asked to choose
            which action to take, based on the 'state' the smartcab is in. """

        # Set the agent state and default action
        self.state = state
        self.next_waypoint = self.planner.next_waypoint()
        action = None

        ########### 
        ## TO DO ##
        ###########
        # When not learning, choose a random action
        # When learning, choose a random action with 'epsilon' probability
        #   Otherwise, choose an action with the highest Q-value for the current state
        
        if self.learning: 
            threshold = random.random()
            if threshold > self.epsilon:
                action = self.get_maxQ(state)
            else:
                action = random.choice(self.valid_actions)
        else:
            action = random.choice(self.valid_actions)
        return action


    def learn(self, state3, action, reward):
        """ The learn function is called after the agent completes an action and
            receives an award. This function does not consider future rewards 
            when conducting learning. """

        ########### 
        ## TO DO ##
        ###########
        # When learning, implement the value iteration update rule
        #   Use only the learning rate 'alpha' (do not use the discount factor 'gamma')
        a,b = state3
        u = ''.join(a)
        if b['oncoming'] == None:
            v1 = ''.join('None')
        else:
            v1 = ''.join(b['oncoming'])
        
        if b['left'] == None:
            v2 = ''.join('None')
        else:
            v2 = ''.join(b['left'])    
       
        if b['right'] == None:
            v3 = ''.join('None')
        else:
            v3 = ''.join(b['right']) 
        
        v4 = ''.join(b['light'])
        ###s = ''.join(state2)
        s = u+v1+v2+v3+v4
        
        if action == None:
            actionString = 'None'
        else:
            actionString = action
        
        self.Q[s][actionString] = self.alpha * reward + (1-self.alpha)*self.Q[s][actionString]
        return


    def update(self):
        """ The update function is called when a time step is completed in the 
            environment for a given trial. This function will build the agent
            state, choose an action, receive a reward, and learn if enabled. """

        state = self.build_state()          # Get current state
        self.createQ(state)                 # Create 'state' in Q-table
        action = self.choose_action(state)  # Choose an action
        reward = self.env.act(self, action) # Receive a reward
        self.learn(state, action, reward)   # Q-learn

        return
        

def run():
    """ Driving function for running the simulation. 
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """

    ##############
    # Create the environment
    # Flags:
    #   verbose     - set to True to display additional output from the simulation
    #   num_dummies - discrete number of dummy agents in the environment, default is 100
    #   grid_size   - discrete number of intersections (columns, rows), default is (8, 6)
    env = Environment()
    
    ##############
    # Create the driving agent
    # Flags:
    #   learning   - set to True to force the driving agent to use Q-learning
    #    * epsilon - continuous value for the exploration factor, default is 1
    #    * alpha   - continuous value for the learning rate, default is 0.5
    agent = env.create_agent(LearningAgent)
    
    ##############
    # Follow the driving agent
    # Flags:
    #   enforce_deadline - set to True to enforce a deadline metric
    env.set_primary_agent(agent)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    #   log_metrics  - set to True to log trial and simulation results to /logs
    #   optimized    - set to True to change the default log file name
    sim = Simulator(env)
    
    ##############
    # Run the simulator
    # Flags:
    #   tolerance  - epsilon tolerance before beginning testing, default is 0.05 
    #   n_test     - discrete number of testing trials to perform, default is 0
    sim.run()


if __name__ == '__main__':
    run()
