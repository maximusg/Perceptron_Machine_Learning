import sys
from math import exp
from ple import PLE
import numpy as np
import random
import perceptron

from pong2player import Pong2Player



WIDTH = 200
HEIGHT = 300
FPS = 600
NUM_INPUTS = 7
NUM_ROUNDS = 50000 #episodes
MAX_SCORE = 10#batch size

PLAYER_SPEED = 1
BALL_SPEED = 5

ACTION_MAP = {
    (None, None): 0,
    ('up', None): 1,
    ('down', None): 2,

    (None, 'up'): 3,
    ('up', 'up'): 4,
    ('down', 'up'): 5,

    (None, 'down'): 6,
    ('up', 'down'): 7,
    ('down', 'down'): 8,
}
    
def agent(x,w): #trained on all vars to find right answer by the ball's y location
   
    x=np.dot(x, w)
   
    x= (1 / (1 + np.exp(-x))) #sigmoid function returns 0 or 1
    if np.rint(x)==0:
        return "down"
    else:
        return "up"

def random(x,w):
    '''Chooses actions randomly'''
    import random as rnd
    return rnd.choice(['up','down'])


def defense(x,w):
    '''Follows the ball'''
    if x[0] > x[3]:
        return 'up'
    else:
        return 'down'


def normalize1(obs):
    # Helper function to convert game state observations to a normalized input vector
    x = [
        obs['player1_y']/HEIGHT,
        obs['player1_velocity']/HEIGHT,
        obs['ball_x']/WIDTH,
        obs['ball_y']/HEIGHT,
        obs['ball_velocity_x']/WIDTH,
        obs['ball_velocity_y']/HEIGHT,
        obs['player2_y']/HEIGHT,
    ]
    return x


def normalize2(obs):
    # Helper function to convert game state observations to a normalized input vector
    x = [
        obs['player2_y']/HEIGHT,
        obs['player2_velocity']/HEIGHT,
        (WIDTH - obs['ball_x'])/WIDTH,
        obs['ball_y']/HEIGHT,
        -obs['ball_velocity_x']/WIDTH,
        obs['ball_velocity_y']/HEIGHT,
        obs['player1_y']/HEIGHT,
    ]
    return x


def main(agent1_name, agent2_name, train=False):
    agent1 = eval(agent1_name)
    agent2 = eval(agent2_name)

    agent1_name = agent1_name.upper()
    agent2_name = agent2_name.upper()

    # Don't need to modify anything in this function. See the constants defined
    # at the top of this file.
    game = Pong2Player(width=WIDTH, height=HEIGHT, MAX_SCORE=MAX_SCORE, ball_speed_ratio=BALL_SPEED,
                       player1_speed_ratio=PLAYER_SPEED, player2_speed_ratio=PLAYER_SPEED,
                       player1_name=agent1_name, player2_name=agent2_name)

    if train:
        p = PLE(game, fps=FPS, display_screen=False, force_fps=True)
    else:
        p = PLE(game, fps=FPS, display_screen=True, force_fps=False)

    p.init()

    agent1_rounds = 0
    agent2_rounds = 0
    agent1_score = 0
    agent2_score = 0
    num_frames = 0
    #new stuff
    lastDir = "right"
    roundset = []         #rounds are the frames which to be cycle through to implement strategy and populate results with
    trainset = []         #rounds get appended here after number of training_Rounds
    results= []           #results of a round, given the startegy for moving the ball
    training_Rounds =500 #number of rounds we want to train perceptron on
    train = False
     #initialize perceptron! using the weights below. 
    init_W = [[ 4.70914779e+00], [ 2.91690586e-01], [ 3.32795718e-02], [-2.68967264e+00], [-3.14904203e-03], [-2.25043979e-02], [-2.06050065e+00]]
    # Can also initialize with a number, but that number must match the number of inputs that will be put in for training
    ptron=perceptron.Perceptron(init_W) #if you were to give it a 7 in initialize 7 random weights
    init_W=ptron.train() #spits out the weights when no input is given. This is uncessary here since we already have weights, but if you wanted to initialize weights randomly first this is the way to do it
    tround =0
    roundY = 0
    roundFirstFrame = True
    debug = False
    while True:
        if p.game_over():
            if game.score_counts['player1'] > game.score_counts['player2']:
                agent1_rounds += 1
                # print(agent1_name, 'won round')
            else:
                agent2_rounds += 1
                # print(agent2_name, 'won round') #don't care of other side wins until perceptron gets good

            if agent1_rounds == NUM_ROUNDS or agent2_rounds == NUM_ROUNDS:
                break

            p.reset_game()

        obs = p.getGameState()
        
        #sets lastDir to the direction of the ball in the first frame of each round
        if roundFirstFrame:
            roundFirstFrame =False
            if normalize2(obs)[4]<0: #based on x direction going right
                lastDir="right"
            else:
                lastDir="left"

        if train: #trains perceptron
            print("PTRON!")
            print("trainset size ", len(trainset)) #just sanity check to ensure trainset == results set, if not there is an issue in trainset
            print("results size", len(results))
            init_W=ptron.train(trainset,results, 101, .001) #trains
            train=False
            #can clean the data and results, or if you don't clear them, this is another strategy but it makes for long training times 
            trainset=[]
            results =[]

            
            
        
        action1 = agent1(normalize1(obs), init_W) #modified these calls to take weights in so it can imitate the new training weight data
        
        action2 = agent2(normalize2(obs), init_W)
       
        #TRAINING STRATEGY
        #this is a check to see which direction the ball is traveling. AND sets roundover>0 if it hits either paddle
        #The strategy is to train to when the ball hits the opponents paddle on the left paddle. LEFT specifically.

        roundOverFlag =0
        curdir=0
        if normalize2(obs)[4]<0: #based on x direction going right
            curdir="right"
        else:
            curdir="left"

        if lastDir == "left" and curdir =="right":
            roundOverFlag = 1 #ball hits players paddle
            roundY = normalize1(obs)[3] #record  ball location
        if lastDir == "right" and curdir =="left":
            pass
            #do nothing here
            # roundOverFlag = 3 #ball hits opponents paddle
            # roundY = normalize1(obs)[3] #record ball location
           
        
        lastDir=curdir #change the last direction to the current direction
        roundset.append( normalize1(obs))#saves all the round data into "roundset" array
     

        reward = p.act(ACTION_MAP[(action1, action2)])
        

        if reward > 0:
            roundOverFlag =1 #win or ball hits
            roundY =normalize1(obs)[3]
            roundFirstFrame=True #LasDir gets reset every time there is a win
            agent1_score += 1
            print(agent1_name, 'win scored')

        elif reward < 0:
            roundOverFlag =1 #win or ball hits
            roundY =normalize1(obs)[3]
            roundFirstFrame=True #LasDir gets reset every time there is a win
            agent2_score += 1
            lastDir ="right" #needst be reset every time there is a win
            # print(agent2_name, 'Loss scored')
        
        # we say a round is over when ball hits players paddle or left player scores a win
        if roundOverFlag>0: #record everything up to ball hits paddel(win), or miss (loss) of left
            tround +=1
            for i in range(len(roundset)):
                if roundset[i][0]<roundY: #if paddel is above where paddle should be 0 else 1
                    results.append(0)
                    res=0
                else:
                    results.append(1)
                    res=1
                if debug:
                    print("place to go:,",roundY," currentPad: ",  roundset[i][0], "currBall: ",roundset[i][3],  "result", res)


            roundOverFlag = 0
         
            trainset.extend(roundset) #append the roundset to our total train data
            roundset=[] #clear the roundset info
         
            if tround%50==0: #just prints every 50 rounds so we know how many rounds  since the last training 
                print("tround ", tround)
           
            if tround %training_Rounds==0: #when we hit the number of training_Rounds we train on the data
                train=True
                tround =0
        
        num_frames += 1

    winner = agent1_name if agent1_rounds > agent2_rounds else agent2_name
    print()
    print('\t\t%s\t%s' % (agent1_name, agent2_name))
    print('Rounds won :\t%d\t%d' %(agent1_rounds, agent2_rounds))
    print('Total score:\t%d\t%d' %(agent1_score, agent2_score))
    print()
    print('Winner: %s (after %d frames)' %(winner, num_frames))

    #learn on trainset w vals
    #print w vals
    # print(trainset)
    # train_weights(trainset,.0005,20)

if __name__ == '__main__':
    agent1_name = sys.argv[1]
    agent2_name = sys.argv[2]

    main(agent1_name, agent2_name, train=False)
