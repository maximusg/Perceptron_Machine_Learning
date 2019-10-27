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
MAX_SCORE = 20#batch size

PLAYER_SPEED = 1
BALL_SPEED = 3

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

def weightAdjust(x,w):
    #record lost

#     obs['player1_y']/HEIGHT,
#     obs['player1_velocity']/HEIGHT,
#     obs['ball_x']/WIDTH,
#     obs['ball_y']/HEIGHT,
#     obs['ball_velocity_x']/WIDTH,
#     obs['ball_velocity_y']/HEIGHT,
#     obs['player2_y']/HEIGHT,
    w[5]+=1
    w[0] = w[0]+(x[0] - x[3])/w[5]
    # w[3] += w[0]
    # w[0] = w[3]/w[2] #average out losses
    # #find gap
    if x[0]<.3:
        w[2] +=1
        w[4] = abs(x[0])/w[2]
        print("avg loss dist:", w[4])
    elif x[0]>.7:
        w[2] +=1
        w[4] = 1-abs(x[0])/w[2]
        print("avg loss dist:", w[4])

    print(w[0])
    print("lost on:",x)
    return w

# def expand(x):
#     ary=[]
#     for i in range(0,len(x)):
#         for j in range(0, len(x)):
#             new = x[i]*x[j]
#             ary.append(new)
    
#     return ary
# def expandS(x,s):
#     for i in range(0,s):
#         x=expand(x)
#     return x

def tanh(prediction):
    pass

def softmax(prediction_tanh):
    pass

# Make a prediction with weights
def predict(row, weights):
    # row = expandS(row,2)
    activation = weights[0]
    # print("row length ",len(row))
    # print("weights length", len(weights))
    for i in range(len(row)-1):
        activation += weights[i + 1] * row[i]
    return 1.0 if activation >= 0.0 else 0.0

# get started with new weights
def train_weights(t,l_rate,n_epoch):#t is the train_set
    # tsize= len(expandS(t[0][0],2))
    w = [random.random() for i in range(tsize)] # w is weights initializdd to zero
    for epoch in range(n_epoch):
        sum_error = 0.0
        for subset in t: 
            size = len(subset)
            i=0
            for row in subset: #need to weight heavier for each subsequent row
                i=+1
                # row = expandS(row,2)
                prediction = predict(row,w)
                error = (row[-1] - prediction)
                # sum_error += error**2
                w[0] = w[0] + l_rate * error
                for i in range(len(row)-1):
                    w[i + 1] = w[i + 1] + l_rate * error * row[i]
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    # print(w)
    return w

def train_weights2(t,l_rate,n_epoch,w):#t is the train_set
    # w = [0.0 for i in range(len(t[0][0]))] # w is weights initializdd to zero
    for epoch in range(n_epoch):
        sum_error = 0.0
        for subset in t: 
            size = len(subset)
            i=0
            for row in subset: #need to weight heavier for each subsequent row
                i=+1
                # row = expandS(row,2)
                prediction = predict(row,w)
                error = (row[-1] - prediction)*(i/size)
                sum_error += error**2
                w[0] = w[0] + l_rate * error
                for i in range(len(row)-1):
                    w[i + 1] = w[i + 1] + l_rate * error * row[i]
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
    # print(w)
    return w

def ptron2(x):
    
    if np.rint(x)==0:
        return "down"
    else:
        return "up"


def defense2(x,w):
 

    # x.extend([1])
    x = expandS(x,2)
    n=predict(x,w)
    # print(n)
    if n>0:
        return "up"
    else:
        return "down"
    
   

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
    agent1_name="PTRON2"
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
    weightArray1 =[]
    roundset = []
    lastDir = "right"
    framecount = 0
    trainset = []
    results= []
    dataReady =True
    train = False
    #updown behavior [[ 0.93351164] [ 0.24549141] [ 0.79765264] [ 0.0525945 ] [ 0.32094939] [ 0.09081188] [-0.06761758]]
    # init_W = [[ 4.70786639], [ 0.29168877], [ 0.03238542], [-2.69096479], [-0.00693419], [-0.02249474], [-2.061787  ]]
    init_W = [[ 4.70914779e+00], [ 2.91690586e-01], [ 3.32795718e-02], [-2.68967264e+00], [-3.14904203e-03], [-2.25043979e-02], [-2.06050065e+00]]
    # init_W = [[ 4.70914779e+00], [ 2.91690586e-01], [ 3.32795718e-02], [-2.68967264e+00], [-3.14904203e-03], [-2.25043979e-02]] #remove other paddle data
    # init_W= [[ 4.70881019e+00], [ 2.91697553e-01], [ 3.24488318e-02], [-2.69011196e+00], [-1.86365111e-03], [-2.26418204e-02]]
    # init_W=[[ 4.66491521], [ 0.29179559], [-0.0480732 ], [-2.73067094], [-0.04328703], [-0.06304903]]
    # init_W= [[ 4.28460927e+00],[2.92877436e-01], [-8.69682698e-01], [-3.10432564e+00], [-1.99989792e-03], [ 1.84148194e-03]]
    # init_W=  [[ 4.92518488], [ 0.29318287],[-0.19209306], [-2.45027753], [-0.01812887], [ 0.01983129]]
    # init_W= [[ 4.96089562], [ 0.29324457], [-0.16887879], [-2.41088307], [-0.04067242], [-0.00876377]]
    # init_W= [[ 0.70705757], [ 0.42945529], [-0.03871037], [ 0.73993365], [ 0.69206227], [ 0.45317198]]
    # init_W= [[ 6.56687267], [ 0.80782401], [-0.0552708 ], [-6.40478168], [ 0.26838884],  [ 0.23331978]]
    # init_W= [[ 6.90326304], [ 0.78392231], [-0.09115288], [-6.80591162], [-0.20238204], [ 0.22673383]]
    # init_W= [[ 6.96210782], [ 0.77984277], [-0.03381056], [-7.06139981], [ 0.03490671], [-0.11488688]]
    # init_W =[[ 7.00540107], [ 0.78108438], [-0.0452162 ], [-7.04205366], [-0.11168782], [ 0.04564088]]
    # init_W= [[ 7.03621517],       [ 0.7808232 ],       [-0.1252082 ],       [-7.15978461], [-0.08084997], [ 0.18158729]]
    # init_W= [[ 7.04402442], [ 0.78071559], [-0.10995117], [-7.15692231], [ 0.01548366], [ 0.18276923]]
    # init_W =[[ 7.12344095e+00],[ 7.79146035e-01], [-1.62993885e-02], [-7.08983235e+00], [-2.71428817e-04], [-7.80383266e-02]]
    ptron=perceptron.Perceptron(init_W)
    p1round =1
    p2round =1
    roundY = 0
    while True:
        if p.game_over():
            if game.score_counts['player1'] > game.score_counts['player2']:
                agent1_rounds += 1
         
                print(agent1_name, 'won round')
            else:
                agent2_rounds += 1
               
                # print(agent2_name, 'won round')

            if agent1_rounds == NUM_ROUNDS or agent2_rounds == NUM_ROUNDS:
                break

            p.reset_game()

        obs = p.getGameState()
        
        
        # if dataReady:
        #     if train:
        #         print("PTRON!")
        #         print("roundset size ", len(roundset))
        #         print("results size", len(results))
        #         ptron.train(roundset,results, 101, .1)
        #         train=False
        #         # trainset=[]
        #         roundset =[]
        #         results = []
            
        #     x=ptron.results(normalize1(obs)[:-1])
        #     if np.rint(x)==0:
        #         action1 = "down"
        #     else:
        #         action1 = "up"
            
            
        # else:
        #     action1 = agent1(normalize1(obs),weightArray1)
        
        # action1 = agent1(normalize1(obs),weightArray1)
        action2 = agent2(normalize2(obs), weightArray1)
        action1 = ptron2(ptron.results(normalize1(obs)))
        # action2 = ptron2(ptron.results(normalize2(obs)[:-1]))
       
        # roundOverFlag =0
        # curdir=0
        # if normalize2(obs)[4]<0: #based on x direction
        #     curdir="right"
        # else:
        #     curdir="left"
        # if lastDir == "left" and curdir =="right":
        #     roundOverFlag = 1#win
        #     roundY = normalize1(obs)[3]
        # if lastDir == "right" and curdir =="left":
        #     roundOverFlag = 1#win
        #     roundY = normalize1(obs)[3]
        #     # framecount =0 set to zero after
        
        # lastDir=curdir
        
          # we win a "round" as ball round trips
        # if roundOverFlag>0: #record everything up to ball hits paddel(win), score (win), or miss (loss)
        #     tround +=1
            # if roundOverFlag==1: #won round
            #     for i in range(0, len(roundset)):
            #         results.append(1)
            # else: #lost round
            #     for i in range(0, len(roundset)):
            #         results.append(0)
            # print("round Data")
            # for i in range(len(roundset)):
            #     if roundset[i][0]<roundY: #if paddel is above where paddle should be 0 else 1
            #         results.append(0)
            #         res=0
            #     else:
            #         results.append(1)
                    # res=1

        # if agent1_rounds + agent2_rounds %2==0: # train every 2 rounds
        #         dataReady=True
        #         train=True
            #     print("tround ", tround)
            # if train == False:
            #     if tround %50==0:
            #         train=True
            #         tround =0
            # if tround==1:
            #     dataReady=True
                # train=False
                # tround =0
        # print(normalize1(obs)[:-1])
        # print(normalize1(obs))

        
        

        reward = p.act(ACTION_MAP[(action1, action2)])
        

        if reward > 0:
            agent1_score += 1
            p1round +=1
            print(agent1_name, 'win scored')
            
        elif reward < 0:
            agent2_score += 1
            p2round +=1
            
        
        roundset.append(normalize1(obs))#saves everything except other paddle data
        if normalize1(obs)[0]>normalize1(obs)[3]:
            results.append(1)
        else:
            results.append(0)

        framecount+=1
        if framecount%7000==0:
            print("in")
            dataReady=True
            train=True
            framecount=0
        

        
        if train:
            print("PTRON!")
            print("roundset size ", len(roundset))
            print("results size", len(results))
            ptron.train(roundset,results, 101, .001)
            train=False
        

        
        
          
        
        
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
