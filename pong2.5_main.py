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
    agent1_name="PTRON"
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
    # init_W = [[ 4.70914779e+00], [ 2.91690586e-01], [ 3.32795718e-02], [-2.68967264e+00], [-3.14904203e-03], [-2.25043979e-02], [-2.06050065e+00]]
    # init_W = [[ 4.70914779e+00], [ 2.91690586e-01], [ 3.32795718e-02], [-2.68967264e+00], [-3.14904203e-03], [-2.25043979e-02]] #remove other paddle data
    # init_W= [[ 4.70881019e+00], [ 2.91697553e-01], [ 3.24488318e-02], [-2.69011196e+00], [-1.86365111e-03], [-2.26418204e-02]]
    # init_W=[[ 4.66491521], [ 0.29179559], [-0.0480732 ], [-2.73067094], [-0.04328703], [-0.06304903]]
    # init_W= [[ 4.28460927e+00],[2.92877436e-01], [-8.69682698e-01], [-3.10432564e+00], [-1.99989792e-03], [ 1.84148194e-03]]
    # init_W= [[0.71644732], [0.11904916], [0.8088049 ], [0.64646261], [0.62136034], [0.26677858]]
    # init_W=[[0.9776436 ], [0.11983563], [0.61094511], [0.10657396], [0.47538274], [0.19739199]]
    # init_W=[[ 0.96486658], [ 0.12046168], [ 0.46447226], [-0.31927739], [ 0.32199904], [ 0.14054958]]         
    # init_W= [[ 2.12598699],[ 0.11155833], [ 0.14956249], [-1.36861349], [ 0.31264509],  [ 0.01066866]]
    # init_W= [[ 4.70672714], [ 0.07455206], [-0.05865142], [-4.62479923], [ 0.08418891], [-0.36843274]]
    # init_W = [[ 5.24021546], [ 0.06149613], [-0.12616879], [-5.06013447], [ 0.05878421], [-0.34087985]]
    init_W = [[ 5.27887366], [ 0.06420452], [-0.07716966], [-5.2112937 ], [ 0.03195959], [-0.28047038]]
    ptron=perceptron.Perceptron(init_W)
    tround =0
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
        
        
        if dataReady:
            if train:
                print("PTRON!")
                print("roundset size ", len(trainset))
                print("results size", len(results))
                ptron.train(trainset,results, 101, .001)
                train=False
                # trainset=[]
                # results =[]

            x=ptron.results(normalize1(obs)[:-1])
            if np.rint(x)==0:
                action1 = "down"
            else:
                action1 = "up"
            
            
        else:
            action1 = agent1(normalize1(obs),weightArray1)
        
        action2 = agent2(normalize2(obs), weightArray1)
       
        roundOverFlag =0
        curdir=0
        if normalize2(obs)[4]<0: #based on x direction
            curdir="right"
        else:
            curdir="left"
        if lastDir == "left" and curdir =="right":
            roundOverFlag = 1#win
            roundY = normalize1(obs)[3]
        if lastDir == "right" and curdir =="left":
            roundOverFlag = 3#win
            roundY = normalize1(obs)[3]
            # framecount =0 set to zero after
        
        lastDir=curdir
        roundset.append( normalize1(obs)[:-1])#saves everything except other paddle data
        # print(normalize1(obs)[:-1])
        # print(normalize1(obs))

        
        

        reward = p.act(ACTION_MAP[(action1, action2)])
        

        if reward > 0:
            roundOverFlag =1 #win or ball hits
            roundY =normalize1(obs)[3]
            # agent1_score += 1
            print(agent1_name, 'win scored')
            
            # weightArray1=weightAdjust(normalize2(obs), weightArray1)
        elif reward < 0:
            roundOverFlag =1 #win or ball hits
            roundY =normalize1(obs)[3]
            agent2_score += 1
            # print(agent2_name, 'Loss scored')
            
            #put the results of round here
            # strat: stay ahead of ball learn a distance to stay ahead on 
            # weightArray1=weightAdjust(normalize2(obs), weightArray1)
        
        # we win a "round" as ball round trips
        if roundOverFlag>0: #record everything up to ball hits paddel(win), score (win), or miss (loss)
            tround +=1
            # if roundOverFlag==1: #won round
            #     for i in range(0, len(roundset)):
            #         results.append(1)
            # else: #lost round
            #     for i in range(0, len(roundset)):
            #         results.append(0)
            # print("round Data")
            for i in range(len(roundset)):
                if roundset[i][0]<roundY: #if paddel is above where paddle should be 0 else 1
                    results.append(0)
                    res=0
                else:
                    results.append(1)
                    res=1
                # print("place to go:,",roundY," currentPad: ",  roundset[i][0], "currBall: ",roundset[i][3],  "result", res)


            roundOverFlag = 0
            if(len(roundset)>=300):
                trainset.extend(roundset[-2000:]) #append the roundset to our total train data
            else:
                trainset.extend(roundset[-300:]) 
            if tround%50==0:
                print("tround ", tround)
            if train == False:
                if tround %500==0:
                    train=True
                    tround =0
            if tround==1:
                dataReady=True
                # train=False
                # tround =0
            
            #can then pull each rounset to weight appropriately in weight method
            # print("results", results)
            # print("train", trainset)
            roundset=[]

        
        

        
        
          
        
        
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
