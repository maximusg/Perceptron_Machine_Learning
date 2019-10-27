import random
from math import exp
from ple.games.pong import Pong
from ple import PLE
import numpy as np


# Scren size
WIDTH = 200
HEIGHT = 300

# Frames per second
FPS = 120

# Total number of features
NUM_INPUTS = 7

# Game length = num
NUM_ROUNDS = 3
MAX_SCORE = 5

ACTION_MAP = {
    'up': 119,
    'down': 115
}

def normalize(obs):
    # Helper function to convert game state observations to a normalized input vector
    x = [
        obs['player_y']/HEIGHT,
        obs['player_velocity']/HEIGHT,
        obs['ball_x']/WIDTH,
        obs['ball_y']/HEIGHT,
        obs['ball_velocity_x']/WIDTH,
        obs['ball_velocity_y']/HEIGHT,
        obs['cpu_y']/HEIGHT,
    ]
    return x


def agent(x): #trained to find right answer to balls location when it hits the paddle
    w = [[ 5.28024833e+00],
       [ 6.42841740e-02],
       [-7.56373401e-02],
       [-5.21343692e+00],
       [ 1.30026530e-04],
       [-2.79735971e-01]]
    x=np.dot(x[:-1], w)
    x = np.clip( x, -500, 500 ) #prevents overflow
    x= (1 / (1 + np.exp(-x)))
    if np.rint(x)==0:
        return "down"
    else:
        return "up"

    # Your agent code here.
    return random.choice(['up','down'])

def agent2(x): #trained to find right answer by the ball's y location got rid of oponents paddle
    w = [[ 7.11331081e+00],
       [ 7.79982320e-01],
       [-1.88764661e-02],
       [-7.10216601e+00],
       [-6.75427375e-04],
       [ 3.45131891e-03]]
    x=np.dot(x[:-1], w)
    x = np.clip( x, -500, 500 ) #prevents overflow
    x= (1 / (1 + np.exp(-x)))
    if np.rint(x)==0:
        return "down"
    else:
        return "up"
def agent3(x): #trained on all vars to find right answer by the ball's y location
    w=[[ 4.70944906],
       [ 0.29171137],
       [ 0.03366906],
       [-2.68984273],
       [ 0.02021455],
       [-0.02465323],
       [-2.06048308]]
    x=np.dot(x, w)
   
    x= (1 / (1 + np.exp(-x)))
    if np.rint(x)==0:
        return "down"
    else:
        return "up"


def main(train=False):
    # Don't modify anything in this function.
    # See the constants defined at the top of this file if you'd like to
    # change the FPS, screen size, or round length
    game = Pong(width=WIDTH, height=HEIGHT, MAX_SCORE=MAX_SCORE)

    if train:
        p = PLE(game, fps=FPS, display_screen=False, force_fps=True)
    else:
        p = PLE(game, fps=FPS, display_screen=True, force_fps=False)

    p.init()

    agent_rounds = 0
    cpu_rounds = 0
    agent_score = 0
    cpu_score = 0
    num_frames = 0
    while True:
        if p.game_over():
            if game.score_counts['agent'] > game.score_counts['cpu']:
                agent_rounds += 1
                print('AGENT won round')
            else:
                cpu_rounds += 1
                print('CPU won round')

            if agent_rounds == NUM_ROUNDS or cpu_rounds == NUM_ROUNDS:
                break

            p.reset_game()

        obs = p.getGameState()
        action = agent(normalize(obs))
        reward = p.act(ACTION_MAP[action])

        if reward > 0:
            agent_score += 1
            print('AGENT scored')
        elif reward < 0:
            cpu_score += 1
            print('CPU scored')

        num_frames += 1

    winner = 'AGENT' if agent_rounds > cpu_rounds else 'CPU'
    print('Winner:', winner)
    print('Num frames      :', num_frames)
    print('AGENT rounds won:',agent_rounds)
    print('CPU   rounds won:',cpu_rounds)
    print('AGENT total score:',agent_score)
    print('CPU   total score:',cpu_score)


if __name__ == '__main__':
    # Use train=True to run the game as fast as possible
    # main(train=True)

    # Use train=False to run the game in real time
    main(train=False)
