import random
from math import exp
from ple.games.pong import Pong
from ple import PLE

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


def agent(x):
    # This function should always return either 'up' or 'down'
    #x=[playersHeight, playersVelocity, ]
    pHeight = x[0]
    playVel = x[1]
    bX=x[2]
    bY=x[3]
    bVx =x[4]
    bVy = x[5]
    oHeight= x[6]
    if pHeight <bY:
        return 'down'
    
    else: 
        return 'up'
    # return random.choice(['up','down'])


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
