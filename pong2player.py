import math
import sys

import pygame
from pygame.constants import K_w, K_s, K_o, K_l
from ple.games.utils.vec2d import vec2d
from ple.games.utils import percent_round_int

#import base
from ple.games.base.pygamewrapper import PyGameWrapper


class Ball(pygame.sprite.Sprite):

    def __init__(self, radius, speed, rng,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)

        self.rng = rng
        self.radius = radius
        self.speed = speed
        self.pos = vec2d(pos_init)
        self.pos_before = vec2d(pos_init)
        self.vel = vec2d((speed, -1.0 * speed))

        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((radius * 2, radius * 2))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.circle(
            image,
            (255, 255, 255),
            (radius, radius),
            radius,
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def line_intersection(self, p0_x, p0_y, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y):

        s1_x = p1_x - p0_x
        s1_y = p1_y - p0_y
        s2_x = p3_x - p2_x
        s2_y = p3_y - p2_y

        s = (-s1_y * (p0_x - p2_x) + s1_x * (p0_y - p2_y)) / (-s2_x * s1_y + s1_x * s2_y)
        t = (s2_x * (p0_y - p2_y) - s2_y * (p0_x - p2_x)) / (-s2_x * s1_y + s1_x * s2_y)

        return (s >= 0 and s <= 1 and t >= 0 and t <= 1)

    def update(self, agentPlayer, cpuPlayer, dt):

        self.pos.x += self.vel.x * dt
        self.pos.y += self.vel.y * dt

        is_pad_hit = False

        if self.pos.x <= agentPlayer.pos.x + agentPlayer.rect_width:
            if self.line_intersection(self.pos_before.x, self.pos_before.y, self.pos.x, self.pos.y, agentPlayer.pos.x + agentPlayer.rect_width / 2, agentPlayer.pos.y - agentPlayer.rect_height / 2, agentPlayer.pos.x + agentPlayer.rect_width / 2, agentPlayer.pos.y + agentPlayer.rect_height / 2):
                self.pos.x = max(0, self.pos.x)
                self.vel.x = -1 * self.vel.x
                self.vel.y += agentPlayer.vel.y * 5.0
                self.pos.x += self.radius
                is_pad_hit = True

                self.vel.x += self.speed * 0.05

        if self.pos.x >= cpuPlayer.pos.x - cpuPlayer.rect_width:
            if self.line_intersection(self.pos_before.x, self.pos_before.y, self.pos.x, self.pos.y, cpuPlayer.pos.x - cpuPlayer.rect_width / 2, cpuPlayer.pos.y - cpuPlayer.rect_height / 2, cpuPlayer.pos.x - cpuPlayer.rect_width / 2, cpuPlayer.pos.y + cpuPlayer.rect_height / 2):
                self.pos.x = min(self.SCREEN_WIDTH, self.pos.x)
                self.vel.x = -1 * self.vel.x
                self.vel.y += cpuPlayer.vel.y * 5.0
                self.pos.x -= self.radius
                is_pad_hit = True

                self.vel.x -= self.speed * 0.05

        # Little randomness in order not to stuck in a static loop
        if is_pad_hit:
            self.vel.y += self.rng.random_sample() * 0.1 - 0.05

        if self.pos.y - self.radius <= 0:
            self.vel.y *= -0.99
            self.pos.y += 1.0

        if self.pos.y + self.radius >= self.SCREEN_HEIGHT:
            self.vel.y *= -0.99
            self.pos.y -= 1.0

        self.pos_before.x = self.pos.x
        self.pos_before.y = self.pos.y

        self.rect.center = (self.pos.x, self.pos.y)


class Player(pygame.sprite.Sprite):

    def __init__(self, speed, rect_width, rect_height,
                 pos_init, SCREEN_WIDTH, SCREEN_HEIGHT):

        pygame.sprite.Sprite.__init__(self)

        self.speed = speed
        self.pos = vec2d(pos_init)
        self.vel = vec2d((0, 0))

        self.rect_height = rect_height
        self.rect_width = rect_width
        self.SCREEN_HEIGHT = SCREEN_HEIGHT
        self.SCREEN_WIDTH = SCREEN_WIDTH

        image = pygame.Surface((rect_width, rect_height))
        image.fill((0, 0, 0, 0))
        image.set_colorkey((0, 0, 0))

        pygame.draw.rect(
            image,
            (255, 255, 255),
            (0, 0, rect_width, rect_height),
            0
        )

        self.image = image
        self.rect = self.image.get_rect()
        self.rect.center = pos_init

    def update(self, dy, dt):
        self.vel.y += dy * dt
        self.vel.y *= 0.9

        self.pos.y += self.vel.y

        if self.pos.y - self.rect_height / 2 <= 0:
            self.pos.y = self.rect_height / 2
            self.vel.y = 0.0

        if self.pos.y + self.rect_height / 2 >= self.SCREEN_HEIGHT:
            self.pos.y = self.SCREEN_HEIGHT - self.rect_height / 2
            self.vel.y = 0.0

        self.rect.center = (self.pos.x, self.pos.y)


class Pong2Player(PyGameWrapper):
    """
    Loosely based on code from marti1125's `pong game`_.

    .. _pong game: https://github.com/marti1125/pong/

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    MAX_SCORE : int (default: 11)
        The max number of points the agent or cpu need to score to cause a terminal state.

    cpu_speed_ratio: float (default: 0.5)
        Speed of opponent (useful for curriculum learning)

    players_speed_ratio: float (default: 0.25)
        Speed of player (useful for curriculum learning)

    ball_speed_ratio: float (default: 0.75)
        Speed of ball (useful for curriculum learning)

    """

    def __init__(self, width=64, height=48, player1_speed_ratio=0.5, player2_speed_ratio=0.5, ball_speed_ratio=0.75,  MAX_SCORE=11,
                 player1_name='PLAYER 1', player2_name='PLAYER 2'):

        actions = {
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

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.player1_name = player1_name
        self.player2_name = player2_name

        # the %'s come from original values, wanted to keep same ratio when you
        # increase the resolution.
        self.ball_radius = percent_round_int(height, 0.03)

        self.player1_speed_ratio = player1_speed_ratio
        self.player2_speed_ratio = player2_speed_ratio
        self.ball_speed_ratio = ball_speed_ratio

        self.paddle_width = percent_round_int(width, 0.023)
        self.paddle_height = percent_round_int(height, 0.15)
        self.paddle_dist_to_wall = percent_round_int(width, 0.0625)
        self.MAX_SCORE = MAX_SCORE

        self.player1_dy = 0.0
        self.player2_dy = 0.0
        self.score_sum = 0.0  # need to deal with 11 on either side winning
        self.score_counts = {
            "player1": 0.0,
            "player2": 0.0
        }

    def _handle_player_events(self):
        self.player1_dy = 0
        self.player2_dy = 0

        # consume events from act
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key in [1, 4, 7]:
                    self.player1_dy = -self.player1.speed

                if key in [2, 5, 8]:
                    self.player1_dy = self.player1.speed

                if key in [3, 4, 5]:
                    self.player2_dy = -self.player2.speed

                if key in [6, 7, 8]:
                    self.player2_dy = self.player2.speed



    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player y position.
            * players velocity.
            * cpu y position.
            * ball x position.
            * ball y position.
            * ball x velocity.
            * ball y velocity.

            See code for structure.

        """
        state = {
            "player1_y": self.player1.pos.y,
            "player1_velocity": self.player1.vel.y,

            "player2_y": self.player2.pos.y,
            "player2_velocity": self.player2.vel.y,

            "ball_x": self.ball.pos.x,
            "ball_y": self.ball.pos.y,
            "ball_velocity_x": self.ball.vel.x,
            "ball_velocity_y": self.ball.vel.y
        }

        return state

    def getScore(self):
        return self.score_sum

    def game_over(self):
        # pong used 11 as max score
        return (self.score_counts['player1'] == self.MAX_SCORE) or (
            self.score_counts['player2'] == self.MAX_SCORE)

    def init(self):
        self.score_counts = {
            "player1": 0.0,
            "player2": 0.0
        }

        self.score_sum = 0.0
        self.ball = Ball(
            self.ball_radius,
            self.ball_speed_ratio * self.height,
            self.rng,
            (self.width / 2, self.height / 2),
            self.width,
            self.height
        )

        self.player1 = Player(
            self.player1_speed_ratio * self.height,
            self.paddle_width,
            self.paddle_height,
            (self.paddle_dist_to_wall, self.height / 2),
            self.width,
            self.height)

        self.player2 = Player(
            self.player2_speed_ratio * self.height,
            self.paddle_width,
            self.paddle_height,
            (self.width - self.paddle_dist_to_wall, self.height / 2),
            self.width,
            self.height)

        self.players_group = pygame.sprite.Group()
        self.players_group.add(self.player1)
        self.players_group.add(self.player2)

        self.ball_group = pygame.sprite.Group()
        self.ball_group.add(self.ball)


    def reset(self):
        self.init()
        # after game over set random direction of ball otherwise it will always be the same
        self._reset_ball(1 if self.rng.random_sample() > 0.5 else -1)


    def _reset_ball(self, direction):
        self.ball.pos.x = self.width / 2  # move it to the center

        # we go in the same direction that they lost in but at starting vel.
        self.ball.speed = self.ball_speed_ratio * self.height
        self.ball.vel.x = self.ball.speed * direction
        self.ball.vel.y = (self.rng.random_sample() *
                           self.ball.speed) - self.ball.speed * 0.5

    def draw_score(self, score, x_pos):
        myfont = pygame.font.SysFont('Free sans', 30)
        text = myfont.render(str(int(score)), False, (80, 80, 80))
        text_rect = text.get_rect(center=(x_pos, 20))
        self.screen.blit(text, text_rect)

    def draw_player_name(self, name, x_pos, rot):
        myfont = pygame.font.SysFont('Free sans', 30)
        text = myfont.render(name, False, (80, 80, 80))
        text = pygame.transform.rotate(text, rot)
        text_rect = text.get_rect(center=(x_pos, self.height/2))
        self.screen.blit(text, text_rect)

    def step(self, dt):
        dt /= 1000.0
        self.screen.fill((0, 0, 0))

        self.player1.speed = self.player1_speed_ratio * self.height
        self.player2.speed = self.player2_speed_ratio * self.height
        # self.ball.speed = self.ball_speed_ratio * self.height

        self._handle_player_events()

        # doesnt make sense to have this, but include if needed.
        self.score_sum += self.rewards["tick"]

        self.ball.update(self.player1, self.player2, dt)

        is_terminal_state = False

        # logic
        if self.ball.pos.x <= 0:
            self.score_sum += self.rewards["negative"]
            self.score_counts["player2"] += 1.0
            self._reset_ball(-1)
            is_terminal_state = True

        if self.ball.pos.x >= self.width:
            self.score_sum += self.rewards["positive"]
            self.score_counts["player1"] += 1.0
            self._reset_ball(1)
            is_terminal_state = True

        if is_terminal_state:
            # winning
            if self.score_counts['player1'] == self.MAX_SCORE:
                self.score_sum += self.rewards["win"]

            # losing
            if self.score_counts['player2'] == self.MAX_SCORE:
                self.score_sum += self.rewards["loss"]
        else:
            self.player1.update(self.player1_dy, dt)
            self.player2.update(self.player2_dy, dt)

        self.draw_score(self.score_counts['player1'], 50)
        self.draw_score(self.score_counts['player2'], self.width - 50)
        self.draw_player_name(self.player1_name, 50, 90)
        self.draw_player_name(self.player2_name, self.width - 50, 270)

        self.players_group.draw(self.screen)
        self.ball_group.draw(self.screen)
