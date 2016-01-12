'''
This file implements a simple pygame interface for the simulator.
The simulator can be controlled with the keyboard, or updated without controls.
'''
import numpy as np
from simulator import Simulator
import pygame
import sys
from config import PITCH_WIDTH, PITCH_LENGTH, PLAYER_CONFIG
from config import GOAL_AREA_WIDTH, GOAL_AREA_LENGTH, GOAL_DEPTH, GOAL_WIDTH
from config import KICKABLE, CATCHABLE
from util import angle_position, angle_between, vector_to_tuple, vector
SCALE_FACTOR = 20

def scale(value):
    ''' Scale up a value. '''
    return int(SCALE_FACTOR * value)

def upscale(position):
    ''' Maps a simulator position to a field position. '''
    pos1 = scale(position[0])
    pos2 = scale(position[1] + PITCH_WIDTH/2)
    return np.array([pos1, pos2])

class Interface:
    ''' Implements a pygame interface that allows keyboard control
        of the player, and draws the field, players, and ball. '''

    def __init__(self, simulator = Simulator()):
        ''' Sets up the colors, pygame, and screen. '''
        pygame.init()
        length = scale(PITCH_LENGTH/2 + GOAL_DEPTH)
        width = scale(PITCH_WIDTH)
        self.window = pygame.display.set_mode((length, width))
        self.clock = pygame.time.Clock()
        self.simulator = simulator
        size = (length, width)
        self.background = pygame.Surface(size)
        self.white = pygame.Color(255, 255, 255, 0)
        self.black = pygame.Color(0, 0, 0, 0)
        self.red = pygame.Color(255, 0, 0, 0)
        self.background.fill(pygame.Color(0, 125, 0, 0))

    def control_update(self):
        ''' Uses input from the keyboard to control the player. '''
        keys_pressed = pygame.key.get_pressed()
        mousex, mousey = pygame.mouse.get_pos()
        mousex = mousex / SCALE_FACTOR
        mousey = mousey / SCALE_FACTOR - PITCH_WIDTH/2
        position = vector(mousex, mousey)
        theta = angle_between(self.simulator.player.position, position)
        action_map = {
            pygame.K_SPACE: ('kick', (100, theta)),
            pygame.K_w: ('dash', (10,)),
            pygame.K_s: ('dash', (-10,)),
            pygame.K_a: ('turn', (np.pi/32,)),
            pygame.K_d: ('turn', (-np.pi/32,)),
            pygame.K_b: ('toball', None),
            pygame.K_g: ('shootgoal', (mousey,)),
            pygame.K_t: ('turnball', (theta,)),
            pygame.K_p: ('dribble', position),
            pygame.K_k: ('kickto', position)
        }
        action = None
        for key in action_map:
            if keys_pressed[key]:
                action = action_map[key]
                break
        reward, end_episode = self.simulator.update(action)
        if end_episode:
            if reward == 50:
                print "GOAL"
            else:
                print "OUT"
            self.simulator = Simulator()

    def update(self):
        ''' Performs a single 100ms update. '''
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        self.control_update()
        self.draw()
        self.clock.tick(100)

    def draw_episode(self, simulator, name):
        ''' Draw each state in the simulator into folder name. '''
        lines = ""
        self.simulator = simulator
        for index, state in enumerate(self.simulator.states):
            self.simulator.player.position = state[0]
            self.simulator.player.orientation = state[1]
            self.simulator.goalie.position = state[2]
            self.simulator.goalie.orientation = state[3]
            self.simulator.ball.position = state[4]
            self.draw()
            pygame.image.save(self.window, 'screens/'+ name + '/' + str(index)+'.png')
            lines += str(index) + '.png\n'
            self.clock.tick()
        with open('screens/' + name + '/filenames.txt', 'w') as filename:
            filename.write(lines)

    def plot_episode(self, name):
	''' Plot an episode in a single image. '''
	old_window = self.window
        surface = pygame.Surface(self.window.get_size())
        surface.blit(self.background, (0, 0))
	count = 1. * len(self.simulator.states)
	for index, state in enumerate(self.simulator.states):
            self.simulator.player.position = state[0]
            self.simulator.player.orientation = state[1]
            self.simulator.goalie.position = state[2]
            self.simulator.goalie.orientation = state[3]
            self.simulator.ball.position = state[4]
	    #self.window = pygame.Surface(self.window.get_size())
            alpha = int(180*(index / count))
            if index % 8 == 0:
                self.window.set_alpha(alpha)
                self.draw(True)
                surface.blit(self.window, (0, 0))
                old_window.blit(surface, (0, 0))
                print alpha
        pygame.image.save(surface, './screens/'+ name + '.png')

    def draw(self, fade = False):
        ''' Draw the field and players. '''
        self.window.blit(self.background, (0, 0))

        # Draw goal and penalty areas
        length = scale(PITCH_LENGTH/2)
        width = scale(PITCH_WIDTH)

        self.draw_vertical(length, 0, width)
        self.draw_box(GOAL_AREA_WIDTH, GOAL_AREA_LENGTH)
        #self.draw_box(PENALTY_AREA_WIDTH, PENALTY_AREA_LENGTH)

        depth = length + scale(GOAL_DEPTH)
        self.draw_horizontal(width/2 - scale(GOAL_WIDTH/2), length, depth)
        self.draw_horizontal(width/2 + scale(GOAL_WIDTH/2), length, depth)

        #self.draw_radius(vector(0, 0), CENTRE_CIRCLE_RADIUS)
        # Draw Players
        self.draw_player(self.simulator.player, self.white)
	if not fade:
            self.draw_radius(self.simulator.player.position, KICKABLE)
        self.draw_player(self.simulator.goalie, self.red)
	if not fade:
            self.draw_radius(self.simulator.goalie.position, CATCHABLE)
        # Draw ball
        self.draw_entity(self.simulator.ball, self.black)
        pygame.display.update()

    def draw_box(self, area_width, area_length):
        ''' Draw a box at the goal line. '''
        lower_corner = scale(PITCH_WIDTH/2 - area_width/2)
        upper_corner = lower_corner + scale(area_width)
        line = scale(PITCH_LENGTH/2 - area_length)
        self.draw_vertical(line, lower_corner, upper_corner)
        self.draw_horizontal(lower_corner, line, scale(PITCH_LENGTH/2))
        self.draw_horizontal(upper_corner, line, scale(PITCH_LENGTH/2))

    def draw_player(self, agent, colour):
        ''' Draw a player with given position and orientation. '''
        size = PLAYER_CONFIG['SIZE']
        self.draw_entity(agent, colour)
        radius_end = size*angle_position(agent.orientation)
        pos = vector_to_tuple(upscale(agent.position))
        end = vector_to_tuple(upscale(agent.position + radius_end))
        pygame.draw.line(self.window, self.black, pos, end)

    def draw_radius(self, position, radius):
        ''' Draw an empty circle. '''
        pos = vector_to_tuple(upscale(position))
        radius = scale(radius)
        pygame.draw.circle(self.window, self.white, pos, radius, 1)

    def draw_entity(self, entity, colour):
        ''' Draws an entity as a ball. '''
        pos = vector_to_tuple(upscale(entity.position))
        radius = scale(entity.size)
        pygame.draw.circle(self.window, colour, pos, radius)

    def draw_horizontal(self, yline, xline1, xline2):
        ''' Draw a horizontal line. '''
        pos1 = (xline1, yline)
        pos2 = (xline2, yline)
        pygame.draw.line(self.window, self.white, pos1, pos2)

    def draw_vertical(self, xline, yline1, yline2):
        ''' Draw a vertical line. '''
        pos1 = (xline, yline1)
        pos2 = (xline, yline2)
        pygame.draw.line(self.window, self.white, pos1, pos2)

def main():
    ''' Runs the interface. '''
    interface = Interface()
    while True:
        interface.update()

if __name__ == '__main__':
    main()
