'''
This file implements a soccer simulator based on the robocup soccer simulator,
but without sensory perceptions, networking, and real-time operation.
'''
import numpy as np
from numpy.random import uniform
from numpy.linalg import norm
from util import vector, angle_between, angle_position
from config import MINPOWER, MAXPOWER, KICKABLE, CATCHABLE, INERTIA_MOMENT
from config import PLAYER_CONFIG, BALL_CONFIG, CATCH_PROBABILITY
from config import PITCH_WIDTH, PITCH_LENGTH, GOAL_WIDTH, GOAL_DEPTH
from config import GOAL_AREA_WIDTH, GOAL_AREA_LENGTH


def bound(value, lower, upper):
    ''' Clips off a value which exceeds the lower or upper bounds. '''
    if value < lower:
        return lower
    elif value > upper:
        return upper
    else:
        return value

def bound_vector(vect, maximum):
    ''' Bounds a vector between a negative and positive maximum range. '''
    xval = bound(vect[0], -maximum, maximum)
    yval = bound(vect[1], -maximum, maximum)
    return vector(xval, yval)

def angle_difference(angle1, angle2):
    ''' Computes the real difference between angles. '''
    return norm_angle(angle1 - angle2)

def norm_angle(angle):
    ''' Normalize the angle between -pi and pi. '''
    while angle > np.pi:
        angle -= 2*np.pi
    while angle < -np.pi:
        angle += 2*np.pi
    return angle

def angle_close(angle1, angle2):
    ''' Determines whether an angle1 is close to angle2. '''
    return abs(angle_difference(angle1, angle2)) < np.pi/8

class Entity:
    ''' This is a base entity class, representing moving objects. '''

    def __init__(self, config):
        self.rand = config['RAND']
        self.accel_max = config['ACCEL_MAX']
        self.speed_max = config['SPEED_MAX']
        self.power_rate = config['POWER_RATE']
        self.decay = config['DECAY']
        self.size = config['SIZE']
        self.position = vector(0, 0)
        self.velocity = vector(0, 0)

    def update(self):
        ''' Update the position and velocity. '''
        self.position += self.velocity
        self.velocity *= self.decay

    def accelerate(self, power, theta):
        ''' Applies a power to the entity in direction theta. '''
        rrand = uniform(-self.rand, self.rand)
        theta = (1 + rrand) * theta
        rmax = self.rand*norm(self.velocity)
        noise = vector(uniform(-rmax, rmax), uniform(-rmax, rmax))
        rate = float(power) * self.power_rate
        acceleration = rate * angle_position(theta) + noise
        acceleration = bound_vector(acceleration, self.accel_max)
        self.velocity += acceleration
        self.velocity = bound_vector(self.velocity, self.speed_max)

    def decollide(self, other):
        ''' Shift overlapping entities apart. '''
        overlap = (self.size + other.size - self.distance(other))/2
        theta1 = angle_between(self.position, other.position)
        theta2 = angle_between(other.position, self.position)
        self.position += overlap * angle_position(theta2)
        other.position += overlap * angle_position(theta1)
        self.velocity *= -1
        other.velocity *= -1

    def colliding(self, other):
        ''' Check if two entities are overlapping. '''
        dist = self.distance(other)
        return dist < self.size + other.size

    def distance(self, other):
        ''' Computes the euclidean distance to another entity. '''
        return norm(self.position - other.position)

    def in_area(self, left, right, bot, top):
        ''' Checks if the entity is in the area. '''
        xval, yval = self.position
        in_length = left <= xval <= right
        in_width = bot <= yval <= top
        return in_length and in_width

class Player(Entity):
    ''' This represents a player with a position,
        velocity and an orientation. '''

    def __init__(self, position, orientation):
        ''' The values for this class are defined by the player constants. '''
        Entity.__init__(self, PLAYER_CONFIG)
        self.position = position
        self.orientation = orientation

    def homothetic_centre(self, ball):
        ''' Computes the homothetic centre between the player and the ball. '''
        ratio = 1. / (self.size + ball.size)
        position = (ball.position*self.size + self.position*ball.size)
        return ratio * position

    def tangent_points(self, htc):
        ''' Finds the tangent points on the player wrt to homothetic centre. '''
        diff = htc - self.position
        square = sum(diff**2)
        if square <= self.size**2:
            delta = 0.0
        else:
            delta = np.sqrt(square - self.size**2)
        xt1 = (diff[0]*self.size**2 + self.size*diff[1]*delta) / square
        xt2 = (diff[0]*self.size**2 - self.size*diff[1]*delta) / square
        yt1 = (diff[1]*self.size**2 + self.size*diff[0]*delta) / square
        yt2 = (diff[1]*self.size**2 - self.size*diff[0]*delta) / square
        tangent1 = vector(xt1, yt1) + self.position
        tangent2 = vector(xt1, yt2) + self.position
        tangent3 = vector(xt2, yt1) + self.position
        tangent4 = vector(xt2, yt2) + self.position
        if norm(tangent1 - self.position) == self.size:
            return tangent1, tangent4
        else:
            return tangent2, tangent3

    def ball_angles(self, ball, angle):
        ''' Determines which angle to kick the ball along. '''
        htc = self.homothetic_centre(ball)
        tangent1, tangent2 = self.tangent_points(htc)
        target = self.position + self.size*angle_position(angle)
        if norm(tangent1 - target) < norm(tangent2 - target):
            return angle_between(htc, tangent1)
        else:
            return angle_between(htc, tangent2)

    def kick_power(self, ball):
        ''' Determines the kick power weighting given ball position. '''
        angle = angle_between(self.position, ball.position)
        dir_diff = abs(angle_difference(angle, self.orientation))
        dist = self.distance(ball)
        return (1 - 0.25*dir_diff/np.pi - 0.25*dist/KICKABLE)

    def facing_ball(self, ball):
        ''' Determines whether the player is facing the ball. '''
        angle = angle_between(self.position, ball.position)
        return self.facing_angle(angle)

    def facing_angle(self, angle):
        ''' Determines whether the player is facing an angle. '''
        return angle_close(self.orientation, angle)

    def turn(self, angle):
        ''' Turns the player. '''
        moment = norm_angle(angle)
        speed = norm(self.velocity)
        angle = moment / (1 + INERTIA_MOMENT * speed)
        self.orientation = self.orientation + angle

    def dash(self, power):
        ''' Dash forward. '''
        power = bound(power, MINPOWER, MAXPOWER)
        self.accelerate(power, self.orientation)

    def can_kick(self, ball):
        ''' Determines whether the player can kick the ball. '''
        return self.distance(ball) <= KICKABLE

    def kick_ball(self, ball, power, direction):
        ''' Kicks the ball. '''
        if self.can_kick(ball):
            power = bound(power, MINPOWER, MAXPOWER)
            power *= self.kick_power(ball)
            ball.accelerate(power, self.orientation + direction)

    def kick_towards(self, ball, power, direction):
        ''' Kick the ball directly to a direction. '''
        self.kick_ball(ball, power, direction - self.orientation)

    def shoot_goal(self, ball, ypos):
        ''' Shoot the goal at a targeted position on the goal line. '''
        ypos = bound(ypos, -GOAL_WIDTH/2, GOAL_WIDTH/2)
        target = vector(PITCH_LENGTH/2 + ball.size, ypos)
        self.kick_to(ball, target)

    def face_ball(self, ball):
        ''' Turn the player towards the ball. '''
        theta = angle_between(self.position, ball.position)
        self.face_angle(theta)

    def face_angle(self, angle):
        ''' Turn the player towards and angle. '''
        self.turn(angle - self.orientation)

    def to_ball(self, ball):
        ''' Move towards the ball. '''
        if not self.facing_ball(ball):
            self.face_ball(ball)
        elif not self.can_kick(ball):
            self.dash(10)

    def kick_to(self, ball, target):
        ''' Kick the ball to a target position. '''
        if not self.can_kick(ball):
            self.to_ball(ball)
        else:
            accel = (1 - ball.decay) * (target - self.position) - ball.velocity
            power = norm(accel) / (self.kick_power(ball) * ball.power_rate)
            theta = np.arctan2(accel[1], accel[0])
            self.kick_towards(ball, power, theta)

    def turn_ball(self, ball, angle):
        ''' Turn the ball around the player. '''
        if not self.can_kick(ball):
            self.to_ball(ball)
        elif not self.facing_angle(angle):
            self.face_angle(angle)
        elif self.size < self.distance(ball):
            theta = self.ball_angles(ball, angle)
            power = 0.1 / self.kick_power(ball)
            self.kick_towards(ball, power, theta)

    def dribble(self, ball, target):
        ''' Dribble the ball to a position. '''
        angle = angle_between(self.position, ball.position)
        theta = angle_between(self.position, target)
        if not self.can_kick(ball):
            self.to_ball(ball)
        elif ball.close_to(target):
            pass
        elif not angle_close(angle, theta):
            self.turn_ball(ball, theta)
        elif not self.facing_angle(theta):
            self.face_angle(theta)
        elif self.distance(ball) < (KICKABLE + self.size + ball.size)/2:
            self.kick_towards(ball, 1.5, theta)
        else:
            self.dash(10)

def keeper_line(ball):
    ''' Finds the line the keeper wants to stay to. '''
    grad = -ball[1]/(PITCH_LENGTH/2 - ball[0])
    yint = ball[1] - grad * ball[0]
    return grad, yint

def keeper_target(ball):
    ''' Target the keeper wants to move towards. '''
    grad, yint = keeper_line(ball)
    if ball[0] < PITCH_LENGTH/2 - GOAL_AREA_LENGTH:
        xval = ball[0]
    else:
        if ball[1] < -GOAL_AREA_WIDTH/2:
            xval = (-GOAL_AREA_WIDTH/2 - yint)/grad
        else:
            xval = (GOAL_AREA_WIDTH/2 - yint)/grad
    xval = bound(xval, PITCH_LENGTH/2 - GOAL_AREA_LENGTH, PITCH_LENGTH/2)
    yval = bound(grad * xval + yint, -GOAL_AREA_WIDTH/2, GOAL_AREA_WIDTH/2)
    return vector(xval, yval)

class Goalie(Player):
    ''' This class defines a special goalie player. '''

    def move(self, ball, player):
        ''' This moves the goalie. '''
        ball_end = ball.position + ball.velocity / (1 - ball.decay)
        diff = ball_end - ball.position
        grad = diff[1] / diff[0]
        yint = ball.position[1] - grad * ball.position[0]
        goal_y = grad * PITCH_LENGTH/2 + yint
        if ball_end[0] > PITCH_LENGTH/2 and -GOAL_WIDTH/2 - CATCHABLE <= goal_y <= GOAL_WIDTH/2 + CATCHABLE and grad != 0:
	    grad2 = -1/grad
	    yint2 = self.position[1] - grad2 * self.position[0]
	    ballx = (yint2 - yint) / (grad - grad2)
	    bally = grad * ballx + yint
	    target = vector(ballx, bally)
	    self.move_towards(20, target)
            self.orientation = angle_between(self.position, target)
        else:
            self.orientation = angle_between(self.position, ball_end)
            self.move_towards(8, ball_end)

    def move_towards(self, power, target):
        ''' Move towards target position. '''
        theta = angle_between(self.position, target)
        self.accelerate(power, theta)

    def can_catch(self, ball):
        ''' Determines whether the goalie can catch the ball. '''
        can_catch = self.distance(ball) < CATCHABLE
        return np.random.random() <= CATCH_PROBABILITY and can_catch

class Ball(Entity):
    ''' This class represents the ball, which has no orientation. '''

    def __init__(self, position):
        ''' The values for this class are defined by the ball constants. '''
        Entity.__init__(self, BALL_CONFIG)
        self.position = position

    def close_to(self, position):
        ''' Determines whether the ball is close to a postion. '''
        return norm(self.position - position) <= 1.5

    def goal_distance(self):
        ''' Returns the distance from the goal box. '''
        if self.position[0] < PITCH_LENGTH/2:
            if self.position[1] < -GOAL_WIDTH/2:
                bot_corner = vector(PITCH_LENGTH/2, -GOAL_WIDTH/2)
                return norm(self.position - bot_corner)
            elif self.position[1] > GOAL_WIDTH/2:
                top_corner = vector(PITCH_LENGTH/2, GOAL_WIDTH/2)
                return norm(self.position - top_corner)
            else:
                return PITCH_LENGTH/2 - self.position[0]
        else:
            if self.position[1] < -GOAL_WIDTH/2:
                return GOAL_WIDTH/2 - self.position[1]
            elif self.position[1] > GOAL_WIDTH/2:
                return self.position[1] - GOAL_WIDTH/2
            else:
                return 0

    def in_field(self):
        ''' Checks if the ball has left the field. '''
        return self.in_area(0, PITCH_LENGTH/2, -PITCH_WIDTH/2, PITCH_WIDTH/2)

    def in_net(self):
        ''' Checks if the ball is in the net. '''
        return self.in_area(PITCH_LENGTH/2, PITCH_LENGTH/2 + GOAL_DEPTH, -GOAL_WIDTH/2, GOAL_WIDTH/2)

    def in_goalbox(self):
        ''' Checks if the ball is in the goal box. '''
        return self.in_area(PITCH_LENGTH/2 - GOAL_AREA_LENGTH, PITCH_LENGTH/2, -GOAL_AREA_WIDTH/2, GOAL_AREA_WIDTH)

class Simulator:
    ''' This class represents the environment. '''

    def __init__(self):
        ''' The entities are set up and added to a space. '''
        initial_player = vector(0, uniform(-PITCH_WIDTH/2, PITCH_WIDTH/2))
        angle = angle_between(initial_player, vector(PITCH_LENGTH/2, 0))
        self.player = Player(initial_player, angle)

        initial_ball = initial_player + KICKABLE * angle_position(angle)
        self.ball = Ball(initial_ball)

        initial_goalie = keeper_target(initial_ball)
        angle2 = angle_between(initial_goalie, initial_ball)
        self.goalie = Goalie(initial_goalie, angle2)

        self.entities = [self.player, self.goalie, self.ball]
        self.states = []

        self.time = 0

    def get_state(self):
        ''' Returns the representation of the current state. '''
        state = np.concatenate((
            self.player.position,
            self.player.velocity,
            [self.player.orientation],
            self.goalie.position,
            self.goalie.velocity,
            [self.goalie.orientation],
            self.ball.position,
            self.ball.velocity))
        return state

    def perform_action(self, action, agent):
        ''' Applies for selected action for the given agent. '''
        if action:
            act, parameters = action
            if act == 'kick':
                agent.kick_ball(self.ball, parameters[0], parameters[1])
            elif act == 'dash':
                agent.dash(parameters[0])
            elif act == 'turn':
                agent.turn(parameters[0])
            elif act == 'toball':
                agent.to_ball(self.ball)
            elif act == 'shootgoal':
                agent.shoot_goal(self.ball, parameters[0])
            elif act == 'turnball':
                agent.turn_ball(self.ball, parameters[0])
            elif act == "dribble":
                agent.dribble(self.ball, parameters)
            elif act == "kickto":
                agent.kick_to(self.ball, parameters[0])

    def resolve_collisions(self):
        ''' Shift apart all colliding entities with one pass. '''
        for index, entity1 in enumerate(self.entities):
            for entity2 in self.entities[index+1:]:
                if entity1.colliding(entity2):
                    entity1.decollide(entity2)

    def terminal_check(self):
        ''' Determines if the episode is ended, and the reward. '''
        if self.ball.in_net():
            end_episode = True
            reward = 50
        elif self.goalie.can_catch(self.ball) or not self.ball.in_field():
            end_episode = True
            reward = -self.ball.goal_distance()
        else:
            end_episode = False
            reward = 0
        if end_episode:
            self.states.append([
                self.player.position.copy(),
                self.player.orientation,
                self.goalie.position.copy(),
                self.goalie.orientation,
                self.ball.position.copy()])
        return reward, end_episode

    def update(self, action):
        ''' Performs a single transition with the given action,
            then returns the new state and a reward. '''
        self.states.append([
                self.player.position.copy(),
                self.player.orientation,
                self.goalie.position.copy(),
                self.goalie.orientation,
                self.ball.position.copy()])
        self.perform_action(action, self.player)
        self.goalie.move(self.ball, self.player)
        for entity in self.entities:
            entity.update()
        self.resolve_collisions()
        return self.terminal_check()

    def is_stable(self):
        ''' Determines whether objects have stopped moving. '''
        speeds = [norm(entity.velocity) for entity in self.entities]
        return max(speeds) < 0.1

    def take_action(self, action):
        ''' Take a full, stabilised update. '''
        steps = 0
        self.time += 1
        if self.time == 100:
            reward = - self.ball.goal_distance()
            end_episode = True
            state = self.get_state()
            return state, reward, end_episode, steps
        end_episode = False
        run = True
        while run:
            steps += 1
            reward, end_episode = self.update(action)
            run = not end_episode
            if action and run:
                act, params = action
                run = not self.player.can_kick(self.ball)
                if act == "dribble":
                    run = not self.ball.close_to(params) or run
                elif act == "kickto":
                    run = norm(self.ball.velocity) > 0.1 or run
                elif act == "turnball":
                    theta = angle_between(self.player.position, self.ball.position)
                    run = not angle_close(theta, params[0]) or run
                elif act == "shootgoal":
                    run = not end_episode
                else:
                    run = False
        state = self.get_state()
        return state, reward, end_episode, steps
