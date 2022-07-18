import math
import random
import time
import turtle

import inline as inline
import matplotlib
import numpy as np
from deap import base
from deap import creator
from deap import tools
from matplotlib import pyplot as plt

XSIZE = YSIZE = 16  # Number of grid cells in each direction (do not change this)

HEADLESS = True


# windows
class DisplayGame:
    def __init__(self, XSIZE, YSIZE):
        # SCREEN
        self.win = turtle.Screen()
        self.win.title("EVCO Snake game")
        self.win.bgcolor("grey")
        self.win.setup(width=(XSIZE * 20) + 40, height=(YSIZE * 20) + 40)
        # self.win.screensize((XSIZE*20)+20,(YSIZE*20)+20)
        self.win.tracer(0)

        # Snake Head
        self.head = turtle.Turtle()
        self.head.shape("square")
        self.head.color("black")

        # Snake food
        self.food = turtle.Turtle()
        self.food.shape("circle")
        self.food.color("yellow")
        self.food.penup()
        self.food.shapesize(0.55, 0.55)
        self.segments = []

    def reset(self, snake):
        self.segments = []
        self.head.penup()
        self.food.goto(-500, -500)
        self.head.goto(-500, -500)
        for i in range(len(snake) - 1):
            self.add_snake_segment()
        self.update_segment_positions(snake)

    def update_food(self, new_food):
        self.food.goto(((new_food[1] - 9) * 20) + 20, (((9 - new_food[0]) * 20) - 10) - 20)

    def update_segment_positions(self, snake):
        self.head.goto(((snake[0][1] - 9) * 20) + 20, (((9 - snake[0][0]) * 20) - 10) - 20)
        for i in range(len(self.segments)):
            self.segments[i].goto(((snake[i + 1][1] - 9) * 20) + 20, (((9 - snake[i + 1][0]) * 20) - 10) - 20)

    def add_snake_segment(self):
        self.new_segment = turtle.Turtle()
        self.new_segment.speed(0)
        self.new_segment.shape("square")
        self.new_segment.color(random.choice(["green", 'black', 'red', 'blue']))
        self.new_segment.penup()
        self.segments.append(self.new_segment)


def transDirect(nextDirec):
    if nextDirec == 0:
        return "left"
    elif nextDirec == 1:
        return "right"
    elif nextDirec == 2:
        return "up"
    elif nextDirec == 3:
        return "down"


# snakemovement
class snake:
    def __init__(self, _XSIZE, _YSIZE):
        self.XSIZE = _XSIZE
        self.YSIZE = _YSIZE
        self.reset()

    def move_AI(self, AIbrain):
        head_x = self.snake[0][0]
        head_y = self.snake[0][1]
        tail_x = self.snake[-1][0]
        tail_y = self.snake[-1][1]
        snake_long = len(self.snake)
        food_x = self.food[0]
        food_y = self.food[1]
        d_food_h = snake_game.sensor_food_ahead()
        d_tail_h = snake_game.sensor_tail_ahead()
        d_wall_h = snake_game.sensor_wall_head()
        # inputs=[1,1,1,1,1,1,1,1,1,1]
        inputs = [head_x, head_y, tail_x, tail_y, food_x, food_y, snake_long, d_food_h,
                  d_tail_h, d_wall_h]
        #print(inputs)
        outcome = AIbrain.feed_forward(inputs)
        #print(outcome)
        nextDirec = np.argmax(outcome)
        nextDirec = transDirect(nextDirec)
        return nextDirec

    def reset(self):
        self.snake = [[8, 10], [8, 9], [8, 8], [8, 7], [8, 6], [8, 5], [8, 4], [8, 3], [8, 2], [8, 1],
                      [8, 0]]  # Initial snake co-ordinates [ypos,xpos]
        self.food = self.place_food()
        self.ahead = []
        self.snake_direction = "right"

    def place_food(self):
        self.food = [random.randint(1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        while (self.food in self.snake):
            self.food = [random.randint(1, (YSIZE - 2)), random.randint(1, (XSIZE - 2))]
        return (self.food)

    def update_snake_position(self):
        self.snake.insert(0, [
            self.snake[0][0] + (self.snake_direction == "down" and 1) + (self.snake_direction == "up" and -1),
            self.snake[0][1] + (self.snake_direction == "left" and -1) + (self.snake_direction == "right" and 1)])

    def food_eaten(self):
        if self.snake[0] == self.food:  # When snake eats the food
            return True
        else:
            last = self.snake.pop()  # [1] If it does not eat the food, it moves forward and so last tail item is removed
            return False

    def snake_turns_into_self(self):
        if self.snake[0] in self.snake[1:]:
            return True
        else:
            return False

    def snake_hit_wall(self):
        if self.snake[0][0] == 0 or self.snake[0][0] == (YSIZE - 1) or self.snake[0][1] == 0 or self.snake[0][1] == (
                XSIZE - 1):
            return True
        else:
            return False

    # Example sensing functions
    def getAheadLocation(self):
        self.ahead = [self.snake[0][0] + (self.snake_direction == "down" and 1) + (self.snake_direction == "up" and -1),
                      self.snake[0][1] + (self.snake_direction == "left" and -1) + (
                              self.snake_direction == "right" and 1)]

    def sense_wall_ahead(self):
        self.getAheadLocation()
        return (self.ahead[0] == 0 or self.ahead[0] == (YSIZE - 1) or self.ahead[1] == 0 or self.ahead[1] == (
                XSIZE - 1))

    def sense_food_ahead(self):
        self.getAheadLocation()
        return self.food == self.ahead

    def sense_tail_ahead(self):
        self.getAheadLocation()
        return self.ahead in self.snake

    # distance caculator

    # sensors about position where the snake is
    def sensor_wall_head(self):
        self.getAheadLocation()
        a = 15 - self.snake[0][0]
        b = 15 - self.snake[0][1]
        if a > b:
            return a
        else:
            return b

    def sensor_food_ahead(self):
        self.getAheadLocation()
        return np.sqrt((self.snake[0][0] - self.food[0]) ** 2 + (self.snake[0][1] - self.food[1]) ** 2)

    def sensor_tail_ahead(self):
        self.getAheadLocation()
        return np.sqrt((self.snake[0][0] - self.snake[-1][0]) ** 2 + (self.snake[0][1] - self.snake[-1][1]) ** 2)


snake_game = snake(XSIZE, YSIZE)
if not HEADLESS:
    display = DisplayGame(XSIZE, YSIZE)

display = 0


def run_game(AIbrain, display, snake_game, headless=HEADLESS):
    score = 0
    nextDirec = 'up'
    live_time=0
    #nextDirec = snake_game.move_AI(AIbrain)
    snake_game.reset()
    if not headless:
        display.reset(snake_game.snake)
        display.win.update()
    snake_game.place_food()
    game_over = False
    snake_direction = "up"
    flag = True
    while not game_over:

        #  ****YOUR AI BELOW HERE******************

        # Here is a very silly random snake controller. It moves with a correlated random walk,
        # and the only sensible decision it makes is not to turn directly back on itself (possible in this game)
        # *** Replace this with your evolved controller here to decide on the direction the snake should take*
        # snake_direction = "down" / snake_direction = "up" / snake_direction = "left" / snake_direction = "right"
        # if random.random() < 1:

        if nextDirec == "left":
            nextDirec = snake_game.move_AI(AIbrain)
            snake_direction = nextDirec
            # new_snake_direction = random.choice(["left", "up", "down"])
        elif nextDirec == "right":
            nextDirec = snake_game.move_AI(AIbrain)
            snake_direction = nextDirec
            # new_snake_direction = random.choice(["right", "up", "down"])
        elif nextDirec == "up":
            nextDirec = snake_game.move_AI(AIbrain)
            snake_direction = nextDirec

            # new_snake_direction = random.choice(["left", "up", "right"])
        elif nextDirec == "down":
            nextDirec = snake_game.move_AI(AIbrain)
            snake_direction = nextDirec
            # new_snake_direction = random.choice(["left", "down", "right"])
        # snake_direction = nextDirec
        snake_game.snake_direction = snake_direction
        print(snake_direction)

        # Here is an example sensing function
        if snake_game.sense_tail_ahead():
            print("Tail ahead!!!!")
            # time.sleep(0.1)

        #  ****YOUR AI ABOVE HERE******************

        snake_game.update_snake_position()

        if live_time/(score+1) >= 100:
            game_over = True
            fitness = score
            print("enough step")
        else:
            fitness = 3 * score + 0.01 * live_time

        if live_time > 200 and score == 0:
            game_over = True
            fitness = 0


        # Check if food is eaten
        if snake_game.food_eaten():
            snake_game.place_food()
            score += 1
            if not headless: display.add_snake_segment()

        # Game over if the snake runs over itself
        if snake_game.snake_turns_into_self():
            game_over = True
            print("Snake turned into itself!")

        # Game over if the snake goes through a wall
        if snake_game.snake_hit_wall():
            game_over = True
            print("Snake hit a wall!")

        if not headless:
            display.update_food(snake_game.food)
            display.update_segment_positions(snake_game.snake)
            display.win.update()
            # time.sleep(0.2)  # Change this to modify the speed the game runs at when displayed.
        live_time+=1


    print("\nFINAL score - " + str(score))
    print()
    if not headless: turtle.done()
    return fitness


class MLP(object):  # 神经网络
    # initialize the MLP(4)
    def __init__(self, numInput, numHidden1, numHidden2, numOutput):
        self.fitness = 0
        self.numInput = numInput
        self.numHidden1 = numHidden1
        self.numHidden2 = numHidden2
        self.numOutput = numOutput

        self.w_i_h1 = np.random.randn(self.numHidden1, self.numInput)
        self.w_h1_h2 = np.random.randn(self.numHidden2, self.numHidden1)
        self.w_h2_o = np.random.randn(self.numOutput, self.numHidden2)

        self.b_i_h1 = [0] * numHidden1
        self.b_i_h2 = [0] * numHidden2
        self.b_i_output = [0] * numOutput

        self.ReLU = lambda x: max(0, x)


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def feed_forward(self, inputs):
        inputsBias = inputs[:]
        inputsBias.insert(len(inputs), 1)

        h1 = np.dot(self.w_i_h1, inputs)  # feed input to h1
        h1 = [h + b for h, b in zip(h1, self.b_i_h1)]

        h1 = [self.ReLU(x) for x in h1]  # activate hidden layer1

        h2 = np.dot(self.w_h1_h2, h1)  # feed h1 to h2
        h2 = [h + b for h, b in zip(h2, self.b_i_h2)]

        h2 = [self.ReLU(x) for x in h2]  # active h2

        output = np.dot(self.w_h2_o, h2)
        #output = [o + b for o, b in zip(output, self.b_i_output)]
        #print(output)

        output = self.softmax(output)
        #print(output)
        return output

    def getWeightsLinear(self):
        flat_w_i_h1 = list(self.w_i_h1.flatten())
        flat_w_h1_h2 = list(self.w_h1_h2.flatten())
        flat_w_h2_o = list(self.w_h2_o.flatten())
        return (flat_w_i_h1 + flat_w_h1_h2 + flat_w_h2_o)

    def setWeightsLinear(self, Wgenome):
        numWeights_I_H1 = self.numHidden1 * self.numInput
        numWeights_H1_H2 = self.numHidden2 * self.numHidden1
        numWeights_H2_O = self.numOutput * self.numHidden2

        # weights
        # first
        first_slice_end = numWeights_I_H1
        self.w_i_h1 = np.array(Wgenome[:first_slice_end])
        self.w_i_h1 = self.w_i_h1.reshape((self.numHidden1, self.numInput))

        # second
        second_slice_end = first_slice_end + numWeights_H1_H2
        self.w_h1_h2 = np.array(Wgenome[first_slice_end:second_slice_end])
        self.w_h1_h2 = self.w_h1_h2.reshape((self.numHidden2, self.numHidden1))

        # third
        third_slice_end = second_slice_end + numWeights_H2_O
        self.w_h2_o = np.array(Wgenome[second_slice_end:third_slice_end])
        self.w_h2_o = self.w_h2_o.reshape((self.numOutput, self.numHidden2))

        # Biases
        fourth_slice_end = third_slice_end + self.numHidden1
        self.b_i_h1 = np.array(Wgenome[third_slice_end:fourth_slice_end])

        fith_slice_end = fourth_slice_end + self.numHidden2
        self.b_i_h2 = np.array(Wgenome[fourth_slice_end:fith_slice_end])

        self.b_i_output = np.array([Wgenome[fith_slice_end]])


# The Genetic Algorithm
numInputNodes = 10
numHiddenNodes1 = 6
numHiddenNodes2 = 6
numOutputNodes = 4

NUM_WEIGHTS = (numInputNodes * numHiddenNodes1) + (numHiddenNodes1 * numHiddenNodes2) + (
        numHiddenNodes2 * numOutputNodes)
NUM_BIASES = numHiddenNodes1 + numHiddenNodes2 + numOutputNodes

IND_SIZE = NUM_BIASES + NUM_WEIGHTS

#myNet = MLP(numInputNodes, numHiddenNodes1, numHiddenNodes2, numOutputNodes)

snakeGame = snake(_XSIZE=XSIZE, _YSIZE=YSIZE)
myNet = MLP(10, 6, 6, 4)

# this is where we try to input data to MLP and get outcome
# outcome = myNet.feed_forward(inputs)
# outcome = max(outcome)

# run_game(display,snake_game, headless=HEADLESS)

#snakeGame = run_game(myNet, display, snake_game, HEADLESS)

creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual,
                 toolbox.attr_float, n=IND_SIZE)


def evaluate(indiv, myNet):
    #fitness = 0
    myNet.setWeightsLinear(indiv)
    snakeGame.reset()
    fitness = run_game(myNet, display, snake_game, headless=True)
    return fitness,


toolbox.register("evaluate", evaluate)
toolbox.register("select", tools.selTournament, tournsize=5)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

stats = tools.Statistics(key=lambda ind: ind.fitness.values)
stats.register("avg", np.mean)
stats.register("std", np.std)
stats.register("min", np.min)
stats.register("max", np.max)

logbook = tools.Logbook()

pop = toolbox.population(n=1000)

fitnesses = [toolbox.evaluate(indiv, myNet) for indiv in pop]
for ind, fit in zip(pop, fitnesses):
    ind.fitness.values = fit

NGEN = 500
for g in range(NGEN):
    print("-- Generation %i --" % g)

    offspring = toolbox.select(pop, len(pop))
    offspring = list(map(toolbox.clone, offspring))

    for mutant in offspring:
        toolbox.mutate(mutant)
        del mutant.fitness.values

    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
    fitnesses = [toolbox.evaluate(indiv, myNet) for indiv in invalid_ind]
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    pop[:] = offspring
    record = stats.compile(pop)
    logbook.record(gen=g, **record)


###drawing

#%matplotlib inline

gen = logbook.select("gen")
_min = logbook.select("min")
_max = logbook.select("max")
avgs = logbook.select("avg")
stds = logbook.select("std")

plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)

fig, ax1 = plt.subplots()
line1 = ax1.plot(gen, avgs)
#line1 = ax1.errorbar(gen, avgs, yerr=stds, errorevery=2)
ax1.set_xlabel("Generation")
ax1.set_ylabel("Mean Fitness")

line2 = ax1.plot(gen, _min)
line3 = ax1.plot(gen, _max)
plt.show()

# bestInd = tools.selBest(pop, 1)[0]
# print(bestInd)
# myNet.setWeightsLinear(bestInd)
# best = toolbox.evaluate(bestInd, myNet)
# # fitness = Snake.run_game(myNet, headless=True, display=0, snake_game=Snake.snake(XSIZE, YSIZE))
# display = DisplayGame(XSIZE, YSIZE)
# snake.run_game(myNet,display, headless=False, snake_game=snake.snake(XSIZE, YSIZE))
# print("Best individual fitness：" + str(best))