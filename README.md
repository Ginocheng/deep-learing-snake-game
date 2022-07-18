## This project belongs to Cheng Ding 
# deep-learing-snake-game
This project is to discuss about training an AI to play snake game
# Introduction
Snake is an arcade game that originated in 1976. In the game, the player
controls a long, thin, straight line (commonly known as a snake) that keeps
moving, and the player can only control the direction of the snake's head (up
and down, left and right), picking up objects (or "food") along the way, and
avoiding touching themselves or other obstacles. Each time the snake eats a
piece of food, its body grows a little, making the game progressively more
difficult[1]. Now we need to let the snake evolve itself in some way to complete
the game. Unlike previous games, the food has to evolve along with the snake.
For multi-objective optimisation problems, it is best to use evolutionary
algorithms[2]. The adaptive and learning capabilities of neural networks are well
suited to this problem, acting as directional controllers for the snake, as well as
controllers for the food. For competition between two agents, we can consider
co-evolutionary approaches to accomplish this[3].

# Method

## Reproducible methods and design

Analysis: The goals of the snake can be simply classified as follows.
1. Survive longer (not hitting its head on the wall or its own body)
2. Get a higher score (eat more food)
3. Take less time
To start by discussing the first two, the snake needs to have obstacle avoidance
to live longer and the learning function of the neural network can help the snake
to do this. The third goal refers to the snake's desire to eat its food each time it
is generated with as little walking as possible.
Consider a situation where a snake may keep turning in circles without actively
seeking food. In this case, the snake will not end the game or make the score
higher, so I also need to add a penalty to this section if the snake goes too far
and the score does not increase. Again, we can do this by modifying the Fitness
algorithm.
The goal of the food is to make it more difficult for him to be eaten by the snake
each time he places it. In other words, even if the snake can eat it, it will take
longer. As an example, if the food is generated in front of the snake's head
every time, it will be easily eaten. With the neural network, the food gets the
coordinates of the place where it should be placed next time. Through training,
a neural network can be selected that is more in line with the requirements.

## Quality of the design of the algorithm
After inputting the appropriate parameters, the neural network will output the
value we require (which may not be so intelligent at first). After training and
learning, better outputs can be obtained[4]. So a genetic algorithm is used,
where I take the weights of each neural network as individuals and use the
algorithm to 'reproduce', 'select', 'mutate', and 'crossover' to get the next
generation of weights that better match our goals[5]. The flow chart of the whole
process is as follows
So for the parameter selection of the input layer of the neural network(use for
snake). Considering that the parameters are to be strongly correlated with the
goals, I chose 10 parameters: x and y coordinates of the snakehead, snake tail
and food, length of the snake, distance from the food to the snakehead,
distance from the snake tail to the snakehead and distance from the wall to the
snakehead.
The whole neural network is divided into 4 layers: input layer, hidden layer 1,
hidden layer 2 and output layer. The number of nodes is 10, 6, 6 and 4
respectively.
The output layer takes four outputs and converts them into probabilities in four
directions using the softmax function[6]. I defined the 0th to 3rd results of the
output layer, in order, as left, right, top and bottom, and selected the one with
the highest probability among the four results as the next direction of the
snake's movement.
For the neural network construction of the food, I chose to use two input nodes,
the first hidden layer with four nodes, the second hidden layer with four nodes
and the output layer with two nodes corresponding to the x and y coordinates
of the food. The activation function is sigmoid, which takes values from 0 to 1.
I project it uniformly onto the axes. x and y both take values from 1 to 14.
In the genetic algorithm, the evaluate function is set to evaluate the fitness of
an individual and FitnessMax is created to maximise the fitness of the target,
selecting the best individual from the randomly selected individuals in tournsize.

![image](https://user-images.githubusercontent.com/39352544/179615155-98744472-e506-428f-9876-a7665d10f787.png)
