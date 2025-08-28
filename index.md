---
---

# Selected Class Projects

## TreasureMaze Q-Learning Agent
**Course:** CS-370 — Current & Emerging Trends in Computer Science  
**Tech:** Python, Q-learning, epsilon-greedy •
**Code:** 
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Treasure Hunt Game Notebook\n",
    "\n",
    "## Read and Review Your Starter Code\n",
    "The theme of this project is a popular treasure hunt game in which the player needs to find the treasure before the pirate does. While you will not be developing the entire game, you will write the part of the game that represents the intelligent agent, which is a pirate in this case. The pirate will try to find the optimal path to the treasure using deep Q-learning. \n",
    "\n",
    "You have been provided with two Python classes and this notebook to help you with this assignment. The first class, TreasureMaze.py, represents the environment, which includes a maze object defined as a matrix. The second class, GameExperience.py, stores the episodes – that is, all the states that come in between the initial state and the terminal state. This is later used by the agent for learning by experience, called \"exploration\". This notebook shows how to play a game. Your task is to complete the deep Q-learning implementation for which a skeleton implementation has been provided. The code blocks you will need to complete has #TODO as a header.\n",
    "\n",
    "First, read and review the next few code and instruction blocks to understand the code that you have been given."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "from __future__ import print_function\n",
    "import os, sys, time, datetime, json, random\n",
    "import numpy as np\n",
    "from keras.models import Sequential\n",
    "from keras.layers.core import Dense, Activation\n",
    "from keras.optimizers import SGD , Adam, RMSprop\n",
    "from keras.layers.advanced_activations import PReLU\n",
    "import matplotlib.pyplot as plt\n",
    "from TreasureMaze import TreasureMaze\n",
    "from GameExperience import GameExperience\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following code block contains an 8x8 matrix that will be used as a maze object:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "metadata": {},
   "outputs": [],
   "source": [
    "maze = np.array([\n",
    "    [ 1.,  0.,  1.,  1.,  1.,  1.,  1.,  1.],\n",
    "    [ 1.,  0.,  1.,  1.,  1.,  0.,  1.,  1.],\n",
    "    [ 1.,  1.,  1.,  1.,  0.,  1.,  0.,  1.],\n",
    "    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],\n",
    "    [ 1.,  1.,  0.,  1.,  1.,  1.,  1.,  1.],\n",
    "    [ 1.,  1.,  1.,  0.,  1.,  0.,  0.,  0.],\n",
    "    [ 1.,  1.,  1.,  0.,  1.,  1.,  1.,  1.],\n",
    "    [ 1.,  1.,  1.,  1.,  0.,  1.,  1.,  1.]\n",
    "])"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This helper function allows a visual representation of the maze object:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def show(qmaze):\n",
    "    plt.grid('on')\n",
    "    nrows, ncols = qmaze.maze.shape\n",
    "    ax = plt.gca()\n",
    "    ax.set_xticks(np.arange(0.5, nrows, 1))\n",
    "    ax.set_yticks(np.arange(0.5, ncols, 1))\n",
    "    ax.set_xticklabels([])\n",
    "    ax.set_yticklabels([])\n",
    "    canvas = np.copy(qmaze.maze)\n",
    "    for row,col in qmaze.visited:\n",
    "        canvas[row,col] = 0.6\n",
    "    pirate_row, pirate_col, _ = qmaze.state\n",
    "    canvas[pirate_row, pirate_col] = 0.3   # pirate cell\n",
    "    canvas[nrows-1, ncols-1] = 0.9 # treasure cell\n",
    "    img = plt.imshow(canvas, interpolation='none', cmap='gray')\n",
    "    return img"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The pirate agent can move in four directions: left, right, up, and down. \n",
    "\n",
    "While the agent primarily learns by experience through exploitation, often, the agent can choose to explore the environment to find previously undiscovered paths. This is called \"exploration\" and is defined by epsilon. This value is typically a lower value such as 0.1, which means for every ten attempts, the agent will attempt to learn by experience nine times and will randomly explore a new path one time. You are encouraged to try various values for the exploration factor and see how the algorithm performs."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [],
   "source": [
    "LEFT = 0\n",
    "UP = 1\n",
    "RIGHT = 2\n",
    "DOWN = 3\n",
    "\n",
    "\n",
    "# Exploration factor\n",
    "epsilon = 0.01\n",
    "\n",
    "# Actions dictionary\n",
    "actions_dict = {\n",
    "    LEFT: 'left',\n",
    "    UP: 'up',\n",
    "    RIGHT: 'right',\n",
    "    DOWN: 'down',\n",
    "}\n",
    "\n",
    "num_actions = len(actions_dict)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The sample code block and output below show creating a maze object and performing one action (DOWN), which returns the reward. The resulting updated environment is visualized."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "reward= -0.04\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x24848ea4908>"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAOsAAADrCAYAAACICmHVAAAABHNCSVQICAgIfAhkiAAAAAlwSFlzAAALEgAACxIB0t1+/AAAADh0RVh0U29mdHdhcmUAbWF0cGxvdGxpYiB2ZXJzaW9uMy4xLjEsIGh0dHA6Ly9tYXRwbG90bGliLm9yZy8QZhcZAAAFtElEQVR4nO3dMWpUexjG4W8ugoUJKLmQxlIY+5kFTDpX4gpO5w5kUguuwFZcwJkFzBSW6SwCEkgjamVxbnEVFBJz5yb5Z97j88BUEd6TGX6YNPkmwzAUsPv+uusHAP4bsUIIsUIIsUIIsUIIsUKIe9v84729veHg4OC2nuUX3759q48fPzbZevr0aT148KDJ1tevX0e51XpvrFsfPnyo8/PzyUVf2yrWg4ODevHixc081RU+f/5cXdc12Xr16lUtFosmW6vVapRbrffGujWfzy/9mh+DIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIcRWf+T706dP9e7du9t6ll+0/OPU3IzNZlNHR0dNtvq+b7KzSyZXXT6fTCbPq+p5VdWjR49mL1++bPFctb+/X6enp022ptNp7e3tNdn68uXLKLeqqs7Oznxm19R1Xa3X6/93PmMYhtdV9bqq6uHDh8Pbt29v+PEutlgsmp3P6Pt+lKcYWp/POD4+9pndIr+zQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQgixQoitzmc8efKk2fmM1WpVV10LuMmtsZpMLvzj7rei7/tmn9nx8XGzUx3L5XIn/sj3VuczDg8PZ2/evGnxXKM9M9F66+TkpMlWVduTFi1PdTx+/LgODw+bbP3ufEYNw/CfX7PZbGil73tbN7BVVc1eLb+35XLZ7PtaLpfNvq/vjV3Yn99ZIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYTzGXew1eqkRcuzD1Xj/sxabTmfsWNbNcKzDz++N1vX43wGjIBYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYIYRYq2qz2dRkMmny2mw2W11BuM5rNpvd9VvLDXLrpqrOzs7q9PS0yVbL+zMt38PWe2PdcuvmCsvlcpT3Z1q+h633xrrl1g2MgFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFghhFirajabNT1p0fJUR0utz5CMdesyzmfcwdbJyUmTrZanOqranyEZ41bXdTUMg/MZu7JVIzzVMQztz5CMcevfJJ3PgGhihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRBihRD37voBGI8fZ0haWK1Wo9yaz+eXfs35jDvYGuv5jDF/Zq22uq6r9XrtfMaubNVIz2eM+TNr5XtjzmdAMrFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCLFCCOczRr7V6lRHVdV0Oh3t+3j//v0mW13X1fv37y88n3FlrD+bz+fDer2+sQf7ndVqVYvFwtY1t46OjppsVVX1fT/a93E6nTbZevbs2aWx+jEYQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQogVQmx1PqOqplXV6h7D31V1bitmq/XeWLemwzDsX/SFrc5ntDSZTNbDMMxtZWy13vsTt/wYDCHECiF2OdbXtqK2Wu/9cVs7+zsr8Ktd/p8V+IlYIYRYIYRYIYRYIcQ/8eViVeWzLxQAAAAASUVORK5CYII=\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "qmaze = TreasureMaze(maze)\n",
    "canvas, reward, game_over = qmaze.act(DOWN)\n",
    "print(\"reward=\", reward)\n",
    "show(qmaze)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function simulates a full game based on the provided trained model. The other parameters include the TreasureMaze object and the starting position of the pirate."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "metadata": {},
   "outputs": [],
   "source": [
    "def play_game(model, qmaze, pirate_cell):\n",
    "    qmaze.reset(pirate_cell)\n",
    "    envstate = qmaze.observe()\n",
    "    while True:\n",
    "        prev_envstate = envstate\n",
    "        # get next action\n",
    "        q = model.predict(prev_envstate)\n",
    "        action = np.argmax(q[0])\n",
    "\n",
    "        # apply action, get rewards and new state\n",
    "        envstate, reward, game_status = qmaze.act(action)\n",
    "        if game_status == 'win':\n",
    "            return True\n",
    "        elif game_status == 'lose':\n",
    "            return False"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This function helps you to determine whether the pirate can win any game at all. If your maze is not well designed, the pirate may not win any game at all. In this case, your training would not yield any result. The provided maze in this notebook ensures that there is a path to win and you can run this method to check."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [],
   "source": [
    "def completion_check(model, qmaze):\n",
    "    for cell in qmaze.free_cells:\n",
    "        if not qmaze.valid_actions(cell):\n",
    "            return False\n",
    "        if not play_game(model, qmaze, cell):\n",
    "            return False\n",
    "    return True"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The code you have been given in this block will build the neural network model. Review the code and note the number of layers, as well as the activation, optimizer, and loss functions that are used to train the model."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "metadata": {},
   "outputs": [],
   "source": [
    "def build_model(maze):\n",
    "    model = Sequential()\n",
    "    model.add(Dense(maze.size, input_shape=(maze.size,)))\n",
    "    model.add(PReLU())\n",
    "    model.add(Dense(maze.size))\n",
    "    model.add(PReLU())\n",
    "    model.add(Dense(num_actions))\n",
    "    model.compile(optimizer='adam', loss='mse')\n",
    "    return model"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# #TODO: Complete the Q-Training Algorithm Code Block\n",
    "\n",
    "This is your deep Q-learning implementation. The goal of your deep Q-learning implementation is to find the best possible navigation sequence that results in reaching the treasure cell while maximizing the reward. In your implementation, you need to determine the optimal number of epochs to achieve a 100% win rate.\n",
    "\n",
    "You will need to complete the section starting with #pseudocode. The pseudocode has been included for you."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [],
   "source": [
    "def qtrain(model, maze, **opt):\n",
    "\n",
    "    # exploration factor\n",
    "    global epsilon \n",
    "\n",
    "    # number of epochs\n",
    "    n_epoch = opt.get('n_epoch', 15000)\n",
    "\n",
    "    # maximum memory to store episodes\n",
    "    max_memory = opt.get('max_memory', 1000)\n",
    "\n",
    "    # maximum data size for training\n",
    "    data_size = opt.get('data_size', 50)\n",
    "\n",
    "    # start time\n",
    "    start_time = datetime.datetime.now()\n",
    "\n",
    "    # Construct environment/game from numpy array: maze (see above)\n",
    "    qmaze = TreasureMaze(maze)\n",
    "\n",
    "    # Initialize experience replay object\n",
    "    experience = GameExperience(model, max_memory=max_memory)\n",
    "    \n",
    "    win_history = []   # history of win/lose game\n",
    "    hsize = qmaze.maze.size//2   # history window size\n",
    "    win_rate = 0.0\n",
    "    \n",
    "    # pseudocode:\n",
    "    # For each epoch:\n",
    "\n",
    "    for epoch in range(n_epoch):\n",
    "        agent_cell = random.choice(qmaze.free_cells) # Agent_cell = randomly select a free cell\n",
    "        qmaze.reset(agent_cell) # Reset the maze with agent set to above position\n",
    "        envstate = qmaze.observe()  # envstate = Environment.current_state\n",
    "        n_episodes = 0\n",
    "        loss = 0.0\n",
    "    \n",
    "        while True: # While state is not game over:\n",
    "            previous_envstate = envstate # previous_envstate = envstate\n",
    "                \n",
    "            if np.random.rand() < epsilon:     # Action = randomly choose action (left, right, up, down) either by exploration or by exploitation\n",
    "                action = np.random.choice(qmaze.valid_actions())\n",
    "            else:\n",
    "                q_values = model.predict(previous_envstate.reshape(1, -1))\n",
    "                action = np.argmax(q_values[0])  \n",
    "\n",
    "            envstate, reward, game_status = qmaze.act(action) # envstate, reward, game_status = qmaze.act(action)\n",
    "    \n",
    "            # episode = [previous_envstate, action, reward, envstate, game_status]\n",
    "            # Store episode in Experience replay object\n",
    "            experience.remember([previous_envstate, action, reward, envstate, game_status])\n",
    "            n_episodes += 1\n",
    "    \n",
    "            if game_status == 'win':\n",
    "                win_history.append(1)\n",
    "                break\n",
    "            elif game_status == 'lose':\n",
    "                win_history.append(0)\n",
    "                break\n",
    "                \n",
    "            inputs, targets = experience.get_data(data_size=data_size) # Train neural network model and evaluate loss\n",
    "            loss = model.train_on_batch(inputs, targets)\n",
    "                \n",
    "    # If the win rate is above the threshold and your model passes the completion check, that would be your epoch.\n",
    "        if len(win_history) > hsize:\n",
    "            win_rate = sum(win_history[-hsize:]) / hsize \n",
    "\n",
    "\n",
    "    #Print the epoch, loss, episodes, win count, and win rate for each epoch\n",
    "        dt = datetime.datetime.now() - start_time\n",
    "        t = format_time(dt.total_seconds())\n",
    "        template = \"Epoch: {:03d}/{:d} | Loss: {:.4f} | Episodes: {:d} | Win count: {:d} | Win rate: {:.3f} | time: {}\"\n",
    "        print(template.format(epoch, n_epoch-1, loss, n_episodes, sum(win_history), win_rate, t))\n",
    "        # We simply check if training has exhausted all free cells and if in all\n",
    "        # cases the agent won.\n",
    "        if win_rate > 0.9 : epsilon = 0.05\n",
    "        if sum(win_history[-hsize:]) == hsize and completion_check(model, qmaze):\n",
    "            print(\"Reached 100%% win rate at epoch: %d\" % (epoch,))\n",
    "            break\n",
    "    \n",
    "    \n",
    "    # Determine the total time for training\n",
    "    dt = datetime.datetime.now() - start_time\n",
    "    seconds = dt.total_seconds()\n",
    "    t = format_time(seconds)\n",
    "\n",
    "    print(\"n_epoch: %d, max_mem: %d, data: %d, time: %s\" % (epoch, max_memory, data_size, t))\n",
    "    return seconds\n",
    "\n",
    "# This is a small utility for printing readable time strings:\n",
    "def format_time(seconds):\n",
    "    if seconds < 400:\n",
    "        s = float(seconds)\n",
    "        return \"%.1f seconds\" % (s,)\n",
    "    elif seconds < 4000:\n",
    "        m = seconds / 60.0\n",
    "        return \"%.2f minutes\" % (m,)\n",
    "    else:\n",
    "        h = seconds / 3600.0\n",
    "        return \"%.2f hours\" % (h,)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Test Your Model\n",
    "\n",
    "Now we will start testing the deep Q-learning implementation. To begin, select **Cell**, then **Run All** from the menu bar. This will run your notebook. As it runs, you should see output begin to appear beneath the next few cells. The code below creates an instance of TreasureMaze."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x24848f04348>"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "qmaze = TreasureMaze(maze)\n",
    "show(qmaze)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In the next code block, you will build your model and train it using deep Q-learning. Note: This step takes several minutes to fully run."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch: 000/14999 | Loss: 0.0040 | Episodes: 2 | Win count: 1 | Win rate: 0.000 | time: 1.7 seconds\n",
      "Epoch: 001/14999 | Loss: 0.0012 | Episodes: 137 | Win count: 1 | Win rate: 0.000 | time: 13.2 seconds\n",
      "Epoch: 002/14999 | Loss: 0.0009 | Episodes: 134 | Win count: 1 | Win rate: 0.000 | time: 26.4 seconds\n",
      "Epoch: 003/14999 | Loss: 0.0004 | Episodes: 132 | Win count: 1 | Win rate: 0.000 | time: 39.0 seconds\n",
      "Epoch: 004/14999 | Loss: 0.0007 | Episodes: 52 | Win count: 2 | Win rate: 0.000 | time: 43.5 seconds\n",
      "Epoch: 005/14999 | Loss: 0.0003 | Episodes: 134 | Win count: 2 | Win rate: 0.000 | time: 56.7 seconds\n",
      "Epoch: 006/14999 | Loss: 0.0007 | Episodes: 134 | Win count: 2 | Win rate: 0.000 | time: 68.8 seconds\n",
      "Epoch: 007/14999 | Loss: 0.0008 | Episodes: 146 | Win count: 2 | Win rate: 0.000 | time: 81.5 seconds\n",
      "Epoch: 008/14999 | Loss: 0.0072 | Episodes: 20 | Win count: 3 | Win rate: 0.000 | time: 83.0 seconds\n",
      "Epoch: 009/14999 | Loss: 0.0065 | Episodes: 140 | Win count: 3 | Win rate: 0.000 | time: 94.6 seconds\n",
      "Epoch: 010/14999 | Loss: 0.0010 | Episodes: 6 | Win count: 4 | Win rate: 0.000 | time: 95.0 seconds\n",
      "Epoch: 011/14999 | Loss: 0.0013 | Episodes: 2 | Win count: 5 | Win rate: 0.000 | time: 95.1 seconds\n",
      "Epoch: 012/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 6 | Win rate: 0.000 | time: 95.1 seconds\n",
      "Epoch: 013/14999 | Loss: 0.0020 | Episodes: 3 | Win count: 7 | Win rate: 0.000 | time: 95.3 seconds\n",
      "Epoch: 014/14999 | Loss: 0.0060 | Episodes: 83 | Win count: 8 | Win rate: 0.000 | time: 102.3 seconds\n",
      "Epoch: 015/14999 | Loss: 0.0011 | Episodes: 139 | Win count: 8 | Win rate: 0.000 | time: 114.3 seconds\n",
      "Epoch: 016/14999 | Loss: 0.0011 | Episodes: 26 | Win count: 9 | Win rate: 0.000 | time: 116.6 seconds\n",
      "Epoch: 017/14999 | Loss: 0.0118 | Episodes: 137 | Win count: 9 | Win rate: 0.000 | time: 129.2 seconds\n",
      "Epoch: 018/14999 | Loss: 0.0008 | Episodes: 134 | Win count: 9 | Win rate: 0.000 | time: 140.6 seconds\n",
      "Epoch: 019/14999 | Loss: 0.0001 | Episodes: 139 | Win count: 9 | Win rate: 0.000 | time: 151.2 seconds\n",
      "Epoch: 020/14999 | Loss: 0.0003 | Episodes: 13 | Win count: 10 | Win rate: 0.000 | time: 152.1 seconds\n",
      "Epoch: 021/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 11 | Win rate: 0.000 | time: 152.1 seconds\n",
      "Epoch: 022/14999 | Loss: 0.0005 | Episodes: 6 | Win count: 12 | Win rate: 0.000 | time: 152.6 seconds\n",
      "Epoch: 023/14999 | Loss: 0.0009 | Episodes: 43 | Win count: 13 | Win rate: 0.000 | time: 156.0 seconds\n",
      "Epoch: 024/14999 | Loss: 0.0007 | Episodes: 135 | Win count: 13 | Win rate: 0.000 | time: 168.7 seconds\n",
      "Epoch: 025/14999 | Loss: 0.0006 | Episodes: 140 | Win count: 13 | Win rate: 0.000 | time: 182.0 seconds\n",
      "Epoch: 026/14999 | Loss: 0.0012 | Episodes: 144 | Win count: 13 | Win rate: 0.000 | time: 195.0 seconds\n",
      "Epoch: 027/14999 | Loss: 0.0004 | Episodes: 36 | Win count: 14 | Win rate: 0.000 | time: 198.4 seconds\n",
      "Epoch: 028/14999 | Loss: 0.0014 | Episodes: 24 | Win count: 15 | Win rate: 0.000 | time: 200.8 seconds\n",
      "Epoch: 029/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 16 | Win rate: 0.000 | time: 200.8 seconds\n",
      "Epoch: 030/14999 | Loss: 0.0014 | Episodes: 17 | Win count: 17 | Win rate: 0.000 | time: 202.6 seconds\n",
      "Epoch: 031/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 18 | Win rate: 0.000 | time: 202.6 seconds\n",
      "Epoch: 032/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 19 | Win rate: 0.562 | time: 202.6 seconds\n",
      "Epoch: 033/14999 | Loss: 0.0015 | Episodes: 144 | Win count: 19 | Win rate: 0.562 | time: 215.6 seconds\n",
      "Epoch: 034/14999 | Loss: 0.0024 | Episodes: 139 | Win count: 19 | Win rate: 0.562 | time: 227.6 seconds\n",
      "Epoch: 035/14999 | Loss: 0.0009 | Episodes: 138 | Win count: 19 | Win rate: 0.562 | time: 239.7 seconds\n",
      "Epoch: 036/14999 | Loss: 0.0008 | Episodes: 140 | Win count: 19 | Win rate: 0.531 | time: 253.1 seconds\n",
      "Epoch: 037/14999 | Loss: 0.0008 | Episodes: 139 | Win count: 19 | Win rate: 0.531 | time: 265.5 seconds\n",
      "Epoch: 038/14999 | Loss: 0.0010 | Episodes: 94 | Win count: 20 | Win rate: 0.562 | time: 273.4 seconds\n",
      "Epoch: 039/14999 | Loss: 0.0010 | Episodes: 22 | Win count: 21 | Win rate: 0.594 | time: 275.2 seconds\n",
      "Epoch: 040/14999 | Loss: 0.0005 | Episodes: 18 | Win count: 22 | Win rate: 0.594 | time: 276.7 seconds\n",
      "Epoch: 041/14999 | Loss: 0.0008 | Episodes: 14 | Win count: 23 | Win rate: 0.625 | time: 277.9 seconds\n",
      "Epoch: 042/14999 | Loss: 0.0007 | Episodes: 139 | Win count: 23 | Win rate: 0.594 | time: 290.6 seconds\n",
      "Epoch: 043/14999 | Loss: 0.0008 | Episodes: 11 | Win count: 24 | Win rate: 0.594 | time: 291.6 seconds\n",
      "Epoch: 044/14999 | Loss: 0.0011 | Episodes: 141 | Win count: 24 | Win rate: 0.562 | time: 303.1 seconds\n",
      "Epoch: 045/14999 | Loss: 0.0007 | Episodes: 134 | Win count: 24 | Win rate: 0.531 | time: 314.1 seconds\n",
      "Epoch: 046/14999 | Loss: 0.0009 | Episodes: 12 | Win count: 25 | Win rate: 0.531 | time: 315.0 seconds\n",
      "Epoch: 047/14999 | Loss: 0.0003 | Episodes: 139 | Win count: 25 | Win rate: 0.531 | time: 327.3 seconds\n",
      "Epoch: 048/14999 | Loss: 0.0004 | Episodes: 138 | Win count: 25 | Win rate: 0.500 | time: 337.4 seconds\n",
      "Epoch: 049/14999 | Loss: 0.0007 | Episodes: 143 | Win count: 25 | Win rate: 0.500 | time: 350.8 seconds\n",
      "Epoch: 050/14999 | Loss: 0.0008 | Episodes: 12 | Win count: 26 | Win rate: 0.531 | time: 351.9 seconds\n",
      "Epoch: 051/14999 | Loss: 0.0007 | Episodes: 138 | Win count: 26 | Win rate: 0.531 | time: 364.0 seconds\n",
      "Epoch: 052/14999 | Loss: 0.0006 | Episodes: 12 | Win count: 27 | Win rate: 0.531 | time: 365.0 seconds\n",
      "Epoch: 053/14999 | Loss: 0.0010 | Episodes: 139 | Win count: 27 | Win rate: 0.500 | time: 376.9 seconds\n",
      "Epoch: 054/14999 | Loss: 0.0017 | Episodes: 38 | Win count: 28 | Win rate: 0.500 | time: 379.9 seconds\n",
      "Epoch: 055/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 29 | Win rate: 0.500 | time: 379.9 seconds\n",
      "Epoch: 056/14999 | Loss: 0.0009 | Episodes: 144 | Win count: 29 | Win rate: 0.500 | time: 392.2 seconds\n",
      "Epoch: 057/14999 | Loss: 0.0019 | Episodes: 9 | Win count: 30 | Win rate: 0.531 | time: 392.9 seconds\n",
      "Epoch: 058/14999 | Loss: 0.0021 | Episodes: 20 | Win count: 31 | Win rate: 0.562 | time: 394.8 seconds\n",
      "Epoch: 059/14999 | Loss: 0.0104 | Episodes: 115 | Win count: 32 | Win rate: 0.562 | time: 6.74 minutes\n",
      "Epoch: 060/14999 | Loss: 0.0019 | Episodes: 22 | Win count: 33 | Win rate: 0.562 | time: 6.77 minutes\n",
      "Epoch: 061/14999 | Loss: 0.0013 | Episodes: 144 | Win count: 33 | Win rate: 0.531 | time: 6.99 minutes\n",
      "Epoch: 062/14999 | Loss: 0.0016 | Episodes: 49 | Win count: 34 | Win rate: 0.531 | time: 7.06 minutes\n",
      "Epoch: 063/14999 | Loss: 0.0014 | Episodes: 10 | Win count: 35 | Win rate: 0.531 | time: 7.07 minutes\n",
      "Epoch: 064/14999 | Loss: 0.0015 | Episodes: 35 | Win count: 36 | Win rate: 0.531 | time: 7.13 minutes\n",
      "Epoch: 065/14999 | Loss: 0.0014 | Episodes: 17 | Win count: 37 | Win rate: 0.562 | time: 7.15 minutes\n",
      "Epoch: 066/14999 | Loss: 0.0011 | Episodes: 6 | Win count: 38 | Win rate: 0.594 | time: 7.16 minutes\n",
      "Epoch: 067/14999 | Loss: 0.0010 | Episodes: 36 | Win count: 39 | Win rate: 0.625 | time: 7.21 minutes\n",
      "Epoch: 068/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 40 | Win rate: 0.656 | time: 7.21 minutes\n",
      "Epoch: 069/14999 | Loss: 0.0018 | Episodes: 35 | Win count: 41 | Win rate: 0.688 | time: 7.25 minutes\n",
      "Epoch: 070/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 42 | Win rate: 0.688 | time: 7.25 minutes\n",
      "Epoch: 071/14999 | Loss: 0.0016 | Episodes: 41 | Win count: 43 | Win rate: 0.688 | time: 7.32 minutes\n",
      "Epoch: 072/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 44 | Win rate: 0.688 | time: 7.32 minutes\n",
      "Epoch: 073/14999 | Loss: 0.0009 | Episodes: 65 | Win count: 45 | Win rate: 0.688 | time: 7.40 minutes\n",
      "Epoch: 074/14999 | Loss: 0.0015 | Episodes: 16 | Win count: 46 | Win rate: 0.719 | time: 7.42 minutes\n",
      "Epoch: 075/14999 | Loss: 0.0012 | Episodes: 9 | Win count: 47 | Win rate: 0.719 | time: 7.43 minutes\n",
      "Epoch: 076/14999 | Loss: 0.0009 | Episodes: 54 | Win count: 48 | Win rate: 0.750 | time: 7.51 minutes\n",
      "Epoch: 077/14999 | Loss: 0.0011 | Episodes: 5 | Win count: 49 | Win rate: 0.781 | time: 7.51 minutes\n",
      "Epoch: 078/14999 | Loss: 0.0007 | Episodes: 7 | Win count: 50 | Win rate: 0.781 | time: 7.52 minutes\n",
      "Epoch: 079/14999 | Loss: 0.0017 | Episodes: 16 | Win count: 51 | Win rate: 0.812 | time: 7.54 minutes\n",
      "Epoch: 080/14999 | Loss: 0.0010 | Episodes: 34 | Win count: 52 | Win rate: 0.844 | time: 7.59 minutes\n",
      "Epoch: 081/14999 | Loss: 0.0012 | Episodes: 24 | Win count: 53 | Win rate: 0.875 | time: 7.62 minutes\n",
      "Epoch: 082/14999 | Loss: 0.0009 | Episodes: 132 | Win count: 53 | Win rate: 0.844 | time: 7.85 minutes\n",
      "Epoch: 083/14999 | Loss: 0.0010 | Episodes: 2 | Win count: 54 | Win rate: 0.875 | time: 7.85 minutes\n",
      "Epoch: 084/14999 | Loss: 0.0009 | Episodes: 15 | Win count: 55 | Win rate: 0.875 | time: 7.87 minutes\n",
      "Epoch: 085/14999 | Loss: 0.0015 | Episodes: 7 | Win count: 56 | Win rate: 0.906 | time: 7.88 minutes\n",
      "Epoch: 086/14999 | Loss: 0.0011 | Episodes: 35 | Win count: 57 | Win rate: 0.906 | time: 7.93 minutes\n",
      "Epoch: 087/14999 | Loss: 0.0014 | Episodes: 5 | Win count: 58 | Win rate: 0.906 | time: 7.94 minutes\n",
      "Epoch: 088/14999 | Loss: 0.0010 | Episodes: 15 | Win count: 59 | Win rate: 0.938 | time: 7.96 minutes\n",
      "Epoch: 089/14999 | Loss: 0.0019 | Episodes: 135 | Win count: 59 | Win rate: 0.906 | time: 8.15 minutes\n",
      "Epoch: 090/14999 | Loss: 0.0006 | Episodes: 108 | Win count: 60 | Win rate: 0.906 | time: 8.32 minutes\n",
      "Epoch: 091/14999 | Loss: 0.0007 | Episodes: 91 | Win count: 61 | Win rate: 0.906 | time: 8.45 minutes\n",
      "Epoch: 092/14999 | Loss: 0.0010 | Episodes: 51 | Win count: 62 | Win rate: 0.906 | time: 8.53 minutes\n",
      "Epoch: 093/14999 | Loss: 0.0007 | Episodes: 6 | Win count: 63 | Win rate: 0.938 | time: 8.54 minutes\n",
      "Epoch: 094/14999 | Loss: 0.0015 | Episodes: 28 | Win count: 64 | Win rate: 0.938 | time: 8.58 minutes\n",
      "Epoch: 095/14999 | Loss: 0.0010 | Episodes: 4 | Win count: 65 | Win rate: 0.938 | time: 8.58 minutes\n",
      "Epoch: 096/14999 | Loss: 0.0010 | Episodes: 23 | Win count: 66 | Win rate: 0.938 | time: 8.62 minutes\n",
      "Epoch: 097/14999 | Loss: 0.0007 | Episodes: 9 | Win count: 67 | Win rate: 0.938 | time: 8.63 minutes\n",
      "Epoch: 098/14999 | Loss: 0.0011 | Episodes: 27 | Win count: 68 | Win rate: 0.938 | time: 8.66 minutes\n",
      "Epoch: 099/14999 | Loss: 0.0012 | Episodes: 24 | Win count: 69 | Win rate: 0.938 | time: 8.70 minutes\n",
      "Epoch: 100/14999 | Loss: 0.0010 | Episodes: 15 | Win count: 70 | Win rate: 0.938 | time: 8.72 minutes\n",
      "Epoch: 101/14999 | Loss: 0.0006 | Episodes: 8 | Win count: 71 | Win rate: 0.938 | time: 8.73 minutes\n",
      "Epoch: 102/14999 | Loss: 0.0011 | Episodes: 25 | Win count: 72 | Win rate: 0.938 | time: 8.77 minutes\n",
      "Epoch: 103/14999 | Loss: 0.0012 | Episodes: 5 | Win count: 73 | Win rate: 0.938 | time: 8.77 minutes\n",
      "Epoch: 104/14999 | Loss: 0.0008 | Episodes: 30 | Win count: 74 | Win rate: 0.938 | time: 8.81 minutes\n",
      "Epoch: 105/14999 | Loss: 0.0018 | Episodes: 39 | Win count: 75 | Win rate: 0.938 | time: 8.86 minutes\n",
      "Epoch: 106/14999 | Loss: 0.0011 | Episodes: 19 | Win count: 76 | Win rate: 0.938 | time: 8.89 minutes\n",
      "Epoch: 107/14999 | Loss: 0.0006 | Episodes: 4 | Win count: 77 | Win rate: 0.938 | time: 8.89 minutes\n",
      "Epoch: 108/14999 | Loss: 0.0010 | Episodes: 22 | Win count: 78 | Win rate: 0.938 | time: 8.92 minutes\n",
      "Epoch: 109/14999 | Loss: 0.0013 | Episodes: 26 | Win count: 79 | Win rate: 0.938 | time: 8.95 minutes\n",
      "Epoch: 110/14999 | Loss: 0.0013 | Episodes: 18 | Win count: 80 | Win rate: 0.938 | time: 8.97 minutes\n",
      "Epoch: 111/14999 | Loss: 0.0007 | Episodes: 10 | Win count: 81 | Win rate: 0.938 | time: 8.98 minutes\n",
      "Epoch: 112/14999 | Loss: 0.0009 | Episodes: 13 | Win count: 82 | Win rate: 0.938 | time: 9.00 minutes\n",
      "Epoch: 113/14999 | Loss: 0.0005 | Episodes: 22 | Win count: 83 | Win rate: 0.938 | time: 9.03 minutes\n",
      "Epoch: 114/14999 | Loss: 0.0011 | Episodes: 64 | Win count: 84 | Win rate: 0.969 | time: 9.12 minutes\n",
      "Epoch: 115/14999 | Loss: 0.0016 | Episodes: 27 | Win count: 85 | Win rate: 0.969 | time: 9.15 minutes\n",
      "Epoch: 116/14999 | Loss: 0.0003 | Episodes: 28 | Win count: 86 | Win rate: 0.969 | time: 9.19 minutes\n",
      "Epoch: 117/14999 | Loss: 0.0010 | Episodes: 5 | Win count: 87 | Win rate: 0.969 | time: 9.20 minutes\n",
      "Epoch: 118/14999 | Loss: 0.0015 | Episodes: 39 | Win count: 88 | Win rate: 0.969 | time: 9.26 minutes\n",
      "Epoch: 119/14999 | Loss: 0.0015 | Episodes: 17 | Win count: 89 | Win rate: 0.969 | time: 9.28 minutes\n",
      "Epoch: 120/14999 | Loss: 0.0013 | Episodes: 21 | Win count: 90 | Win rate: 0.969 | time: 9.31 minutes\n",
      "Epoch: 121/14999 | Loss: 0.0009 | Episodes: 36 | Win count: 91 | Win rate: 1.000 | time: 9.36 minutes\n",
      "Epoch: 122/14999 | Loss: 0.0009 | Episodes: 50 | Win count: 92 | Win rate: 1.000 | time: 9.44 minutes\n",
      "Epoch: 123/14999 | Loss: 0.0012 | Episodes: 17 | Win count: 93 | Win rate: 1.000 | time: 9.46 minutes\n",
      "Epoch: 124/14999 | Loss: 0.0022 | Episodes: 53 | Win count: 94 | Win rate: 1.000 | time: 9.54 minutes\n",
      "Epoch: 125/14999 | Loss: 0.0010 | Episodes: 18 | Win count: 95 | Win rate: 1.000 | time: 9.57 minutes\n",
      "Epoch: 126/14999 | Loss: 0.0014 | Episodes: 48 | Win count: 96 | Win rate: 1.000 | time: 9.64 minutes\n",
      "Epoch: 127/14999 | Loss: 0.0014 | Episodes: 11 | Win count: 97 | Win rate: 1.000 | time: 9.66 minutes\n",
      "Epoch: 128/14999 | Loss: 0.0016 | Episodes: 29 | Win count: 98 | Win rate: 1.000 | time: 9.71 minutes\n",
      "Epoch: 129/14999 | Loss: 0.0020 | Episodes: 8 | Win count: 99 | Win rate: 1.000 | time: 9.72 minutes\n",
      "Epoch: 130/14999 | Loss: 0.0014 | Episodes: 6 | Win count: 100 | Win rate: 1.000 | time: 9.73 minutes\n",
      "Epoch: 131/14999 | Loss: 0.0011 | Episodes: 26 | Win count: 101 | Win rate: 1.000 | time: 9.77 minutes\n",
      "Epoch: 132/14999 | Loss: 0.0016 | Episodes: 47 | Win count: 102 | Win rate: 1.000 | time: 9.85 minutes\n",
      "Epoch: 133/14999 | Loss: 0.0017 | Episodes: 8 | Win count: 103 | Win rate: 1.000 | time: 9.86 minutes\n",
      "Epoch: 134/14999 | Loss: 0.0000 | Episodes: 1 | Win count: 104 | Win rate: 1.000 | time: 9.86 minutes\n",
      "Epoch: 135/14999 | Loss: 0.0012 | Episodes: 84 | Win count: 105 | Win rate: 1.000 | time: 10.00 minutes\n",
      "Epoch: 136/14999 | Loss: 0.0011 | Episodes: 4 | Win count: 106 | Win rate: 1.000 | time: 10.01 minutes\n",
      "Epoch: 137/14999 | Loss: 0.0013 | Episodes: 17 | Win count: 107 | Win rate: 1.000 | time: 10.03 minutes\n",
      "Epoch: 138/14999 | Loss: 0.0011 | Episodes: 21 | Win count: 108 | Win rate: 1.000 | time: 10.06 minutes\n",
      "Epoch: 139/14999 | Loss: 0.0009 | Episodes: 3 | Win count: 109 | Win rate: 1.000 | time: 10.07 minutes\n",
      "Epoch: 140/14999 | Loss: 0.0011 | Episodes: 27 | Win count: 110 | Win rate: 1.000 | time: 10.11 minutes\n",
      "Epoch: 141/14999 | Loss: 0.0016 | Episodes: 70 | Win count: 111 | Win rate: 1.000 | time: 10.23 minutes\n",
      "Epoch: 142/14999 | Loss: 0.0015 | Episodes: 13 | Win count: 112 | Win rate: 1.000 | time: 10.25 minutes\n",
      "Epoch: 143/14999 | Loss: 0.0014 | Episodes: 4 | Win count: 113 | Win rate: 1.000 | time: 10.26 minutes\n",
      "Epoch: 144/14999 | Loss: 0.0009 | Episodes: 2 | Win count: 114 | Win rate: 1.000 | time: 10.27 minutes\n",
      "Epoch: 145/14999 | Loss: 0.0012 | Episodes: 32 | Win count: 115 | Win rate: 1.000 | time: 10.32 minutes\n",
      "Epoch: 146/14999 | Loss: 0.0013 | Episodes: 14 | Win count: 116 | Win rate: 1.000 | time: 10.35 minutes\n",
      "Epoch: 147/14999 | Loss: 0.0011 | Episodes: 11 | Win count: 117 | Win rate: 1.000 | time: 10.37 minutes\n",
      "Epoch: 148/14999 | Loss: 0.0012 | Episodes: 8 | Win count: 118 | Win rate: 1.000 | time: 10.39 minutes\n",
      "Epoch: 149/14999 | Loss: 0.0027 | Episodes: 10 | Win count: 119 | Win rate: 1.000 | time: 10.42 minutes\n",
      "Epoch: 150/14999 | Loss: 0.0006 | Episodes: 21 | Win count: 120 | Win rate: 1.000 | time: 10.45 minutes\n",
      "Epoch: 151/14999 | Loss: 0.0015 | Episodes: 16 | Win count: 121 | Win rate: 1.000 | time: 10.47 minutes\n",
      "Epoch: 152/14999 | Loss: 0.0010 | Episodes: 30 | Win count: 122 | Win rate: 1.000 | time: 10.52 minutes\n",
      "Epoch: 153/14999 | Loss: 0.0015 | Episodes: 87 | Win count: 123 | Win rate: 1.000 | time: 10.66 minutes\n",
      "Epoch: 154/14999 | Loss: 0.0009 | Episodes: 4 | Win count: 124 | Win rate: 1.000 | time: 10.67 minutes\n",
      "Epoch: 155/14999 | Loss: 0.0011 | Episodes: 107 | Win count: 125 | Win rate: 1.000 | time: 10.83 minutes\n",
      "Epoch: 156/14999 | Loss: 0.0009 | Episodes: 30 | Win count: 126 | Win rate: 1.000 | time: 10.88 minutes\n",
      "Epoch: 157/14999 | Loss: 0.0012 | Episodes: 44 | Win count: 127 | Win rate: 1.000 | time: 10.95 minutes\n",
      "Epoch: 158/14999 | Loss: 0.0008 | Episodes: 45 | Win count: 128 | Win rate: 1.000 | time: 11.02 minutes\n",
      "Epoch: 159/14999 | Loss: 0.0011 | Episodes: 21 | Win count: 129 | Win rate: 1.000 | time: 11.05 minutes\n",
      "Epoch: 160/14999 | Loss: 0.0017 | Episodes: 29 | Win count: 130 | Win rate: 1.000 | time: 11.09 minutes\n",
      "Epoch: 161/14999 | Loss: 0.0011 | Episodes: 62 | Win count: 131 | Win rate: 1.000 | time: 11.20 minutes\n",
      "Epoch: 162/14999 | Loss: 0.0008 | Episodes: 39 | Win count: 132 | Win rate: 1.000 | time: 11.26 minutes\n",
      "Epoch: 163/14999 | Loss: 0.0007 | Episodes: 18 | Win count: 133 | Win rate: 1.000 | time: 11.30 minutes\n",
      "Epoch: 164/14999 | Loss: 0.0006 | Episodes: 13 | Win count: 134 | Win rate: 1.000 | time: 11.32 minutes\n",
      "Epoch: 165/14999 | Loss: 0.0006 | Episodes: 11 | Win count: 135 | Win rate: 1.000 | time: 11.34 minutes\n",
      "Reached 100% win rate at epoch: 165\n",
      "n_epoch: 165, max_mem: 512, data: 32, time: 11.35 minutes\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "681.021488"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "model = build_model(maze)\n",
    "qtrain(model, maze, epochs=1000, max_memory=8*maze.size, data_size=32)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This cell will check to see if the model passes the completion check. Note: This could take several minutes."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x2484b52fcc8>"
      ]
     },
     "execution_count": 23,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "completion_check(model, qmaze)\n",
    "show(qmaze)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "This cell will test your model for one game. It will start the pirate at the top-left corner and run play_game. The agent should find a path from the starting position to the target (treasure). The treasure is located in the bottom-right corner."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<matplotlib.image.AxesImage at 0x2484b559948>"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "pirate_start = (0, 0)\n",
    "play_game(model, qmaze, pirate_start)\n",
    "show(qmaze)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Save and Submit Your Work\n",
    "After you have finished creating the code for your notebook, save your work. Make sure that your notebook contains your name in the filename (e.g. Doe_Jane_ProjectTwo.ipynb). This will help your instructor access and grade your work easily. Download a copy of your IPYNB file and submit it to Brightspace. Refer to the Jupyter Notebook in Apporto Tutorial if you need help with these tasks."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}

**What I built**
- Implemented the **Q-training algorithm** that learns from agent–environment interactions
- Added **epsilon-greedy** to balance exploration vs. exploitation
- Integrated with provided environment classes (**TreasureMaze**, **GameExperience**) that handled maze mechanics and experience storage

**Why it matters**
- Demonstrates practical **reinforcement learning**: algorithm design, policy control, and integrating with an existing environment

<details markdown="1">
<summary>Reflection (course integration & ethics)</summary>

- **What code were you given? What code did you create yourself?**<br>
  In this project I was provided with foundational code that included the environment setup being the maze structure for the TreasureMaze and GameExperience classes. These classes handled the basic mechanics of the game environment such as how the agent interacts with the maze and how experiences are stored for later use. I provided the code responsible for the Q-training algorithm that defined how the agent learned from its interactions with the environment and updating it's memory. I also integrated the epsilon-greedy strategy to balance exploration and exploitation so the agent could learn how to navigate the maze to find the treasure.<br>

- **What do computer scientists do and why does it matter?**<br>
  Computer scientists are problem solvers who use computational methods to find efficient solutions to complex problems. This involves not just coding but also understanding the underlying principles of computation, data management, and software design. It matters because they drive innovation by developing new algorithms, software, and technologies that push the boundaries of what is possible in fields like in this case artificial intelligence.

- **How do I approach a problem as a computer scientist?**<br>
  When approaching a problem as a computer scientist I try to follow a systematic process to ensure that the solution is both effective and efficient. This consists of understaning the clients requirements, planning out the design of the project, and then implementing a iterative product that has been throughly tested. If any feedback is given i try to go back and incorporate those changes.

- **What are my ethical responsibilities to the end user and the organization?**<br>
  My ethical responsibilties to end users and the orginization would be to use secure and transparent methods for processing data. If data isn’t handled responsibly, either the company itself could exploit its customers, or a bad actor could cause a breach. To address this, I would advocate for complete transparency with users about their data, provide an easy and effective way for them to opt out of data usage, and employ robust security measures to store and encrypt user information.
</details>


---
