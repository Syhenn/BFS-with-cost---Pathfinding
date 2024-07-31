"""
Created on Wed Jul 30 14:12:35 2024

@author: Syhenn
"""


import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import heapq
import random
#----------------------# BFS avec coût #---------------------------------#
def bfs_with_cost(grid, start, goal):
    rows, cols = len(grid.grid), len(grid.grid[0])
    queue = [(0, start)]  
    visited = set()
    visited.add(start)
    parent = {start: None}
    cost_so_far = {start: 0} #Cout cumulé
    explored = set()  # Pour stocker les cellules explorées
    
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current_cost, current = heapq.heappop(queue) #Supp et retourne l'elem le plus bas de la file
        explored.add(current)
        if current == goal:
            break
        for direction in directions:
            neighbor = (current[0] + direction[0], current[1] + direction[1])
            if grid.is_valid_cell(neighbor[0], neighbor[1]):
                new_cost = current_cost + grid.costs[neighbor[0]][neighbor[1]]
                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    heapq.heappush(queue, (new_cost, neighbor))
                    visited.add(neighbor)
                    parent[neighbor] = current

    path = []
    step = goal
    while step is not None:
        path.append(step)
        step = parent[step]
    path.reverse()
    return path, cost_so_far[goal], explored


#--------------------# Grid et visualisation #---------------------------#
class Grid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.grid = [[0 for _ in range(cols)] for _ in range(rows)]
        self.costs = [[1 for _ in range(cols)] for _ in range(rows)]

    def set_obstacle(self, row, col):
        self.grid[row][col] = 1

    def set_cost(self, row, col, cost):
        if self.grid[row][col] == 0:
            self.costs[row][col] = cost

    def is_valid_cell(self, row, col):
        return 0 <= row < self.rows and 0 <= col < self.cols and self.grid[row][col] == 0

    def plot(self, path=None, start=None, goal=None, explored=None):
        grid = np.array(self.grid)
        display_grid = np.copy(grid)
        if explored:
            for (x, y) in explored:
                display_grid[x][y] = 5
        if path:
            for (x, y) in path:
                display_grid[x][y] = 2
        if start:
            display_grid[start[0]][start[1]] = 3
        if goal:
            display_grid[goal[0]][goal[1]] = 4

        cmap = plt.cm.colors.ListedColormap(['white', 'black', 'blue', 'green', 'red', 'orange'])
        bounds = [0, 1, 2, 3, 4, 5, 6]
        norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
        plt.imshow(display_grid, cmap=cmap, norm=norm)
        for i in range(self.rows):
            for j in range(self.cols):
                plt.text(j, i, f'{self.costs[i][j]}', ha='center', va='center', color='black')
        plt.grid(which='both', color='black', linestyle='-', linewidth=2)
        plt.xticks([])
        plt.yticks([])
        plt.show()

def generate_random_grid(min_size=5, max_size=40, obstacle_prob=0.2, max_cost=5):
    rows = random.randint(min_size, max_size)
    cols = random.randint(min_size, max_size)
    grid = Grid(rows, cols)

    # Couts random
    for i in range(rows):
        for j in range(cols):
            cost = random.randint(1, max_cost)
            grid.set_cost(i, j, cost)
    
    # Obstacle et random pos
    for i in range(rows):
        for j in range(cols):
            if random.random() < obstacle_prob:
                grid.set_obstacle(i, j)
    
    return grid


#---------------------------# Param Grid #-----------------------------#

grid = generate_random_grid()

start = (random.randint(0, grid.rows - 1), random.randint(0, grid.cols - 1))
goal = (random.randint(0, grid.rows - 1), random.randint(0, grid.cols - 1))
while not grid.is_valid_cell(start[0], start[1]):
    start = (random.randint(0, grid.rows - 1), random.randint(0, grid.cols - 1))
while not grid.is_valid_cell(goal[0], goal[1]) or goal == start:
    goal = (random.randint(0, grid.rows - 1), random.randint(0, grid.cols - 1))

#BFS avec cout
path, total_cost, explored = bfs_with_cost(grid, start, goal)

# Grille
grid.plot(path=path, start=start, goal=goal, explored=explored)
print("Coût total BFS:", total_cost)
