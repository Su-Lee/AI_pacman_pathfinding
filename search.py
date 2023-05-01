# search.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Michael Abir (abir2@illinois.edu) on 08/28/2018
# Modified by Shang-Tse Chen (stchen@csie.ntu.edu.tw) on 03/03/2022

"""
This is the main entry point for HW1. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
# Search should return the path.
# The path should be a list of tuples in the form (row, col) that correspond
# to the positions of the path taken by your search algorithm.
# maze is a Maze object based on the maze from the file specified by input filename
# searchMethod is the search method specified by --method flag (bfs,dfs,astar,astar_multi,fast)

import math
import heapq
from queue import PriorityQueue
from collections import defaultdict
import itertools


def search(maze, searchMethod):
    return {
        "bfs": bfs,
        "astar": astar,
        "astar_corner": astar_corner,
        "astar_multi": astar_multi,
        "fast": fast,
    }.get(searchMethod)(maze)


def bfs(maze):
    """
    Runs BFS for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # initialize starting position and the state queue
    start = maze.getStart()
    state_queue = [(start, [start], set())]
    visited = set()

    # BFS loop
    while state_queue:

        # pop the first state from the queue
        curr_state, path, visited_goals = state_queue.pop(0)

        # if the current state is an objective, we have found a solution
        if maze.isObjective(curr_state[0], curr_state[1]):
            return path

        # get neighbors and add to queue
        for neighbor in maze.getNeighbors(curr_state[0], curr_state[1]):

            # check if neighbor has already been visited
            if neighbor not in visited:

                # update the path and the visited set
                new_path = path + [neighbor]
                new_visited = visited_goals.copy()

                # if neighbor is a goal, add it to the visited goals set
                if maze.isObjective(neighbor[0], neighbor[1]):
                    new_visited.add(neighbor)

                # add new state to queue
                state_queue.append((neighbor, new_path, new_visited))
                visited.add(neighbor)

        # sort the queue based on the number of visited goals (descending) and path length (ascending)
        state_queue = sorted(state_queue, key=lambda x: (-len(x[2]), len(x[1])))

    # if no solution is found, return None
    return None

# https://stackoverflow.com/questions/13578287/a-a-star-algorithm-clarification
def astar(maze):
    """
    Runs A star for part 1 of the assignment.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    start = maze.getStart()
    objectives = maze.getObjectives()

    # calculate heuristic function
    def heuristic(pos):
        # use Manhattan distance instead of Euclidean distance
        distances = [abs(pos[0] - obj[0]) + abs(pos[1] - obj[1]) for obj in objectives]
        return min(distances)

    # initialize open set and closed set
    open_set = PriorityQueue()
    open_set.put((0, start))
    closed_set = set()
    came_from = {}

    # initialize g_score and f_score
    g_score = {start: 0}
    f_score = {start: heuristic(start)}

    while not open_set.empty():
        _, current = open_set.get()

        if maze.isObjective(current[0], current[1]):
            path = [current]
            while current != start:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        closed_set.add(current)

        for neighbor in maze.getNeighbors(current[0], current[1]):
            if neighbor in closed_set:
                continue

            tentative_g_score = g_score[current] + 1

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor)

                if neighbor not in [pos for _, pos in open_set.queue]:
                    open_set.put((f_score[neighbor], neighbor))

    return None


def astar_corner(maze):
    """
    Runs A star for part 2 of the assignment in the case where there are four corner objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
        """
    objectives = maze.getObjectives()
    shortest_path = []
    for objective_order in itertools.permutations(objectives):
        current_path = []
        last_visited = maze.getStart()
        for objective in objective_order:

            # calculate heuristic function
            def heuristic(pos):
                # use Manhattan distance instead of Euclidean distance
                distances = abs(pos[0] - objective[0]) + abs(pos[1] - objective[1])
                return distances

            # initialize open set and closed set
            open_set = PriorityQueue()
            open_set.put((0, last_visited))
            closed_set = set()
            came_from = {}

            # initialize g_score and f_score
            g_score = {last_visited: 0}
            f_score = {last_visited: heuristic(last_visited)}

            # Find the closest objective.
            closest_objective = None
            closest_objective_distance = float('inf')

            while not open_set.empty():
                _, current = open_set.get()

                # Check if the current position is an objective.
                if current in objectives:
                    # Update the closest objective if necessary.
                    distance_to_current = g_score[current]
                    if distance_to_current < closest_objective_distance:
                        closest_objective = current
                        closest_objective_distance = distance_to_current

                closed_set.add(current)

                # Check the neighbors of the current position.
                for neighbor in maze.getNeighbors(current[0], current[1]):
                    if neighbor in closed_set:
                        continue

                    tentative_g_score = g_score[current] + 1

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        f_score[neighbor] = tentative_g_score + heuristic(neighbor)

                        if neighbor not in [pos for _, pos in open_set.queue]:
                            open_set.put((f_score[neighbor], neighbor))

            # If a closest objective is found, construct the path and add it to the paths list.
            if closest_objective is not None:
                path = [closest_objective]
                temp = closest_objective
                # print("at objective: ", temp)
                while closest_objective != last_visited:
                    closest_objective = came_from[closest_objective]
                    path.append(closest_objective)
                path.reverse()
                current_path.append(path)
                last_visited = temp

        print(current_path)
        final_path = []
        for p in current_path:
            final_path.extend(p)

        if len(shortest_path) == 0:
            shortest_path = final_path
        elif len(shortest_path) > len(final_path):
            shortest_path = final_path
            print(shortest_path)

    return shortest_path


def astar_multi(maze):
    """
    Runs A star for part 3 of the assignment in the case where there are
    multiple objectives.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here

    return fast(maze)


def fast(maze):
    """
    Runs suboptimal search algorithm for part 4.

    @param maze: The maze to execute the search on.

    @return path: a list of tuples containing the coordinates of each state in the computed path
    """
    # TODO: Write your code here
    # get the start position and objectives
    start = maze.getStart()
    objectives = maze.getObjectives()

    # Calculate heuristic function.
    def first_heuristic(pos, remaining_objectives):
        # Use the sum of Manhattan distances to the remaining objectives as the heuristic value.
        return sum(abs(pos[0] - obj[0]) + abs(pos[1] - obj[1]) for obj in remaining_objectives)

    def heuristic(pos, remaining_objectives):
        # Use the sum of Manhattan distances to the remaining objectives as the heuristic value.
        return min(abs(pos[0] - obj[0]) + abs(pos[1] - obj[1]) for obj in remaining_objectives)

    # Initialize the paths list.
    paths = []
    last_visited = start

    # Find a path to each objective.
    while objectives:
        # Initialize the open set and closed set.
        open_set = PriorityQueue()
        open_set.put((0, last_visited))
        closed_set = set()
        came_from = {}

        # Initialize the g_score and f_score.
        g_score = {last_visited: 0}
        f_score = {last_visited: first_heuristic(last_visited, objectives)}

        # Find the closest objective.
        closest_objective = None
        closest_objective_distance = float('inf')

        while not open_set.empty():
            _, current = open_set.get()

            # Check if the current position is an objective.
            if current in objectives:
                # Update the closest objective if necessary.
                distance_to_current = g_score[current]
                if distance_to_current < closest_objective_distance:
                    closest_objective = current
                    closest_objective_distance = distance_to_current

            closed_set.add(current)

            # Check the neighbors of the current position.
            for neighbor in maze.getNeighbors(current[0], current[1]):
                if neighbor in closed_set:
                    continue

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, objectives)

                    if neighbor not in [pos for _, pos in open_set.queue]:
                        open_set.put((f_score[neighbor], neighbor))

        # If a closest objective is found, construct the path and add it to the paths list.
        if closest_objective is not None:
            path = [closest_objective]
            temp = closest_objective
            # print("at objective: ", temp)
            while closest_objective != last_visited:
                closest_objective = came_from[closest_objective]
                path.append(closest_objective)
            path.reverse()
            paths.append(path)

            last_visited = temp
            # Remove the closest objective from the list of objectives.
            objectives.remove(path[-1])
        else:
            # If no path to any objective is found, return None.
            # print("fail to find the solution")
            return None

    # return concatenated path
    final_path = []
    for p in paths:
        # print("path: ", p)
        final_path.extend(p)

    return final_path


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def mst_distance(pos, objectives):
    """
    Calculates the minimum spanning tree (MST) distance from pos to the remaining objectives.
    """
    n = len(objectives)
    # Build a complete graph with weights equal to the Euclidean distances between objectives.
    graph = {(i, j): euclidean_distance(objectives[i], objectives[j]) for i in range(n) for j in range(i+1, n)}
    # Add edges from pos to each objective.
    for i, obj in enumerate(objectives):
        graph[('pos', i)] = euclidean_distance(pos, obj)
    # Add start node to the graph
    graph[('pos', n)] = 0
    # Add edges from the start node to each objective.
    for i, obj in enumerate(objectives):
        graph[('pos', i)] = euclidean_distance(pos, obj)
    # Find the MST using Prim's algorithm.
    mst = set()
    start_node = ('pos', n)
    visited = set([start_node])
    edges = [(cost, start_node, ('obj', i)) for i, cost in enumerate(graph[start_node])]
    heapq.heapify(edges)
    while edges:
        cost, n1, n2 = heapq.heappop(edges)
        if n2 not in visited:
            visited.add(n2)
            mst.add((n1, n2))
            for i, cost in enumerate(graph[n2]):
                heapq.heappush(edges, (cost, n2, ('obj', i)))
    # Calculate the sum of weights of edges in the MST.
    return sum(graph[n1][n2] for n1, n2 in mst)


def calculate_mst_distance(pos, remaining_objectives):
    # Create a graph where each node is an objective and the weight between nodes is the Manhattan distance
    # between them.

    def heuristic(position, objective):
        # Use the sum of Manhattan distances to the remaining objectives as the heuristic value.
        return abs(position[0] - objective[0]) + abs(position[1] - objective[1])

    graph = defaultdict(dict)
    objectives = [pos] + remaining_objectives
    n = len(objectives)
    for i in range(n):
        for j in range(i + 1, n):
            weight = heuristic(objectives[i], objectives[j])
            graph[i][j] = weight
            graph[j][i] = weight

    # Run Prim's algorithm to find the minimum spanning tree.
    pq = PriorityQueue()
    visited = set()
    start_node = 0
    for neighbor, weight in graph[start_node].items():
        pq.put((weight, start_node, neighbor))
    visited.add(start_node)
    mst_weight = 0
    while not pq.empty():
        weight, u, v = pq.get()
        if v in visited:
            continue
        visited.add(v)
        mst_weight += weight
        for neighbor, weight in graph[v].items():
            if neighbor not in visited:
                pq.put((weight, v, neighbor))

    # Return the total distance of the minimum spanning tree.
    print("mst: ", mst_weight)
    return mst_weight

