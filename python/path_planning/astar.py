from heapq import heappush, heappop


def manhattan_distance(goal, gain: float = 1):
    def distance(start):
        x_dist = abs(goal.row - start.row)
        y_dist = abs(goal.col - start.col)
        return (x_dist + y_dist) * gain

    return distance


class Node:
    def __init__(self, state, parent, cost=0.0, heuristic=0.0):
        self.state = state
        self.parent = parent
        self.cost = cost
        self.heuristic = heuristic

    def __lt__(self, other):
        return (self.cost + self.heuristic) < (other.cost + other.heuristic)

    def copy(self):
        return Node(self.state, self.parent, self.cost, self.heuristic)

    def to_path(self):
        node = self.copy()
        path = []
        while node.parent is not None:
            path.append(node.state)
            node = node.parent
        return path[::-1]


class PriorityQueue:
    def __init__(self):
        self._container = []

    @property
    def empty(self):
        return not self._container

    def push(self, val):
        heappush(self._container, val)

    def pop(self):
        return heappop(self._container)


def astar(initial, goal_test, neighbours, heuristic):
    frontier = PriorityQueue()
    frontier.push(Node(initial, None, 0.0, heuristic(initial)))

    explored = {initial: 0.0}

    while not frontier.empty:
        current_node = frontier.pop()
        state = current_node.state

        if goal_test(state):
            return current_node

        for child in neighbours(state):
            cost = current_node.cost + 1

            if child not in explored or explored[child] > cost:
                explored[child] = cost
                frontier.push(Node(child, current_node, cost, heuristic(child)))

    return None
