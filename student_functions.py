
import numpy as np
from collections import deque
from queue import PriorityQueue

def BFS(matrix, start, end):
    """
    BFS algorithm:
    Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes, each key is a visited node,
        each value is the adjacent node visited before it.
    path: list
        Founded path
    """
    # TODO: 
    # Convert the matrix to a numpy array
    # Number of nodes in the graph
    n = len(matrix)
    
    # Initialize the visited dictionary and path list
    visited = {}
    path = []

    # Create a queue for BFS
    queue = deque([start])

    # Mark the start node as visited
    visited[start] = None

    # BFS algorithm
    while queue:
        current_node = queue.popleft()

        # If the current node is the end node, reconstruct the path and return
        if current_node == end:
            while current_node is not None:
                path.insert(0, current_node)
                current_node = visited[current_node]
            return visited, path

        # Visit all adjacent nodes of the current node
        for neighbor, weight in enumerate(matrix[current_node]):
            if weight != 0 and neighbor not in visited:
                visited[neighbor] = current_node
                queue.append(neighbor)

    # If no path is found
    return visited, path

def DFS(matrix, start, end):
    """
    DFS algorithm
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited 
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """

    # TODO: 
    def dfs_recursive(node):
        nonlocal visited, path
        visited[node] = True
        path.append(node)

        for neighbor in range(len(matrix[node])):
            if matrix[node][neighbor] != 0 and not visited[neighbor]:
                dfs_recursive(neighbor)

    # Validate start and end nodes
    if start < 0 or start >= len(matrix) or end < 0 or end >= len(matrix):
        raise ValueError("Invalid start or end node")

    # Initialize visited dictionary and path list
    visited = {i: False for i in range(len(matrix))}
    path = []

    # Perform DFS
    dfs_recursive(start)

    return visited, path


def UCS(matrix, start, end):
    """
    Uniform Cost Search algorithm
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:  
    # Priority queue to store nodes and their costs
    def neighbors(node):
        return [i for i in range(len(matrix[node])) if matrix[node][i] != 0]

    def get_cost(from_node, to_node):
        return matrix[from_node][to_node]

    visited = {}
    path = []

    queue = PriorityQueue()
    queue.put((0, start))

    while not queue.empty():
        cost, node = queue.get()

        if node not in visited:
            visited[node] = None  # None indicates the starting node
            if node == end:
                break

            for neighbor in neighbors(node):
                if neighbor not in visited:
                    total_cost = cost + get_cost(node, neighbor)
                    queue.put((total_cost, neighbor))
                    visited[neighbor] = node  # Store the parent of the current node

    current = end
    while current is not None:
        path.insert(0, current)
        current = visited[current]

    return visited, path
    
    # If no path is found
    # return visited, []

def IDS(matrix, start, end):
    """
    Iterative deepening search algorithm
    Parameters:
    ---------------------------
    matrix: np array
        The graph's adjacency matrix
    start: integer
        starting node
    end: integer
        ending node
    
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO:  
    def DFS(matrix, current, end, depth, visited, path,__visited):
        if current == end:
            return True
        if depth == 0:
            return False

        for neighbor, weight in enumerate(matrix[current]):
            if weight != 0 and neighbor not in visited:
                visited[neighbor] = current
                __visited.update(visited)
                path.append(neighbor)
                if DFS(matrix, neighbor, end, depth - 1, visited, path,__visited):
                    return True
                path.pop()

        return False
    __visited = {}
    n = len(matrix)
    for depth in range(n):
        visited = {}
        path = [start]

        if DFS(matrix, start, end, depth, visited, path,__visited):
            return __visited, path

    return {}, []

def GBFS(matrix, start, end):
    """
    Greedy Best First Search algorithm 
    heuristic : edge weights
     Parameters:
    ---------------------------
    matrix: np array 
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
   
    Returns
    ---------------------
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 
    # path=[]
    # visited={}
    # return visited, path
    def best_first_search(actual_Src, target, n):
        visited = set()
        pq = PriorityQueue()
        pq.put((0, actual_Src))
        
        while not pq.empty():
            cost, u = pq.get()
            
            if u in visited:
                continue
            
            visited.add(u)
            path.append(u)
            
            if u == target:
                break
            
            neighbors = [(v, matrix[u][v]) for v in range(n) if matrix[u][v] > 0]
            for v, c in sorted(neighbors, key=lambda x: x[1]):
                if v not in visited:
                    pq.put((c, v))
    
    # Initialize variables
    path = []
    visited = {}
    n = len(matrix)
    
    # Call the GBFS algorithm
    best_first_search(start, end, n)
    
    # Construct the visited dictionary
    for i in range(1, len(path)):
        visited[path[i]] = path[i-1]
    
    return visited, path
def Astar(matrix, start, end, pos):
    """
    A* Search algorithm
    heuristic: eclid distance based positions parameter
     Parameters:
    ---------------------------
    matrix: np array UCS
        The graph's adjacency matrix
    start: integer 
        starting node
    end: integer
        ending node
    pos: dictionary. keys are nodes, values are positions
        positions of graph nodes
    ---------------------------
    Returns:
    visited
        The dictionary contains visited nodes: each key is a visited node, 
        each value is the key's adjacent node which is visited before key.
    path: list
        Founded path
    """
    # TODO: 

    # path=[]
    # visited={}
    # return visited, path
    # Helper class to represent the graph
    class Graph:
        def __init__(self, adjac_matrix, pos):
            self.adjac_matrix = adjac_matrix
            self.pos = pos

        def get_neighbors(self, v):
            return [(i, weight) for i, weight in enumerate(self.adjac_matrix[v]) if weight > 0]

        def h(self, n):
            # Calculate Euclidean distance as the heuristic
            x1, y1 = self.pos[n]
            x2, y2 = self.pos[end]
            return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        def a_star_algorithm(self, start, stop):
            open_lst = set([start])
            closed_lst = set([])
            poo = {start: 0}
            par = {start: start}

            while open_lst:
                n = min(open_lst, key=lambda x: poo[x] + self.h(x))

                if n == stop:
                    reconst_path = []
                    while par[n] != n:
                        reconst_path.append(n)
                        n = par[n]

                    reconst_path.append(start)
                    reconst_path.reverse()
                    return reconst_path

                for (m, weight) in self.get_neighbors(n):
                    if m not in open_lst and m not in closed_lst:
                        open_lst.add(m)
                        par[m] = n
                        poo[m] = poo[n] + weight
                    else:
                        if poo[m] > poo[n] + weight:
                            poo[m] = poo[n] + weight
                            par[m] = n
                            if m in closed_lst:
                                closed_lst.remove(m)
                                open_lst.add(m)

                open_lst.remove(n)
                closed_lst.add(n)

            return None

    # Convert the adjacency matrix to a list of lists
    adjac_list = matrix.tolist()

    # Create a Graph instance
    graph = Graph(adjac_list, pos)

    # Run A* algorithm
    path = graph.a_star_algorithm(start, end)

    # Construct visited dictionary
    visited = {}
    for i in range(len(path) - 1):
        visited[path[i]] = path[i + 1]

    return visited, path