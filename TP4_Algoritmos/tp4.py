from graph import Graph
from typing import List, Dict, Set, Optional, Any,Tuple
from collections import deque
import time
import random

def createGraph(directional):
    page_graph = Graph()
    with open('C:/Users/iegre/OneDrive/Escritorio/repositorio Git/Mi Repositorio/Mi-repo-de-la-facu/TP4_Algoritmos/web-Google.txt', 'r') as file:
        for l in file:
            if "# FromNodeId	ToNodeId" in l:
                break

        for l in file:
            if not l:
                break
            edge = tuple(int(v.replace("\n", "").replace("\t", "")) for v in l.split("\t"))
            for v in edge:
                if not page_graph.vertex_exists(v):
                    page_graph.add_vertex(str(v))
            
            if(not directional):
                page_graph.add_edge(str(edge[0]), str(edge[1]))
                page_graph.add_edge(str(edge[1]), str(edge[0]))
            else:
                page_graph.add_edge(str(edge[0]), str(edge[1]))

    return page_graph

directedGraph = createGraph(directional=True)
nonDirectedGraph = createGraph(directional=False)

#1)

def iterative_dfs(graph:Graph,start_vertex: str, visited: Set[str],) -> Set[str]:
    stack = [start_vertex]
    component = set()
    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            visited.add(vertex) 
            component.add(vertex)
            stack.extend(graph.get_neighbors(vertex))                                
    return component

def find_WCC(graph: Graph) -> List[Set[str]]:

    visited = set()
    components = []
    
    for vertex in graph._graph:
        if vertex not in visited:
            component = iterative_dfs(graph,vertex, visited)
            components.append(component)
    
    return components


wccs = find_WCC(nonDirectedGraph)

largest_wcc = max(wccs, key=len)

print(f"Size of the largest WCC: {len(largest_wcc)}")
print(f"Total amount of WCC:{len(wccs)}")


# 2)

def bfs_shortest_paths(graph:Graph, start_vertex: str):
    distances = {vertex:-1 for vertex in graph._graph}
    distances[start_vertex] = 0

    queue = deque([start_vertex])

    while queue:
        current_vertex = queue.popleft()
        current_distance = distances[current_vertex]

        for neighbor in graph.get_neighbors(current_vertex):
            if distances[neighbor] == -1:
                distances[neighbor] = current_distance + 1
                queue.append(neighbor)

    return distances

def bfs_shortest_paths_with_time(graph:Graph, start_vertex: str):

    start_time = time.time()

    distances = bfs_shortest_paths(graph,start_vertex)

    end_time = time.time()
    elapsed_time = end_time - start_time

    return distances, elapsed_time


times =[]
allDistances = []
maxDistances = []

for _ in range(100):
    starting_vertex = random.choice(list(directedGraph._graph.items()))
    distances , elapsed_time = bfs_shortest_paths_with_time(directedGraph,starting_vertex[0]) 
    times.append(elapsed_time)
    allDistances.append(distances)
    maxDistances.append(max(distances.values()))

print(f"Time taken to find shortest path of 100 nodes: {sum(times)}")
print(f"Aproximate time taken to find shortest path of 1 node: {sum(times)/100}")
print(f"Aproximate time taken to find shortest path of all nodes: {sum(times)/100 * 875713}")
print(f"Aproximate time taken in hours to find shortest path of all nodes: {sum(times)/100 * 875713 /60/60}")

# 3) cantidad de triangulos

def find_cycle_triangles(graph:Graph) -> List[Tuple[str, str, str]]:
    cycle_triangles = []
    for v1 in graph._graph:
        for v2 in graph.get_neighbors(v1):
            for v3 in graph.get_neighbors(v2):
                if v1 in graph.get_neighbors(v3):
                    cycle_triangles.append((v1, v2, v3))
    return cycle_triangles

cantidadTriangulos20 = find_cycle_triangles(directedGraph)
print(f"Quantity of triangles:{len(cantidadTriangulos20)}")

# 4)
 
print(f"The diameter of the graph is aproximatly: {max(maxDistances)}")

# 5)

def random_walk(graph:Graph, start_vertex: str, steps: int) -> Dict[str, int]:
    visit_count = {}
    current_vertex = start_vertex
    visit_count[current_vertex] = 1

    for _ in range(steps):
        neighbors = graph.get_neighbors(current_vertex)
        if not neighbors:
            break 
        current_vertex = random.choice(neighbors)
        if current_vertex  in visit_count:
            visit_count[current_vertex] += 1
        else:
            visit_count[current_vertex] = 1

    return visit_count

steps = 300
quantOfRandomNodes = 100
finalResult = {}
for _ in range(quantOfRandomNodes):
    starting_vertex = random.choice(list(directedGraph._graph.items()))
    result = random_walk(directedGraph,starting_vertex[0], steps)
    for item in result:
        if item in finalResult:
            finalResult[item] += result[item]
        else:
            finalResult[item] = result[item]

top_10_vertices = sorted(finalResult.items(), key=lambda item: item[1], reverse=True)[:10]
print(top_10_vertices)
for vertice in top_10_vertices:
    print(f"This node {vertice[0]} has Page Rank of  {vertice[1] / len(directedGraph._graph.keys())}")

# 6) 

def find_cycles(graph:Graph, timeout: float):
    start_time = time.time()

    def iterative_dfs(start_vertex: str, size: int):

        stack = [(start_vertex, [start_vertex])]
        visited = set()

        while stack:
            if time.time() - start_time > timeout:
                raise TimeoutError("Search timed out")

            current_vertex, path = stack.pop()

            if len(path) == size:
                if start_vertex in graph.get_neighbors(path[-1]):

                    return path
                continue

            for neighbor in graph.get_neighbors(current_vertex):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor]))
        return None

    for size in range(2, len(graph._graph)):
        try:
            for vertex in graph._graph:
                cycle = iterative_dfs(vertex, size)
                if cycle:
                    print(f"Cycle of size {size} found: {cycle}")
                    break
            continue
        except TimeoutError:
            print("Timeout reached")
            break

    return []


find_cycles(directedGraph,600)


# EXTRAS

# clustering coeficient

def calculate_clustering_coefficient(graph: Graph):
    def local_clustering_coefficient(vertex: str):
        neighbors = graph.get_neighbors(vertex)
        if len(neighbors) < 2:
            return 0.0
        total_possible_edges = len(neighbors) * (len(neighbors) - 1)
        actual_edges = 0

        for i in range(len(neighbors)):
            for j in range(len(neighbors)):
                if i != j and graph.edge_exists(neighbors[i], neighbors[j]):
                    actual_edges += 1

        return actual_edges / total_possible_edges
    
    vertices = graph._graph.keys()
    total_clustering_coefficient = 0.0
    for vertex in vertices:
        total_clustering_coefficient += local_clustering_coefficient(vertex)
    
    return total_clustering_coefficient / len(vertices) if len(vertices) > 0 else 0.0

print(calculate_clustering_coefficient(nonDirectedGraph))    


# Betweenness centrality

def betweennessCentrality(graph:Graph):
    nodeQuantityOfShortestPaths = {}
    allDistances = []
    counterOfTotalShortestPaths = 0
    highestBC = [" ",-1]
    for _ in range(100):
        counterOfShortestPathsThorughNode = 0
        starting_vertex = random.choice(list(graph._graph.items()))
        distances = bfs_shortest_paths(graph,starting_vertex[0])
        allDistances.append(distances)
        for dist in distances.values():
            if dist > 0:
                counterOfTotalShortestPaths += 1
                counterOfShortestPathsThorughNode += 1
        nodeQuantityOfShortestPaths[starting_vertex[0]] = counterOfShortestPathsThorughNode
    for vertex in nodeQuantityOfShortestPaths:
        BCOfVertex = nodeQuantityOfShortestPaths[vertex] / counterOfTotalShortestPaths
        if BCOfVertex  > highestBC[1]:
            highestBC[0] = vertex
            highestBC[1] = BCOfVertex
    
    return highestBC

BC = betweennessCentrality(directedGraph)  
print(f"The node {BC[0]} has the highest betweenness centrality of {BC[1]}")     