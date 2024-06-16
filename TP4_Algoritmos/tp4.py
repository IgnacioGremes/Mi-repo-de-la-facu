from graph import Graph
from typing import List, Dict, Set, Optional, Any,Tuple
from collections import deque
import time
import random


#/home/ig/Desktop/Repositorios Git/Mi Repositorio/Mi-repo-de-la-facu/TP4_Algoritmos/web-Google.txt
#'C:/Users/iegre/OneDrive/Escritorio/repositorio Git/Mi Repositorio/Mi-repo-de-la-facu/TP4_Algoritmos/web-Google.txt'
def createGraph(directional, transposed):
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
            elif(transposed):
                page_graph.add_edge(str(edge[1]), str(edge[0]))
            else:
                page_graph.add_edge(str(edge[0]), str(edge[1]))

    return page_graph

# transposedGraph = createGraph(True,True)
directedGraph = createGraph(True,False)
nonDirectedGraph = createGraph(False,False)

# def kosaraju_iterative(graph:Graph):    #Funciona
#     stack = []
#     visited = set()
#     sccs = []

#     for vertex in graph._graph:
#         if vertex not in visited:
#             stack.append(vertex)
#             visited.add(vertex)
#             while stack:
#                 current = stack[-1]
#                 if current in graph._graph:
#                     neighbors = [neighbor for neighbor in graph.get_neighbors(current) if neighbor not in visited]
#                     if neighbors:
#                         stack.append(neighbors[0])
#                         visited.add(neighbors[0])
#                     else:
#                         stack.pop()
#                         sccs.append(current)
#                 else:
#                     stack.pop()

#     reversed_graph = {}
#     for u in graph._graph:
#         for v in graph.get_neighbors(u):
#             if v not in reversed_graph:
#                 reversed_graph[v] = set()
#             reversed_graph[v].add(u)

#     visited.clear()
#     scc_components = []
#     for node in reversed(sccs):
#         if node not in visited:
#             stack = [node]
#             component = []
#             visited.add(node)
#             while stack:
#                 current = stack.pop()
#                 component.append(current)
#                 for neighbor in reversed_graph.get(current, []):
#                     if neighbor not in visited:
#                         stack.append(neighbor)
#                         visited.add(neighbor)
#             scc_components.append(component)

#     return scc_components

def find_WCC(graph: Graph) -> List[Set[str]]:
    def iterative_dfs(start_vertex: str, visited: Set[str],components) -> Set[str]:
        stack = [start_vertex]
        component = set()
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex) 
                component.add(vertex)
                stack.extend(graph.get_neighbors(vertex))                                
        return component

    visited = set()
    components = []
    
    for vertex in graph._graph:
        if vertex not in visited:
            # print(f"Doing DFS{vertex}")
            component = iterative_dfs(vertex, visited,components)
            components.append(component)
            # if component != None:
    
    return components

#1) Cantidad de componentes conexos y conexo mas grande 

wccs = find_WCC(nonDirectedGraph)

# Find the largest SCC by size
largest_wcc = max(wccs, key=len)

# print(f"The largest SCC is: {largest_scc}")
print(f"Size of the largest SCC: {len(largest_wcc)}")
print(f"Total amount of SCC:{len(wccs)}")

# 3) cantidad de triangulos

# def calcularTraingulos(graph:Graph,transposedGraph:Graph):
#     for vertex in graph._graph:
#         for neighbor in graph.get_neighbors(vertex):
#             if vertex < neighbor:



# def intersection(set1: Set[str], set2: Set[str]) -> Set[str]:
#     return set1 & set2  # Using set intersection
# def intersection(lst1, lst2):
#     lst3 = [value for value in lst1 if value in lst2]
#     return lst3


# def count_cycle_triangles(G: Graph, GT: Graph) -> Tuple[int, List[Tuple[str, str, str]]]:
#     # V = list(G._graph.keys())
#     # N_plus = {v: set(G.get_neighbors(v)) for v in V}
#     # N_minus = {v: set(GT.get_neighbors(v)) for v in V}

# # N-(u,G) = N+ (u,GT)

#     c = 0  # Initialize cycle triangle count
#     triangles = []  # Optional: list of cycle triangles

#     for u in G._graph:
#         for v in G.get_neighbors(u):
#             # if u < v:
#             S = intersection (G.get_neighbors(v), GT.get_neighbors(u))
#             # S = intersection(N_minus[u], N_plus[v])
#             for w in S:
#                 # if u < w:
#                 triangles.append((u, v, w))  # For listing
#                 c += 1

#     return c, triangles

def find_cycle_triangles(graph:Graph) -> List[Tuple[str, str, str]]:
    cycle_triangles = []
    for v1 in graph._graph:
        for v2 in graph.get_neighbors(v1):
            for v3 in graph.get_neighbors(v2):
                if v1 in graph.get_neighbors(v3):
                    cycle_triangles.append((v1, v2, v3))
    return cycle_triangles

# def find_cycle_triangles(graph:Graph) -> List[Tuple[str, str, str]]:
#     """
#     Finds all cycle triangles in the graph using an adjacency matrix.
#     :return: List of tuples, each containing three vertices that form a triangle
#     """
#     vertices = list(graph._graph.keys())
#     n = len(vertices)
#     vertex_index = {vertices[i]: i for i in range(n)}

#     # Create the adjacency matrix
#     adj_matrix = [[0] * n for _ in range(n)]
#     for vertex in vertices:
#         for neighbor in graph.get_neighbors(vertex):
#             adj_matrix[vertex_index[vertex]][vertex_index[neighbor]] = 1

#     triangles = []
#     for i in range(n):
#         for j in range(i + 1, n):
#             if adj_matrix[i][j]:
#                 for k in range(j + 1, n):
#                     if adj_matrix[i][k] and adj_matrix[j][k]:
#                         triangles.append((vertices[i], vertices[j], vertices[k]))

#     return triangles

# cantidadTriangulos, triangulos = count_cycle_triangles(directedGraph,transposedGraph)
cantidadTriangulos20 = find_cycle_triangles(directedGraph)



print(f"La cantidad de triagulos son:{len(cantidadTriangulos20)/3}")

# 3)

def bfs_shortest_paths(graph:Graph, start_vertex: str):
    # if start_vertex not in graph._graph:
    #     raise ValueError("The start vertex does not exist")

    distances = {vertex: float('inf') for vertex in graph._graph}
    distances[start_vertex] = 0

    queue = deque([start_vertex])

    while queue:
        current_vertex = queue.popleft()
        current_distance = distances[current_vertex]

        for neighbor in graph.get_neighbors(current_vertex):
            if distances[neighbor] == float('inf'):  # not visited yet
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

for _ in range(100):
    starting_vertex = random.choice(list(directedGraph._graph.items()))
    distances , elapsed_time = bfs_shortest_paths_with_time(directedGraph,starting_vertex[0]) 
    times.append(elapsed_time)
    allDistances.append(distances)

print(f"Time taken to find shortest path of 100 nodes: {sum(times)}")
print(f"Aproximate time taken to find shortest path of 1 node: {sum(times)/100}")
print(f"Aproximate time taken to find shortest path of all nodes: {sum(times)/100 * 875713}")
print(f"Aproximate time taken in hours to find shortest path of all nodes: {sum(times)/100 * 875713 /60/60}")

