from graph import Graph
from typing import List, Dict, Set, Optional, Any,Tuple
from collections import deque,defaultdict
import time
import random


#/home/ig/Desktop/Repositorios Git/Mi Repositorio/Mi-repo-de-la-facu/TP4_Algoritmos/web-Google.txt
#'C:/Users/iegre/OneDrive/Escritorio/repositorio Git/Mi Repositorio/Mi-repo-de-la-facu/TP4_Algoritmos/web-Google.txt'
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

#1) Cantidad de componentes conexos y conexo mas grande 

wccs = find_WCC(nonDirectedGraph)

# Find the largest SCC by size
largest_wcc = max(wccs, key=len)

# print(f"The largest SCC is: {largest_scc}")
print(f"Size of the largest WCC: {len(largest_wcc)}")
print(f"Total amount of WCC:{len(wccs)}")

# 3) cantidad de triangulos

def find_cycle_triangles(graph:Graph) -> List[Tuple[str, str, str]]:
    cycle_triangles = []
    for v1 in graph._graph:
        for v2 in graph.get_neighbors(v1):
            for v3 in graph.get_neighbors(v2):
                if v1 in graph.get_neighbors(v3):
                    cycle_triangles.append((v1, v2, v3))
    return cycle_triangles

#--------------------------------------------
# cantidadTriangulos20 = find_cycle_triangles(directedGraph)



# print(f"La cantidad de triagulos son:{len(cantidadTriangulos20)/3}")

# 3)

def bfs_shortest_paths(graph:Graph, start_vertex: str):
    # if start_vertex not in graph._graph:
    #     raise ValueError("The start vertex does not exist")

    #float('inf')
    distances = {vertex:-1 for vertex in graph._graph}
    distances[start_vertex] = 0

    queue = deque([start_vertex])

    while queue:
        current_vertex = queue.popleft()
        current_distance = distances[current_vertex]

        for neighbor in graph.get_neighbors(current_vertex):
            if distances[neighbor] == -1: #float('inf'):  # not visited yet
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

# for _ in range(100):
#     starting_vertex = random.choice(list(directedGraph._graph.items()))
#     distances , elapsed_time = bfs_shortest_paths_with_time(directedGraph,starting_vertex[0]) 
#     times.append(elapsed_time)
#     allDistances.append(distances)
#     maxDistances.append(max(distances.values()))

# print(f"Time taken to find shortest path of 100 nodes: {sum(times)}")
# print(f"Aproximate time taken to find shortest path of 1 node: {sum(times)/100}")
# print(f"Aproximate time taken to find shortest path of all nodes: {sum(times)/100 * 875713}")
# print(f"Aproximate time taken in hours to find shortest path of all nodes: {sum(times)/100 * 875713 /60/60}")

# # 4) 
# print(f"The diameter of the graph is aproximatly: {max(maxDistances)}")

# 5)

# def random_walk(graph:Graph, start_vertex: str, steps: int) -> Dict[str, int]:
#     if not graph.vertex_exists(start_vertex):
#         raise ValueError("The start vertex does not exist")

#     visit_count = {}
#     current_vertex = start_vertex
#     visit_count[current_vertex] = 1

#     for _ in range(steps):
#         neighbors = graph.get_neighbors(current_vertex)
#         if not neighbors:
#             break 
#         current_vertex = random.choice(neighbors)
#         if current_vertex  in visit_count:
#             visit_count[current_vertex] += 1
#         else:
#             visit_count[current_vertex] = 1

#     return visit_count

# steps = 100
# finalResult = {}
# for _ in range(steps):
#     starting_vertex = random.choice(list(directedGraph._graph.items()))
#     result = random_walk(directedGraph,starting_vertex[0], steps)
#     for item in result:
#         if item in finalResult:
#             finalResult[item] += result[item]
#         else:
#             finalResult[item] = result[item]
        
# noRepeatsResultValues = []
# for value in finalResult.values():
#     if value not in noRepeatsResultValues:
#         noRepeatsResultValues.append(value)
# orderedValues = sorted(noRepeatsResultValues)
# res = []
# for i in range(1,6):
#     for key in finalResult :
#         if finalResult[key] == orderedValues[-i]:
#             print(f"This node {key} appeared {orderedValues[-i]} times")

# 6) 

# def find_cycles(graph: Graph) -> List[List[str]]:
#     index = 0
#     stack = []
#     index_map = {}
#     lowlink_map = {}
#     on_stack = {}
#     cycles = []
#     call_stack = []

#     for vertex in graph._graph.keys():
#         if vertex not in index_map:
#             call_stack.append((vertex, 'entry'))

#             while call_stack:
#                 current, stage = call_stack.pop()

#                 if stage == 'entry':
#                     index_map[current] = index
#                     lowlink_map[current] = index
#                     index += 1
#                     stack.append(current)
#                     on_stack[current] = True

#                     call_stack.append((current, 'exit'))
#                     for neighbor in graph.get_neighbors(current):
#                         if neighbor not in index_map:
#                             call_stack.append((neighbor, 'entry'))
#                         elif on_stack.get(neighbor, False):
#                             lowlink_map[current] = min(lowlink_map[current], index_map[neighbor])

#                 elif stage == 'exit':
#                     for neighbor in graph.get_neighbors(current):
#                         if neighbor in index_map and on_stack.get(neighbor, False):
#                             lowlink_map[current] = min(lowlink_map[current], lowlink_map[neighbor])

#                     if lowlink_map[current] == index_map[current]:
#                         cycle = []
#                         while True:
#                             w = stack.pop()
#                             on_stack[w] = False
#                             cycle.append(w)
#                             if w == current:
#                                 break
#                         if len(cycle) > 1 or (len(cycle) == 1 and current in graph.get_neighbors(current)):
#                             cycles.append(cycle)

#     return cycles

# def find_cycles(graph:Graph) -> List[List[str]]:
#     visited = set()
#     stack = []
#     parent_map: Dict[str, str] = {}
#     cycles = []

#     for start_vertex in graph._graph.keys():
#         if start_vertex not in visited:
#             stack.append((start_vertex, None))  # None as parent of start vertex

#             while stack:
#                 current, parent = stack.pop()

#                 if current in visited:
#                     # Cycle detected, reconstruct the cycle path
#                     cycle = []
#                     x = current
#                     while x != parent and x is not None:
#                         cycle.append(x)
#                         x = parent_map.get(x, None)
#                     if x is not None:
#                         cycle.append(x)
#                         cycles.append(cycle[::-1])
#                 else:
#                     visited.add(current)
#                     parent_map[current] = parent

#                     for neighbor in graph.get_neighbors(current):
#                         if neighbor != parent:  # To avoid immediate back edge to parent
#                             stack.append((neighbor, current))

#     return cycles


# def find_cycles(graph:Graph):
#     visited = set()
#     stack = []
#     cycles = []

#     for start_vertex in graph._graph.keys():
#         if start_vertex not in visited:
#             stack.append(start_vertex)  # None as parent of start vertex

#             while stack:
#                 current = stack.pop()
                
#                 if current in visited:
#                     # Cycle detected, reconstruct the cycle path
#                     cycle = []
#                     x = current
#                     while x != parent:
#                         cycle.append(x)
#                         x = parent_map[x]
#                     cycle.append(current)
#                     cycles.append(cycle[::-1])
#                 else:
#                     visited.add(current)
#                     parent_map[current] = parent

#                     for neighbor in graph.get_neighbors(current):
#                         if neighbor != parent:  # To avoid immediate back edge to parent
#                             stack.append((neighbor, current))

#     return cycles

# def find_cycles_of_increasing_size(graph: Graph, timeout: int) -> List[List[str]]:
#     start_time = time.time()
#     cycles = []

#     def iterative_dfs(start_vertex: str, target_length: int) -> None:
#         stack = [(start_vertex, [start_vertex])]
#         while stack:
#             current_vertex, path = stack.pop()
#             if len(path) == target_length:
#                 if current_vertex == start_vertex:
#                     cycles.append(path[:])
#                     print(f"Cycle of size {target_length} has been found")
#                     print(cycles)
#                 return
#             for neighbor in graph.get_neighbors(current_vertex):
#                 if neighbor not in path or (len(path) == target_length - 1 and neighbor == start_vertex):
#                     stack.append((neighbor, path + [neighbor]))
#             if time.time() - start_time > timeout:
#                 return

#     target_length = 2
#     while time.time() - start_time <= timeout:
#         for vertex in graph._graph:
#             iterative_dfs(vertex, target_length)
#         target_length += 1

#     return cycles

def find_cycles(graph:Graph, timeout: float) -> List[List[str]]:
    start_time = time.time()

    def iterative_dfs(start_vertex: str, size: int) -> Optional[List[str]]:
        stack = [(start_vertex, [start_vertex], {start_vertex})]

        while stack:
            if time.time() - start_time > timeout:
                raise TimeoutError("Search timed out")

            current_vertex, path, visited = stack.pop()
            if len(path) == size:
                if path[-1] in graph.get_neighbors(start_vertex):
                    return path
                continue

            for neighbor in graph.get_neighbors(current_vertex):
                if neighbor not in visited:
                    new_visited = visited.copy()
                    new_visited.add(neighbor)
                    stack.append((neighbor, path + [neighbor], new_visited))

        return None

    for size in range(2, len(graph._graph) + 1):
        try:
            for vertex in graph._graph:
                cycle = iterative_dfs(vertex, size)
                if cycle:
                    print(f"Cycle of size {size} found: {cycle}")
                    break
            # else:
            continue
            break
        except TimeoutError:
            print("Timeout reached")
            break

    return []


# print(len(max(find_cycles(directedGraph,600),key=len)))
find_cycles(directedGraph,600)

# clustering coeficient

# def calculate_clustering_coefficient(graph: Graph) -> float:
#     def local_clustering_coefficient(vertex: str) -> float:
#         neighbors = graph.get_neighbors(vertex)
#         if len(neighbors) < 2:
#             return 0.0
#         total_possible_edges = len(neighbors) * (len(neighbors) - 1)
#         actual_edges = 0

#         for i in range(len(neighbors)):
#             for j in range(len(neighbors)):
#                 if i != j and graph.edge_exists(neighbors[i], neighbors[j]):
#                     actual_edges += 1

#         return actual_edges / total_possible_edges
    
#     vertices = graph._graph.keys()
#     total_clustering_coefficient = 0.0
#     for vertex in vertices:
#         total_clustering_coefficient += local_clustering_coefficient(vertex)
    
#     return total_clustering_coefficient / len(vertices) if len(vertices) > 0 else 0.0

# print(calculate_clustering_coefficient(directedGraph))    
# print(calculate_clustering_coefficient(nonDirectedGraph))    
