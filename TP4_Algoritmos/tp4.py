from graph import Graph
from typing import List, Dict, Set, Optional, Any
# class TarjanSCC:
#     def __init__(self, graph):
#         self.graph = graph
#         self.index = 0
#         self.stack = []
#         self.indexes = {}
#         self.lowlinks = {}
#         self.on_stack = set()
#         self.sccs = []

#     def run(self):
#         for vertex in self.graph._graph:
#             if vertex not in self.indexes:
#                 self.strong_connect(vertex)
#         return self.sccs

#     def strong_connect(self, vertex):
#         self.indexes[vertex] = self.index
#         self.lowlinks[vertex] = self.index
#         self.index += 1
#         self.stack.append(vertex)
#         self.on_stack.add(vertex)

#         for neighbor in self.graph.get_neighbors(vertex):
#             if neighbor not in self.indexes:
#                 self.strong_connect(neighbor)
#                 self.lowlinks[vertex] = min(self.lowlinks[vertex], self.lowlinks[neighbor])
#             elif neighbor in self.on_stack:
#                 self.lowlinks[vertex] = min(self.lowlinks[vertex], self.indexes[neighbor])

#         if self.lowlinks[vertex] == self.indexes[vertex]:
#             scc = []
#             while True:
#                 w = self.stack.pop()
#                 self.on_stack.remove(w)
#                 scc.append(w)
#                 if w == vertex:
#                     break
#             self.sccs.append(scc)


# def tarjan_scc(graph: Graph):
#     index = {}   # To store the index of each node
#     lowlink = {} # To store the lowlink value of each node
#     stack = []   # Stack to simulate function calls
#     result = []  # To store the SCCs
#     on_stack = set()  # To track which nodes are currently on the stack
#     index_counter = 0
    
#     # Function to perform DFS
#     def dfs(v):
#         nonlocal index_counter
#         index[v] = index_counter
#         lowlink[v] = index_counter
#         index_counter += 1
#         stack.append(v)
#         on_stack.add(v)

#         for neighbor in graph.get_neighbors(v):
#             if neighbor not in index:
#                 dfs(neighbor)
#                 lowlink[v] = min(lowlink[v], lowlink[neighbor])
#             elif neighbor in on_stack:
#                 lowlink[v] = min(lowlink[v], index[neighbor])

#         if lowlink[v] == index[v]:
#             scc = []
#             while True:
#                 node = stack.pop()
#                 on_stack.remove(node)
#                 scc.append(node)
#                 if node == v:
#                     break
#             result.append(scc)

#     # Perform DFS for each unvisited node
#     for node in graph._graph:
#         if node not in index:
#             dfs(node)

#     return result


# def tarjan_iterative(graph:Graph):
#     stack = []
#     index = {}
#     low_link = {}
#     on_stack = {}
#     sccs = []

#     for v in graph._graph:
#         index[v] = -1
#         low_link[v] = -1
#         on_stack[v] = False

#     index_counter = 0

#     for v in graph._graph:
#         if index[v] == -1:
#             stack.append((v, 0))
#             while stack:
#                 current, state = stack.pop()
#                 if state == 0:
#                     index[current] = index_counter
#                     low_link[current] = index_counter
#                     index_counter += 1
#                     on_stack[current] = True
#                     stack.append((current, 1))
#                     for neighbor in graph.get_neighbors(current):
#                         if index[neighbor] == -1:
#                             stack.append((neighbor, 0))
#                         elif on_stack[neighbor]:
#                             low_link[current] = min(low_link[current], index[neighbor])
#                 elif state == 1:
#                     for neighbor in graph.get_neighbors(current):
#                         if on_stack[neighbor]:
#                             low_link[current] = min(low_link[current], index[neighbor])
#                     if low_link[current] == index[current]:
#                         scc = []
#                         while True:
#                             node = stack.pop()[0]
#                             on_stack[node] = False
#                             scc.append(node)
#                             if node == current:
#                                 break
#                         sccs.append(scc)

#     return sccs

# def kosaraju_iterative(graph:Graph):
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
#     for u in graph:
#         for v in graph[u]:
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

# def tarjan_iterative(graph:Graph):
#     stack = []
#     index = {}
#     low_link = {}
#     on_stack = {}
#     sccs = []

#     for v in graph._graph:
#         index[v] = -1
#         low_link[v] = -1
#         on_stack[v] = False

#     index_counter = 0

#     for v in graph._graph:
#         if index[v] == -1:
#             stack.append((v, 0))
#             while stack:
#                 current, state = stack.pop()
#                 if state == 0:
#                     index[current] = index_counter
#                     low_link[current] = index_counter
#                     index_counter += 1
#                     on_stack[current] = True
#                     stack.append((current, 1))
#                     for neighbor in graph.get_neighbors(current):
#                         if index.get(neighbor, -1) == -1:
#                             stack.append((neighbor, 0))
#                         elif on_stack[neighbor]:
#                             low_link[current] = min(low_link[current], index[neighbor])
#                 elif state == 1:
#                     for neighbor in graph.get_neighbors(current):
#                         if on_stack.get(neighbor, False):
#                             low_link[current] = min(low_link[current], index.get(neighbor, -1))
#                     if low_link[current] == index[current]:
#                         scc = []
#                         while stack:
#                             node = stack.pop()[0]
#                             on_stack[node] = False
#                             scc.append(node)
#                             if node == current:
#                                 break
#                         sccs.append(scc)

#     return sccs

def kosaraju_iterative(graph:Graph):    #Funciona
    stack = []
    visited = set()
    sccs = []

    for vertex in graph._graph:
        if vertex not in visited:
            stack.append(vertex)
            visited.add(vertex)
            while stack:
                current = stack[-1]
                if current in graph._graph:
                    neighbors = [neighbor for neighbor in graph.get_neighbors(current) if neighbor not in visited]
                    if neighbors:
                        stack.append(neighbors[0])
                        visited.add(neighbors[0])
                    else:
                        stack.pop()
                        sccs.append(current)
                else:
                    stack.pop()

    reversed_graph = {}
    for u in graph._graph:
        for v in graph.get_neighbors(u):
            if v not in reversed_graph:
                reversed_graph[v] = set()
            reversed_graph[v].add(u)

    visited.clear()
    scc_components = []
    for node in reversed(sccs):
        if node not in visited:
            stack = [node]
            component = []
            visited.add(node)
            while stack:
                current = stack.pop()
                component.append(current)
                for neighbor in reversed_graph.get(current, []):
                    if neighbor not in visited:
                        stack.append(neighbor)
                        visited.add(neighbor)
            scc_components.append(component)

    return scc_components

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
        page_graph.add_edge(str(edge[0]), str(edge[1]))

sccs = kosaraju_iterative(page_graph)
# sccs = tarjan.run()

# Find the largest SCC by size
largest_scc = max(sccs, key=len)

# print(f"The largest SCC is: {largest_scc}")
print(f"Size of the largest SCC: {len(largest_scc)}")
print(f"Total amount of SCC:{len(sccs)}")