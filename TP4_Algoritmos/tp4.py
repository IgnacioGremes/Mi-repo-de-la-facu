from graph import Graph
from typing import List, Dict, Set, Optional, Any

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

# def iterative_dfs(graph: Graph, start_vertex: str) -> List[str]:
#     if not graph.vertex_exists(start_vertex):
#         raise ValueError(f"Vertex {start_vertex} does not exist in the graph")

#     visited = set()
#     stack = [start_vertex]

#     while stack:
#         vertex = stack.pop()
#         if vertex not in visited:
#             visited.add(vertex)
#      
#             # Add neighbors in reverse order to visit them in correct order
#             neighbors = graph.get_neighbors(vertex)
#             for neighbor in reversed(neighbors):
#                 stack.append(neighbor)

#     return visited

def find_WCC(graph: Graph) -> List[Set[str]]:
    def iterative_dfs(start_vertex: str, visited: Set[str],components) -> Set[str]:
        stack = [start_vertex]
        component = set()
        while stack:
            vertex = stack.pop()
            if vertex not in visited:
                visited.add(vertex) 
                component.add(vertex)
                # Add neighbors in reverse order to visit them in correct order
                # neighbors = graph.get_neighbors(vertex)
                # for neighbor in reversed(neighbors):
                #     stack.append(neighbor)
                for neighbor in graph.get_neighbors(vertex):
                    if neighbor not in visited:
                        stack.append(neighbor)
                #     else:
                #         for i in range(len(components)):
                #             if neighbor in components[i]:
                #                 components[i] = components[i].union(component)
                #                 return
                                
                            

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

page_graph = Graph()
#/home/ig/Desktop/Repositorios Git/Mi Repositorio/Mi-repo-de-la-facu/TP4_Algoritmos/web-Google.txt
#'C:/Users/iegre/OneDrive/Escritorio/repositorio Git/Mi Repositorio/Mi-repo-de-la-facu/TP4_Algoritmos/web-Google.txt'
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
        page_graph.add_edge(str(edge[1]), str(edge[0]))

wccs = find_WCC(page_graph)

# Find the largest SCC by size
largest_wcc = max(wccs, key=len)

# print(f"The largest SCC is: {largest_scc}")
print(f"Size of the largest SCC: {len(largest_wcc)}")
print(f"Total amount of SCC:{len(wccs)}")