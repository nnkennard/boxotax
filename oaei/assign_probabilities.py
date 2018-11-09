import sys
import collections


DUMMY_BASE_PROBABILITY = 1.0
ROOT_STRING = "!!ROOT"
ROOT_INDEX = "0"


def assign_conditional_probabilities(root_node, conditional_probabilities,
    graph, probabilities):
  if root_node in graph:
    for child in graph[root_node]:
      conditional_probabilities[child][root_node] =probabilities[
          child]/probabilities[root_node]
      conditional_probabilities[root_node][child] = 1.0
      assign_conditional_probabilities(child, conditional_probabilities,
          graph, probabilities)



def assign_probabilities(root_node, probabilities, graph):
  if root_node not in probabilities:
    total_prob = 0.0
    for child in graph[root_node]:
      total_prob += assign_probabilities(child, probabilities, graph)
    probabilities[root_node] = total_prob

  return probabilities[root_node]

def bfs_order(root, graph):
  bfs = []
  to_visit = [root]
  while len(to_visit):
    current = to_visit.pop(0)
    bfs.append(current)
    to_visit+=[child for child in graph[current] 
        if child not in bfs 
        and child not in to_visit]
  return bfs  

def transitive_reduction(root_node, graph):
  all_nodes = set(graph.keys())
  all_nodes.update(sum(graph.values(), []))
  intransitive_graph_constructor = collections.defaultdict(set)
  for parent, children in graph.iteritems():
    intransitive_graph_constructor[parent].update(children)


  for i in bfs_order(root_node, graph):
    for j in graph[i]:
      for k in graph[j]:
        if k in graph[i]:
          intransitive_graph_constructor[i] -= set([k])

  return intransitive_graph_constructor



def main():

  input_file = sys.argv[1]
  graph = collections.defaultdict(list)

  with open(input_file, 'r') as f:
    for line in f:
      parent, child, _, _ = line.split()
      graph[parent].append(child)

  superclass_nodes = set(graph.keys())
  subclass_nodes = set(sum(graph.values(),[]))
  superclass_only_nodes = superclass_nodes - subclass_nodes
  subclass_only_nodes = subclass_nodes - superclass_nodes


  intransitive_graph = transitive_reduction(ROOT_INDEX, graph)


  probabilities = {}
  for node in subclass_only_nodes:
    probabilities[node] = DUMMY_BASE_PROBABILITY

  total = assign_probabilities(ROOT_INDEX, probabilities, intransitive_graph)

  conditional_probabilities = collections.defaultdict(dict)

  assign_conditional_probabilities(ROOT_INDEX,
      conditional_probabilities, graph, probabilities)

  for a, b_probs in conditional_probabilities.iteritems():
    for b, conditional_prob in b_probs.iteritems():
      print(a+"\t"+b+"\t"+str(conditional_prob))


  for key, value in probabilities.iteritems():
    probabilities[key] = value/total
if __name__ == "__main__":
  main()
