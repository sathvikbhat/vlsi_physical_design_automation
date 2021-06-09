import sys
import random
import numpy
import math
import matplotlib.pyplot as plt
import PIL
from PIL import Image, ImageDraw


GATE_SIZES = {}
GATE_SIZES['NOT'] = [2,1]
GATE_SIZES['NAND2'] = [2,2]
GATE_SIZES['NAND3'] = [4,2]
GATE_SIZES['NAND4'] = [4,6]
GATE_SIZES['NOR2'] = [2,3]
GATE_SIZES['NOR3'] = [2,5]
GATE_SIZES['NOR4'] = [3,6]
GATE_SIZES['AND2'] = [4,3]
GATE_SIZES['AND3'] = [4,6]
GATE_SIZES['AND4'] = [5,8]
GATE_SIZES['OR2'] = [3,4]
GATE_SIZES['OR3'] = [6,4]
GATE_SIZES['OR4'] = [5,8]
GATE_SIZES['DFF'] = [10,10]


class node:
    def __init__(self, name, node_type = ''):
        self.name = name
        self.node_type = node_type
        self.output = False
        
        self.adjacent = []
        self.adjacent_distance = []
        
        self.input_nodes = []
        self.output_nodes = []
        
    # Graph BFS, DFS, Shortest Path Algo
        self.visited = False
        self.distance = 0
        self.parent_node = None
        
    # Partitioning algorithm        
        self.kl_locked = False
     
    # Floorplan parameters
        self.floorplan_dimensions = []
        self.slicing_tree_left_child = None
        self.slicing_tree_right_child = None
        self.floorplan_center_vertices = []
        
    #Placement Parameters
        self.placement_dimensions = []
        self.placement_left_vertices = []
        self.placement_orientation = 0
        
    #Routing Parameters
        self.routing_input_positions = []
        self.routing_input_position_number = 0
        self.routing_output_position_number = 0
        self.routing_output_positions = []
        self.routing_vertices = []

        
    def add_child(self, child_node):
        self.adjacent.append(child_node)
    
    def isleaf(self):
        return (self.adjacent == [])
        

class routing_node:
    def __init__(self, position = []):
        self.routing_blocked = False
        self.adjacent = []
        
        self.cost = 65535
        self.position = position
        self.routing_prev_dir = 'E'
        self.source_node = False
        
        
        
class undirected_graph:
    
    def __init__(self, file_name):
        self.all_nodes = []
        self.adjacency_matrix = []
        self.create_graph(file_name)
        self.slicing_tree_expression = []

    def search_node(self, node_name):
        for x in self.all_nodes:
            if x.name == node_name:
                return x
        new_node = self.create_new_node(node_name)
        self.all_nodes.append(new_node)
        return new_node
    
    def create_new_node(self, node_name, node_type=''):
        new_node = node(node_name, node_type)
        return new_node
    
    def add_nodes(self, node_name, node_type, adjacency_list):
        new_node = self.search_node(node_name)
        new_node.node_type = node_type
        for i in range(len(adjacency_list)):
            dist = random.randint(1,9)
            adjacent_node = self.search_node(adjacency_list[i])
            new_node.adjacent.append(adjacent_node)
            new_node.adjacent_distance.append(dist)
            new_node.input_nodes.append(adjacent_node)
            adjacent_node.adjacent.append(new_node)
            adjacent_node.adjacent_distance.append(dist)
            adjacent_node.output_nodes.append(new_node)
    
    def create_adjacency_matrix(self):
        for curr_node in self.all_nodes:
            adjacent_list = []
            for all_node in self.all_nodes:
                if all_node not in curr_node.adjacent:
                    adjacent_list.append(0)
                else:
                    adjacent_list.append(curr_node.adjacent_distance[curr_node.adjacent.index(all_node)])
            self.adjacency_matrix.append(adjacent_list)

    def create_graph(self, file_name):
        fo = open(file_name, "r")
        for line in fo:
            line = line.strip()
            
            if line.startswith("#") or line=="":
                pass
            
            elif line.startswith("INPUT"):
                start_index = line.find("(")
                end_index = line.find(")")
                node_name = line[start_index + 1: end_index]
                new_node = self.create_new_node(node_name, 'INPUT')
                self.all_nodes.append(new_node)
            
            elif line.startswith("OUTPUT"):
                start_index = line.find("(")
                end_index = line.find(")")
                node_name = line[start_index + 1: end_index]
                new_node = self.create_new_node(node_name, 'OUTPUT')
                new_node.output = True
                self.all_nodes.append(new_node)

            else:
                split_string = line.split("=")
                node_name = split_string[0].strip()
                
                start_index = split_string[1].find("(")
                end_index = split_string[1].find(")")
                node_type = split_string[1][:start_index]
                
                
                sub_string = split_string[1][start_index + 1: end_index]
                adjacent_nodes = []
                start_index = 0
                num = sub_string.find(",")
                num_fanout = 1
                while(num != -1):
                    adjacent_nodes.append(sub_string[start_index:num].strip())
                    num_fanout = num_fanout + 1
                    sub_string = sub_string[num+1:]
                    num = sub_string.find(",")
                adjacent_nodes.append(sub_string[start_index:end_index].strip())
                
                node_type = node_type.strip()
                if node_type.startswith('NOT'):
                    node_type = 'NOT'
                elif node_type.startswith('DFF'):
                    node_type = 'DFF'
                else:
                    node_type = node_type + str(num_fanout)
                
                self.add_nodes(node_name, node_type, adjacent_nodes)
        
        fo.close()
        self.create_adjacency_matrix()
        
    def print_graph(self):
        for x in self.all_nodes:
            print(x.name, '(' + x.node_type + ')', end = "\tAdjacent Nodes:  ")
            for i in range(len(x.adjacent)):
                print(x.adjacent[i].name + "(distance=" + str(x.adjacent_distance[i]) + ')',end = ", ")
            print()
    
    def print_adjacency_matrix(self):
        print('\nOrder: ')
        for x in self.all_nodes:
            print(x.name, end = ' - ')
        print()
        
        for x in self.adjacency_matrix:
            print(x)
            
    def get_graph(self):
        return self.all_nodes


#Depth First Search Algorithm
    def dfs(self, first_node_name):
        print("Depth first search of the graph with ", first_node_name, " as the starting node is: ")

        for x in self.all_nodes:
            x.visited = False
            if x.name == first_node_name:
                first_node = x
        self.depth_first_search(first_node)
    
    def depth_first_search(self, curr_node):
        print(curr_node.name, end = "-> ")
        curr_node.visited = True
        for x in curr_node.adjacent:
            if x.visited == False:
                self.depth_first_search(x)

                
#Breadth First Search Algorithm
    def bfs(self, first_node_name):
        print("Breadth first search of the graph with ", first_node_name, " as the starting node is: ")
        queue = []
        for x in self.all_nodes:
            if x.name == first_node_name:
                first_node = x
                first_node.distance = 0
            else:
                x.distance = -1
        
        queue.append(first_node)
        
        while(queue):
            curr_node = queue.pop(0)
            print(curr_node.name,"(", curr_node.distance,")", end = "-> ", sep = "")
            for adj_node in curr_node.adjacent:
                if adj_node.distance == -1:
                    adj_node.distance = curr_node.distance + 1
                    queue.append(adj_node)
                    

#Shortest Path Algorithm
    def dijsktra_shortest_path(self, first_node_name):
        for x in self.all_nodes:
            if x.name == first_node_name:
                first_node = x
                first_node.distance = 0
                first_node.visited = True
                first_node.parent_node = first_node
            else:
                x.distance = 65535
                x.visited = False
                x.parent_node = None
        
        S = []
        curr_node = first_node
        
        while len(S)!=len(self.all_nodes):
            S.append(curr_node)

            for i in range(len(curr_node.adjacent)):
                if curr_node.adjacent[i].distance > curr_node.distance + curr_node.adjacent_distance[i]:
                    curr_node.adjacent[i].distance = curr_node.distance + curr_node.adjacent_distance[i]
                    curr_node.adjacent[i].parent_node = curr_node
            
            min_distance = 65535            
            for x in self.all_nodes:
                if x not in S and min_distance > x.distance:
                    min_distance = x.distance
                    curr_node = x
        
        total_weight = 0
        print("Shortest path to all nodes from", first_node_name, ":")
        for x in S:
            print("Node:", x.name, "\tDistance=", x.distance, "\tParent Node:", x.parent_node.name)
            for i in range(len(x.adjacent)):
                if x.adjacent[i] == x.parent_node:
                    total_weight = total_weight + x.adjacent_distance[i]
                    break
        
        print('\nTotal sum of edge weights in the shortest path:', total_weight)
    
            
    def print_shortest_path(self, starting_node_name, ending_node_name):
        for x in self.all_nodes:
            if x.name == starting_node_name:
                starting_node = x
            if x.name == ending_node_name:
                ending_node = x
        
        self.dijsktra_shortest_path(starting_node_name)
        curr_node = ending_node
        path = []
        
        while(curr_node != starting_node):
            path.insert(0, curr_node)
            curr_node = curr_node.parent_node
        path.insert(0, curr_node)
        
        print("\nShortest Path From", starting_node_name, "to", ending_node_name, ":")
        for x in path:
            print(x.name, end = "-->")
            

#Minimum Spanning Tree Algorithm
    def print_prim_algorithm(self, current_node):
        sum_weight = 0
        for x in current_node.adjacent:
            if x.parent_node == current_node:
                print(current_node.name, '-', x.name, ': \t Weight=', x.distance)
                sum_weight = sum_weight + x.distance + self.print_prim_algorithm(x)
        return sum_weight
                
    def prim_minimum_spanning_tree(self, starting_node_name):
        S = []
        S_bar = []
        for curr_node in self.all_nodes:
            if curr_node.name == starting_node_name:
                starting_node = curr_node
                starting_node.parent_node = None
                starting_node.distance = 0
                S_bar.append(starting_node)
            else:
                curr_node.parent_node = None
                curr_node.distance = 65535
                S_bar.append(curr_node)
        
        while(S_bar):
            min_dist = 65535
            for x in S_bar:
                if x.distance <  min_dist:
                    curr_node = x
                    min_dist = x.distance
            S_bar.remove(curr_node)
            S.append(curr_node)
            
            for i in range(len(curr_node.adjacent)):
                if curr_node.adjacent[i].distance > curr_node.adjacent_distance[i] and curr_node.adjacent[i] not in S:
                    curr_node.adjacent[i].distance = curr_node.adjacent_distance[i]
                    curr_node.adjacent[i].parent_node = curr_node
        
        print('\nPrim\'s Minimum spanning Tree:' )
        sum_weight = self.print_prim_algorithm(starting_node)
        print('\nSum of all edge weights = ', sum_weight)
        

#K-L Partitioning
    def calculate_D_value(self, first_partition, second_partition):
        D_first = []
        D_second = []
        cut_size = 0
        for curr_node in first_partition:
            c_internal = 0
            c_external = 0
            for i in range(len(curr_node.adjacent)):
                if curr_node.adjacent[i] not in first_partition:
                    c_external = c_external + curr_node.adjacent_distance[i]
                    cut_size = cut_size + curr_node.adjacent_distance[i]
                else:
                    c_internal = c_internal + curr_node.adjacent_distance[i]
            
            D_value_curr_node = c_external - c_internal
            D_first.append(D_value_curr_node)

        for curr_node in second_partition:
            c_internal = 0
            c_external = 0
            for i in range(len(curr_node.adjacent)):
                if curr_node.adjacent[i] not in second_partition:
                    c_external = c_external + curr_node.adjacent_distance[i]
                else:
                    c_internal = c_internal + curr_node.adjacent_distance[i]
            
            D_value_curr_node = c_external - c_internal
            D_second.append(D_value_curr_node)
        
        return D_first, D_second, cut_size

            
    def kl_partition(self):
        
        first_partition = [self.all_nodes[i] for i in range(len(self.all_nodes)//2)]
        second_partition = [self.all_nodes[i] for i in range(len(self.all_nodes)//2, len(self.all_nodes))]
        if len(self.all_nodes)%2 != 0:
            first_partition.append(self.create_new_node('dummy'))
        
        print('Initial Partitions: ')
        print('First Partition: \t', end = ' ')
        for x in first_partition:
            print(x.name, end = ' ')
        print()
        print('Second Partition: \t', end = ' ')
        for x in second_partition:
            print(x.name, end = ' ')
        print()
        
        D_first, D_second, cut_size = self.calculate_D_value(first_partition, second_partition)
        print('D values for First partition: ', D_first)
        print('D values for Second Partition: ', D_second)
        print('\nInitial Cut size = ', cut_size)
        
        iteration = 1        
        
        while(1):
            print('\nIteration: ', iteration)
            g_array = []
                        
            first_partition_swaps = []
            second_partition_swaps = []
            
            for i in range(len(D_first)):
                g_max = -1000
                
                for x in range(len(first_partition)):
                    if first_partition[x].kl_locked == False:
                        for y in range(len(second_partition)):
                            if second_partition[y].kl_locked == False:
                                if second_partition[y] not in first_partition[x].adjacent:
                                    g_current_nodes = D_first[x] + D_second[y]
                                else:
                                    g_current_nodes = D_first[x] + D_second[y] - (2*first_partition[x].adjacent_distance[first_partition[x].adjacent.index(second_partition[y])])

                                if g_max <= g_current_nodes:
                                    g_max = g_current_nodes
                                    first_partition_node = first_partition[x]
                                    second_partition_node = second_partition[y]
                
                first_partition_node.kl_locked = True
                second_partition_node.kl_locked = True
                
                first_partition_swaps.append(first_partition_node)
                second_partition_swaps.append(second_partition_node)
                
                for x in range(len(first_partition)):
                    if first_partition[x].kl_locked == False:
                        if first_partition_node not in first_partition[x].adjacent:
                            D_first[x] = D_first[x]
                        else:
                            first_index = first_partition[x].adjacent.index(first_partition_node)
                            D_first[x] = D_first[x] + (2 * first_partition[x].adjacent_distance[first_index])
                        
                        if second_partition_node not in first_partition[x].adjacent:
                            D_first[x] = D_first[x]
                        else:
                            second_index = first_partition[x].adjacent.index(second_partition_node)
                            D_first[x] = D_first[x] - (2 * first_partition[x].adjacent_distance[second_index])

                for x in range(len(second_partition)):
                    if second_partition[x].kl_locked == False:
                        if first_partition_node not in second_partition[x].adjacent:
                            D_second[x] = D_second[x]
                        else:
                            first_index = second_partition[x].adjacent.index(first_partition_node)
                            D_second[x] = D_second[x] - (2 * second_partition[x].adjacent_distance[first_index])
                        
                        if second_partition_node not in second_partition[x].adjacent:
                            D_second[x] = D_second[x]
                        else:
                            second_index = second_partition[x].adjacent.index(second_partition_node)
                            D_second[x] = D_second[x] + (2 * second_partition[x].adjacent_distance[second_index])

                g_array.append(g_max)
                print("Locked nodes for swapping: ", first_partition_node.name, ' - ', second_partition_node.name, end = '\t\t')
                print("g" + str(i+1) +  " value: ", g_max)
            
            
            max_partial_sum = -1000
            max_index = 0
            for m in range(len(g_array)):
                partial_sum = 0
                for n in range(m+1):
                    partial_sum = partial_sum + g_array[n]
                if partial_sum > max_partial_sum:
                    max_partial_sum = partial_sum
                    max_index = m
            
            print('\ng Values after locking all nodes in', iteration, 'iteration: ', g_array)
            print('Maximum Partial Sum = ', max_partial_sum)
            
            if max_partial_sum <= 0:
                print('Maximum Partial Sum is less than 0!. Ending algorithm')
                break
                
            print('Swapping Nodes: ')
            for m in range(max_index+1):
                print(first_partition_swaps[m].name, '-', second_partition_swaps[m].name)

            
            for m in range(max_index+1):
                first_partition[first_partition.index(first_partition_swaps[m])] = second_partition_swaps[m]
                second_partition[second_partition.index(second_partition_swaps[m])] = first_partition_swaps[m]
            
            
            D_first, D_second, cut_size = self.calculate_D_value(first_partition, second_partition)
            print('\nAfter Swapping: ')
            print("\nPartition1:\t", end = ' ')
            for x in first_partition:
                print(x.name, end=' ')
            print()
            print('D values\t', D_first)
            print("Partition2:\t", end = ' ')
            for x in second_partition:
                print(x.name, end=' ')
            print()
            print('D Values\t', D_second)
            print('\nCut Size = ', cut_size)
            
            for x in first_partition:
                x.kl_locked = False
            for x in second_partition:
                x.kl_locked = False

            iteration = iteration + 1
        
        print('\n\nFinal Partitions: ')
        print('First Partition: \t', end = ' ')
        for x in first_partition:
            print(x.name, end = ' ')
        print()
        print('Second Partition: \t', end = ' ')
        for x in second_partition:
            print(x.name, end = ' ')
        print()
        print('Final Cut Size = ', cut_size)
        
        
#Partitioning using Simulated Annealing
    def simulated_annealing_partition(self, init_temp = 20, r = 0.9, cost_lambda = 1):
        
        first_partition = [self.all_nodes[i] for i in range(len(self.all_nodes)//2)]
        second_partition = [self.all_nodes[i] for i in range(len(self.all_nodes)//2, len(self.all_nodes))]
                
        print('First Partition: \t', end = ' ')
        for x in first_partition:
            print(x.name, end = ' ')
        print()
        print('Second Partition: \t', end = ' ')
        for x in second_partition:
            print(x.name, end = ' ')
        print()
        cost = self.compute_cost(first_partition, second_partition, cost_lambda)
        print('Initial Cost', cost)
        print()
        
        all_costs = []
        
        best_first_partition = [x for x in first_partition]
        best_second_partition = [x for x in second_partition]
        best_cost = self.compute_cost(first_partition, second_partition, cost_lambda)
        
        termination = 5                              # Terminate if past n best costs are same
        acceptance_prob = 0.5                        # Minimum probability for acceptance for an uphill move
        max_uphill_moves = 10                        # Maximum accepted successive uphill moves
        max_total_iters = 10*len(self.all_nodes)     # Maximum random selections for a given temperature

        current_first_partition = [x for x in best_first_partition]
        current_second_partition = [x for x in best_second_partition]
        current_cost = best_cost
        all_costs.append(current_cost)
        current_temperature = init_temp
        
        best_cost_array = []
        moves_p_selection = []
        frozen = False
        i=0
        while not (frozen):
            
            print('\nCurrent Temperature', current_temperature)
            continuous_uphill_moves = 0
            total_iters = 0
            total_uphill_moves = 0
            total_accepted_moves = 0
            
            while not (total_iters > max_total_iters or continuous_uphill_moves > max_uphill_moves):
                i = i+1
                random_index = random.randint(0,len(self.all_nodes) - 1)
                selected_node = self.all_nodes[random_index]
                print("\nIteration: ", i)
                print("Selected Node:", selected_node.name, end = '\t')
                
                if selected_node not in current_first_partition:
                    transferred_first_partition = [x for x in current_first_partition]
                    transferred_first_partition.append(selected_node)
                    transferred_second_partition = []
                    for x in current_second_partition:
                        if x!= selected_node:
                            transferred_second_partition.append(x)
                
                else:
                    transferred_second_partition = [x for x in current_second_partition]
                    transferred_second_partition.append(selected_node)
                    transferred_first_partition = []
                    for x in current_first_partition:
                        if x != selected_node:
                            transferred_first_partition.append(x)
                
                transfer_cost = self.compute_cost(transferred_first_partition, transferred_second_partition, cost_lambda)
                print('Transfer Cost', transfer_cost)
                
                delta_cost = transfer_cost - current_cost
                
                if delta_cost < 0:
                    current_cost = transfer_cost
                    current_first_partition = [x for x in transferred_first_partition]
                    current_second_partition = [x for x in transferred_second_partition]

                    all_costs.append(current_cost)
                    total_accepted_moves = total_accepted_moves + 1
                    continuous_uphill_moves = 0
                    print('Move Accepted')
                    
                    if current_cost < best_cost:
                        best_first_partition = [x for x in current_first_partition]
                        best_second_partition = [x for x in current_second_partition]
                        best_cost = current_cost
                        
                elif acceptance_prob < math.exp(-delta_cost/current_temperature):
                    current_cost = transfer_cost
                    current_first_partition = [x for x in transferred_first_partition]
                    current_second_partition = [x for x in transferred_second_partition]
                    total_uphill_moves = total_uphill_moves + 1
                    continuous_uphill_moves = continuous_uphill_moves + 1
                    
                    total_accepted_moves = total_accepted_moves + 1
                    print('Move Accepted')
                    all_costs.append(current_cost)
                    
                else:
                    total_uphill_moves = total_uphill_moves + 1
                    continuous_uphill_moves = continuous_uphill_moves + 1
                    print('Move Not Accepted')
                
                print('First Partition: \t', end = ' ')
                for x in current_first_partition:
                    print(x.name, end = ' ')
                print()
                print('Second Partition: \t', end = ' ')
                for x in current_second_partition:
                    print(x.name, end = ' ')
                print()
                print('Cost: ', self.compute_cost(current_first_partition, current_second_partition, cost_lambda))

                total_iters = total_iters + 1
            
            current_temperature = current_temperature * r
            
            best_cost_array.append(best_cost)
            moves_p_selection.append(total_accepted_moves/total_uphill_moves)
            if len(best_cost_array) > termination:
                if best_cost_array[-1] == best_cost_array[-termination]:
                    frozen = True
                    print("No better partition since last", termination,"temperature updates")
                    print("Stopping Algorithm!")

        print('Cost Curve:')
        plt.plot(all_costs)
        plt.xlabel('iter')
        plt.ylabel('Cost')
        plt.show()

        print('Best First Partition: \n\t', end = ' ')
        for x in best_first_partition:
            print(x.name, end = ' ')
        print()
        print('Best Second Partition: \n\t', end = ' ')
        for x in best_second_partition:
            print(x.name, end = ' ')
        print()
        final_best_cost = self.compute_cost(best_first_partition, best_second_partition, cost_lambda)
        print('Cost: ', final_best_cost)
        
        D_first, D_second, cut_size = self.calculate_D_value(best_first_partition, best_second_partition)
        print('Cut Size: ', cut_size)
        print('Balance Size for First Partition: ', len(best_first_partition)/(len(best_first_partition) + len(best_second_partition)))
          
    
    def compute_cost(self, first_partition, second_partition, cost_lambda):
        cut_cost = 0
        for x in first_partition:
            for i in range(len(x.adjacent)):
                if x.adjacent[i] not in first_partition:
                    cut_cost = cut_cost + x.adjacent_distance[i]
        
        balance_cost = abs(len(first_partition) - len(second_partition))
        return cut_cost + (cost_lambda*balance_cost)
    
    
    
# Floorplanning Simulated Annealing  
    def floorplan_simulated_annealing(self, GATE_SIZES, T0 = 100, Tf = 10, r = 0.85, k=10, lambda_cost = 0.5):
        slicing_tree = []
        
        for x in self.all_nodes:
            if x.node_type == 'INPUT' or x.node_type == 'OUTPUT':
                pass
            else:
                slicing_tree.append(x)
                x.floorplan_dimensions = GATE_SIZES[x.node_type]
        number_of_operands = len(slicing_tree)
        slice_cut = 'V'
        
        self.all_gates = [x for x in slicing_tree]
        
        top_node = slicing_tree.pop(0)
        top_node.floorplan_center_vertices = [top_node.floorplan_dimensions[0]/2, top_node.floorplan_dimensions[1]/2]
        
        while(slicing_tree):
            new_node = node(slice_cut)
            right_child = slicing_tree.pop(0)
            new_node.slicing_tree_right_child = right_child
            new_node.slicing_tree_left_child = top_node
            
            if slice_cut == 'V':
                width_newnode = top_node.floorplan_dimensions[0] + right_child.floorplan_dimensions[0]
                height_newnode = max(top_node.floorplan_dimensions[1], right_child.floorplan_dimensions[1])
                new_node.floorplan_dimensions = [width_newnode, height_newnode]
                
                x_vertex = top_node.floorplan_dimensions[0] + right_child.floorplan_dimensions[0]/2
                y_vertex = right_child.floorplan_dimensions[1]/2
                right_child.floorplan_center_vertices = [x_vertex, y_vertex]
            
            else:
                width_newnode = max(top_node.floorplan_dimensions[0], right_child.floorplan_dimensions[0])
                height_newnode = top_node.floorplan_dimensions[1] + right_child.floorplan_dimensions[1]
                new_node.floorplan_dimensions = [width_newnode, height_newnode]

                x_vertex = right_child.floorplan_dimensions[0]/2
                y_vertex = top_node.floorplan_dimensions[1] + right_child.floorplan_dimensions[1]/2
                right_child.floorplan_center_vertices = [x_vertex, y_vertex]

            top_node = new_node
            top_node.floorplan_vertices = [top_node.floorplan_dimensions[0]/2, top_node.floorplan_dimensions[1]/2]
                    
            
        print('Slicing Tree Traversal:')
        self.slicing_tree_expression = []
        self.floorplan_traversal(top_node)
        print()

        initial_cost = self.floorplan_cost(top_node, lambda_cost)
        print("Initial Cost = ", initial_cost)

        print('Slicing Tree Expression: ')
        for x in self.slicing_tree_expression:
            print(x.name, end='-')
        print()
        
        self.plotting_graph(top_node, self.slicing_tree_expression)
        
        best_slicing_tree = [x for x in self.slicing_tree_expression]
        best_floorplan_cost = initial_cost
        best_tree_top_node = top_node
        
        current_slicing_tree = [x for x in self.slicing_tree_expression]
        current_floorplan_cost = initial_cost
        current_tree_top_node = top_node
        current_gates = [x for x in self.all_gates]
        
        
        curr_temp = T0
        number_of_gates = len(self.all_gates)
        no_steps_per_temperature = k * number_of_gates
        frozen = False 
        
        print('\n\n\nSimulated Annealing Process\n')
        
        while not frozen:
            print('\n\n\n\nCurrent Temperature = ', curr_temp)
            iteration = 1
            
            print('Current Expression: ')
            for x in current_slicing_tree:
                print(x.name, end = '-')
            print('\n')
            
            while iteration <= no_steps_per_temperature:
                
                print('\n\nIteration: ', iteration)
                random_move = random.randint(1,3)
                if random_move == 1:
                    print('\nM1 Move')
                    random_node1 = random.randint(0, number_of_gates-2)
                    random_node2 = random_node1 + 1

                    node1 = current_gates[random_node1]
                    node2 = current_gates[random_node2]
                    
                    transfer_slicing_tree = []
                    for x in current_slicing_tree:
                        if x.name == node1.name:
                            transfer_slicing_tree.append(node2)
                        elif x.name == node2.name:
                            transfer_slicing_tree.append(node1)
                        else:
                            transfer_slicing_tree.append(x)

                            
                    
                if random_move == 2:
                    print('\nM2 Move')
                    
                    random_operator = random.randint(0, number_of_gates-2)
                    
                    transfer_slicing_tree = [x for x in current_slicing_tree]
                    
                    pos = -1
                    index = 0
                    for i in range(len(transfer_slicing_tree)):
                        if transfer_slicing_tree[i].name == 'H' or transfer_slicing_tree[i].name == 'V':
                            pos = pos + 1
                        if pos == random_operator:
                            index = i
                            break
                    
                    if transfer_slicing_tree[index].name == 'V':
                        transfer_slicing_tree[index] = node('H')
                    elif transfer_slicing_tree[index].name == 'H':
                        transfer_slicing_tree[index] = node('V')
                    
                    for i in range(index-1, 0, -1):
                        if transfer_slicing_tree[i].name == 'H':
                            transfer_slicing_tree[i] = node('V')
                        elif transfer_slicing_tree[i].name == 'V':
                            transfer_slicing_tree[i] = node('H')
                        else:
                            break

                    for i in range(index + 1, len(transfer_slicing_tree)):
                        if transfer_slicing_tree[i].name == 'H':
                            transfer_slicing_tree[i] = node('V')
                        elif transfer_slicing_tree[i].name == 'V':
                            transfer_slicing_tree[i] = node('H')
                        else:
                            break
                    
                    
                if random_move == 3:
                    
                    print('\nM3 Move')
                    
                    transfer_slicing_tree = [x for x in current_slicing_tree]
                    
                    swap_pos = []
                    no_operators = 0
                    no_operands = 0
                    
                    
                    for i in range(len(transfer_slicing_tree)-1):
                        if transfer_slicing_tree[i].name == 'V' or transfer_slicing_tree[i].name == 'H':
                            no_operators = no_operators + 1
                            if transfer_slicing_tree[i+1].name != 'H' and transfer_slicing_tree[i].name != 'V':
                                if i + 2 != len(transfer_slicing_tree):
                                    if transfer_slicing_tree[i+2].name != transfer_slicing_tree[i].name:
                                        swap_pos.append(i)
                                else:
                                    swap_pos.append(i)
                            else:
                                pass
                        
                        else:
                            no_operands = no_operands + 1
                            if transfer_slicing_tree[i + 1].name == 'H' or transfer_slicing_tree[i+1].name == 'V':
                                if no_operators + 1 < no_operands - 1:
                                    if transfer_slicing_tree[i-1].name != transfer_slicing_tree[i+1].name:
                                        swap_pos.append(i)
                                        
                                        
                    if len(swap_pos) == 0:
                        print('M3 Not possible')
                    else:
                        swap1 = swap_pos[random.randint(0, len(swap_pos) - 1)]
                        swap2 = swap1 + 1
                        transfer_slicing_tree[swap2], transfer_slicing_tree[swap1] = transfer_slicing_tree[swap1], transfer_slicing_tree[swap2]

                    
                print('Transfer Slicing Tree')
                for x in transfer_slicing_tree:
                    print(x.name, end = ' ')
                print()
                    
                transfer_tree = self.change_slicing_tree(transfer_slicing_tree)
                transfer_cost = self.floorplan_cost(transfer_tree, lambda_cost)
                 
                print('Delta Cost = ', current_floorplan_cost - transfer_cost)
                
                acceptance_probability = random.random()/2
                
                if transfer_cost < current_floorplan_cost:
                    print('Move Accepted')
                    current_slicing_tree = [x for x in transfer_slicing_tree]
                    current_tree_top_node = self.change_slicing_tree(current_slicing_tree)
                    current_floorplan_cost = self.floorplan_cost(current_tree_top_node, lambda_cost)
                    current_gates = []
                    for x in current_slicing_tree:
                        if x.name != 'V' and x.name != 'H':
                            current_gates.append(x)
                            
                elif acceptance_probability < math.exp((current_floorplan_cost - transfer_cost)/transfer_cost*curr_temp):
                    print('Uphill Move Accepted')
                    current_slicing_tree = [x for x in transfer_slicing_tree]
                    current_tree_top_node = self.change_slicing_tree(current_slicing_tree)
                    current_floorplan_cost = self.floorplan_cost(current_tree_top_node, lambda_cost)
                    current_gates = []
                    for x in current_slicing_tree:
                        if x.name != 'V' and x.name != 'H':
                            current_gates.append(x)
                            
                else:
                    print('Move Rejected')
                    current_tree_top_node = self.change_slicing_tree(current_slicing_tree)
                    current_floorplan_cost = self.floorplan_cost(current_tree_top_node, lambda_cost)

                    
                if current_floorplan_cost < best_floorplan_cost:
                    best_slicing_tree = [x for x in current_slicing_tree]
                    best_floorplan_cost = current_floorplan_cost
                    best_tree_top_node = current_tree_top_node
                

                iteration = iteration + 1
                
            curr_temp = curr_temp * r
            if curr_temp < 20:
                frozen = True
                print('\n\nFinal Temperature Reached!!!')
                

        
        best_tree_top_node = self.change_slicing_tree(best_slicing_tree)

        print('\n\n\nBest Slicing Tree')
        for x in best_slicing_tree:
            print(x.name, x.floorplan_dimensions, end = ' -- > ')
        print()
                    
        
        self.plotting_graph(best_tree_top_node, best_slicing_tree)

        best_floorplan_cost = self.floorplan_cost(best_tree_top_node, lambda_cost)
        
        print('Best cost = ', best_floorplan_cost)
            
                    
    def plotting_graph(self, top_node, slicing_tree_expression):
        best_image_width = 720
        
        width_factor = int(best_image_width/max(top_node.floorplan_dimensions[0], top_node.floorplan_dimensions[1]))
        height_factor = width_factor
        
        w, h = top_node.floorplan_dimensions[0]*width_factor, top_node.floorplan_dimensions[1]*height_factor
        img = Image.new("RGB", (w, h), color = (255, 255, 255))
        img1 = ImageDraw.Draw(img)
        for x in slicing_tree_expression:
            if x.name != 'V' and x.name != 'H':
                shape = [((x.floorplan_center_vertices[0]-x.floorplan_dimensions[0]/2)*width_factor, (x.floorplan_center_vertices[1]-x.floorplan_dimensions[1]/2)*height_factor), ((x.floorplan_center_vertices[0]+x.floorplan_dimensions[0]/2)*width_factor, (x.floorplan_center_vertices[1]+x.floorplan_dimensions[1]/2)*height_factor)]
                img1.rectangle(shape, fill ="#ffff33", outline ="black")
        img.show()

        
    def change_slicing_tree(self, slicing_tree):
        slicing_tree_slack = []
        first_node = slicing_tree[0]
        first_node.floorplan_center_vertices = [first_node.floorplan_dimensions[0]/2, first_node.floorplan_dimensions[1]/2]
        slicing_tree_slack.append(first_node)
        
        i = 1
        while i<len(slicing_tree):
            
            current_node = slicing_tree[i]
            
            if current_node.name == 'V':
                current_node.slicing_tree_right_child = slicing_tree_slack.pop()
                current_node.slicing_tree_left_child = slicing_tree_slack.pop()
                current_node.slicing_tree_right_child.floorplan_center_vertices = [current_node.slicing_tree_right_child.floorplan_dimensions[0]/2, current_node.slicing_tree_right_child.floorplan_dimensions[1]/2]
                            
                current_node_x = current_node.slicing_tree_left_child.floorplan_dimensions[0] + current_node.slicing_tree_right_child.floorplan_dimensions[0]
                current_node_y = max(current_node.slicing_tree_left_child.floorplan_dimensions[1], current_node.slicing_tree_right_child.floorplan_dimensions[1])
                current_node.floorplan_dimensions = [current_node_x, current_node_y]
                current_node.floorplan_center_vertices = [current_node.floorplan_dimensions[0]/2, current_node.floorplan_dimensions[1]/2]

            elif current_node.name == 'H':
                current_node.slicing_tree_right_child = slicing_tree_slack.pop()
                current_node.slicing_tree_left_child = slicing_tree_slack.pop()
                current_node.slicing_tree_right_child.floorplan_center_vertices = [current_node.slicing_tree_right_child.floorplan_dimensions[0]/2, current_node.slicing_tree_right_child.floorplan_dimensions[1]/2]
                
                current_node_y = current_node.slicing_tree_left_child.floorplan_dimensions[1] + current_node.slicing_tree_right_child.floorplan_dimensions[1]
                current_node_x = max(current_node.slicing_tree_left_child.floorplan_dimensions[0], current_node.slicing_tree_right_child.floorplan_dimensions[0])
                current_node.floorplan_dimensions = [current_node_x, current_node_y]
                current_node.floorplan_center_vertices = [current_node.floorplan_dimensions[0]/2, current_node.floorplan_dimensions[1]/2]

            slicing_tree_slack.append(current_node)
            i = i+1

        top_node = slicing_tree_slack.pop()
        self.change_vertices(top_node)
        
        return top_node
    
    def change_vertices(self, curr_node):
        
        if curr_node.slicing_tree_left_child != None:
            x_vertex = curr_node.floorplan_center_vertices[0] - curr_node.floorplan_dimensions[0]/2 + curr_node.slicing_tree_left_child.floorplan_dimensions[0]/2
            y_vertex = curr_node.floorplan_center_vertices[1] - curr_node.floorplan_dimensions[1]/2 + curr_node.slicing_tree_left_child.floorplan_dimensions[1]/2
            curr_node.slicing_tree_left_child.floorplan_center_vertices = [x_vertex, y_vertex]
            self.change_vertices(curr_node.slicing_tree_left_child)

            if curr_node.name == 'V':
                x_vertex = curr_node.floorplan_center_vertices[0] - curr_node.floorplan_dimensions[0]/2 + curr_node.slicing_tree_left_child.floorplan_dimensions[0] + curr_node.slicing_tree_right_child.floorplan_dimensions[0]/2
                y_vertex = curr_node.floorplan_center_vertices[1] - curr_node.floorplan_dimensions[1]/2 + curr_node.slicing_tree_right_child.floorplan_dimensions[1]/2
            
            elif curr_node.name == 'H':
                x_vertex = curr_node.floorplan_center_vertices[0] - curr_node.floorplan_dimensions[0]/2 + curr_node.slicing_tree_right_child.floorplan_dimensions[0]/2
                y_vertex = curr_node.floorplan_center_vertices[1] - curr_node.floorplan_dimensions[1]/2 + curr_node.slicing_tree_left_child.floorplan_dimensions[1] + curr_node.slicing_tree_right_child.floorplan_dimensions[1]/2
            curr_node.slicing_tree_right_child.floorplan_center_vertices = [x_vertex, y_vertex]
            self.change_vertices(curr_node.slicing_tree_right_child)
        

    
    def floorplan_cost(self, top_node, lambda_cost=0.5):
        area_cost = top_node.floorplan_dimensions[0] * top_node.floorplan_dimensions[1]
        
        penalty_cost = 0
        if top_node.floorplan_dimensions[0] / (top_node.floorplan_dimensions[0] + top_node.floorplan_dimensions[1]) > 0.7:
            penalty_cost = area_cost
        elif top_node.floorplan_dimensions[0] / (top_node.floorplan_dimensions[0] + top_node.floorplan_dimensions[1]) < 0.3:
            penalty_cost = area_cost
            
        wiring_cost = 0
        for x in self.all_nodes:
            for i in range(len(x.adjacent)):
                y = x.adjacent[i]
                if x.node_type == 'INPUT':
                    wiring_cost = wiring_cost + (y.floorplan_center_vertices[0] * x.adjacent_distance[i])
                elif y.node_type == 'INPUT':
                    wiring_cost = wiring_cost + (x.floorplan_center_vertices[0] * x.adjacent_distance[i])
                else:
                    wiring_cost = wiring_cost + (abs(x.floorplan_center_vertices[0] - y.floorplan_center_vertices[0]) * x.adjacent_distance[i])
                    wiring_cost = wiring_cost + (abs(x.floorplan_center_vertices[1] - y.floorplan_center_vertices[1]) * x.adjacent_distance[i])
            
            if x.output:
                wiring_cost = wiring_cost + 2*(top_node.floorplan_dimensions[0])
                
        return area_cost + penalty_cost + lambda_cost*wiring_cost/2

                                                         
#Post Order Traversal Algorithm
    def floorplan_traversal(self, curr_node):
        if curr_node.slicing_tree_left_child != None:
            self.floorplan_traversal(curr_node.slicing_tree_left_child)
                
        if curr_node.slicing_tree_right_child != None:
            self.floorplan_traversal(curr_node.slicing_tree_right_child)

        print(curr_node.name, curr_node.floorplan_dimensions, end = " -> ")
        self.slicing_tree_expression.append(curr_node)
        
        
        
# Placement Simulated Annealing
    def placement_simulated_annealing(self, GATE_SIZES, target_density = 0.6, aspect_ratio = 1, routing_per_cell = 4, T0 = 200, Tf = 10, k=5, r=0.85, lambda_overlap_cost = 10):
        io_pads = []
        netlist_gates = []
        
        total_area_gates = 0
        for x in self.all_nodes:
            if x.node_type == 'INPUT':
                io_pads.append(x)
                x.placement_dimensions = [1,1]
                x.placement_left_vertices = [0,0]
            else:
                netlist_gates.append(x)
                x.placement_dimensions = GATE_SIZES[x.node_type]
                total_area_gates = total_area_gates + x.placement_dimensions[0]*x.placement_dimensions[1]
                if x.output == True:
                    new_node = node(x.name + '_OUTPUT', 'OUTPUT')
                    io_pads.append(new_node)
                    
        
        print('IO Pads')
        for x in io_pads:
            print(x.name, end=' ')
        print('\nGates')
        for x in netlist_gates:
            print(x.name, end= ' ')
        
        
        i = 0
        x_vertex = 0
        y_vertex = 0                        

        target_core_area = int(total_area_gates/target_density)
        target_height = int(math.sqrt(target_core_area/aspect_ratio))
        target_width = int(target_height * aspect_ratio)
        print('Target Area: ', target_core_area, 'Width: ', target_width, 'Height: ', target_height)
                        
        i = 0
        while i<len(netlist_gates):
            curr_gate = netlist_gates[i]
            curr_gate.placement_left_vertices = [int(curr_gate.floorplan_center_vertices[0] - curr_gate.placement_dimensions[0]/2), int(curr_gate.floorplan_center_vertices[1]- curr_gate.placement_dimensions[1]/2)]
            i = i+1
        
        self.rearrange_io_pads(io_pads, target_width, target_height)
        
        self.rearrange_netlist_gates(netlist_gates, target_width, target_height)

        self.plotting_placement(netlist_gates, io_pads, target_width, target_height)            
            
        best_placement_vertices = []
        best_placement_orientation = []
        for x in netlist_gates:
            best_placement_vertices.append([i for i in x.placement_left_vertices])
            best_placement_orientation.append(x.placement_orientation)
                      
        current_area, current_overlap_area, current_wiring_cost = self.placement_cost(netlist_gates, io_pads, target_width, target_height)
        current_placement_cost = current_wiring_cost + lambda_overlap_cost * current_overlap_area
        print('Initial placement:')
        print('Initial Core Area: ', target_width, 'X', target_height)
        print('Area Occupied: ', current_area)
        print('Overlap Area: ', current_overlap_area)
        print('Cost: ', current_placement_cost)
        density = current_area/(target_width*target_height)        
        print('Placement Density = ', density)

        best_placement_cost = current_placement_cost
        best_overlap_area = current_overlap_area
        
        curr_temp = T0
        no_moves_per_temp = k * len(netlist_gates)
        #no_moves_per_temp = 1
        
        print('No of Gates:', len(netlist_gates))
        frozen =  False
        
        print('\n\n\nSimulated Annealing Process: ')
        while not frozen:
            print('\n\nCurrent Temperature: ', curr_temp)
            iteration = 1
            
            while iteration <= no_moves_per_temp:
                print('\n\nIteration: ', iteration, ' for current temperature')
                
                random_move = random.random()
                
                if random_move <= 0.8:
                    print('M1 Move: Displace')
                    
                    random_gate = random.randint(0, len(netlist_gates) - 1)
                    transfer_gate = netlist_gates[random_gate]
                    
                    random_x_vertex = random.randint(0, target_width-1)
                    random_y_vertex = random.randint(0, target_height-1)
                    
                    initial_x_vertex = transfer_gate.placement_left_vertices[0]
                    initial_y_vertex = transfer_gate.placement_left_vertices[1]
                    
                    transfer_gate.placement_left_vertices = [random_x_vertex, random_y_vertex]
                    self.rearrange_netlist_gates(netlist_gates, target_width, target_height)
                                        
                    transfer_area, transfer_overlap_area, transfer_wiring_cost = self.placement_cost(netlist_gates, io_pads, target_width, target_height)
                    transfer_cost = transfer_wiring_cost + lambda_overlap_cost*transfer_overlap_area
                    
                    delta_cost = current_placement_cost - transfer_cost

                    print('Best Placement Cost:', best_placement_cost)
                    print('Transfer Cost:', transfer_cost)
                    print('Delta Cost', delta_cost)
                                        
                    #self.plotting_placement(netlist_gates, io_pads, target_width, target_height)
                    if transfer_cost < best_placement_cost:
                        best_placement_vertices = []
                        best_placement_orientation = []
                        for x in netlist_gates:
                            best_placement_vertices.append([x.placement_left_vertices[0], x.placement_left_vertices[1]])
                            best_placement_orientation.append(x.placement_orientation)
                        best_placement_cost = transfer_cost
                        
                    if delta_cost > 0:
                        print('Move Accepted')
                        current_placement_cost = transfer_cost
                        
                    elif math.exp(delta_cost/curr_temp) > random.random()/2:
                        current_placement_cost = transfer_cost
                        print('Uphill Move Accepted')
                    
                    else:
                        print('Move Rejected')
                    
                        transfer_gate.placement_left_vertices = [initial_x_vertex, initial_y_vertex]
                        
                        if random.random() > 0.9:
                            print('\n\n\nMove3')
                            
                            random_gate = random.randint(0, len(netlist_gates)-1)
                            transfer_gate = netlist_gates[random_gate]
                                                        
                            if transfer_gate.placement_orientation == 0:
                                transfer_gate.placement_orientation = 1
                            else:
                                transfer_gate.placement_orientation = 0
                            
                            self.rearrange_netlist_gates(netlist_gates, target_width, target_height)
                           
                            transfer_area, transfer_overlap_area, transfer_wiring_cost = self.placement_cost(netlist_gates, io_pads, target_width, target_height)
                            transfer_cost = transfer_wiring_cost + lambda_overlap_cost*transfer_overlap_area
                            
                            delta_cost = current_placement_cost - transfer_cost
                            #self.plotting_placement(netlist_gates, io_pads, target_width, target_height)

                            print('Best Placement Cost:', best_placement_cost)
                            print('Transfer Cost:', transfer_cost)
                            
                            print('Delta Cost', delta_cost)
                                                        
                            if transfer_cost < best_placement_cost:
                                best_placement_vertices = []
                                best_placement_orientation = []
                                for x in netlist_gates:
                                    best_placement_vertices.append([x.placement_left_vertices[0], x.placement_left_vertices[1]])
                                    best_placement_orientation.append(x.placement_orientation)
                                best_placement_cost = transfer_cost
                                
                            if delta_cost > 0:
                                print('Move Accepted')
                                current_placement_cost = transfer_cost
                                
                            elif math.exp(delta_cost/curr_temp) > random.random()/2:
                                print('Uphill Move Accepted')
                                current_placement_cost = transfer_cost
                                
                            else:  
                                print('Move Rejected')
                                if transfer_gate.placement_orientation == 0:
                                    transfer_gate.placement_orientation = 1
                                else:
                                    transfer_gate.placement_orientation = 0

                                
                else:
                    print('M2 Move: Interchange')
                    
                    random_index1 = random.randint(0, len(netlist_gates) - 1)
                    random_index2 = random.randint(0, len(netlist_gates) - 1)
                    
                    transfer_gate1 = netlist_gates[random_index1]
                    transfer_gate2 = netlist_gates[random_index2]
                    
                    x_vertices1 = transfer_gate1.placement_left_vertices[0]
                    y_vertices1 = transfer_gate1.placement_left_vertices[1]
                    x_vertices2 = transfer_gate2.placement_left_vertices[0]
                    y_vertices2 = transfer_gate2.placement_left_vertices[1]
                    
                    transfer_gate1.placement_left_vertices = [x_vertices2, y_vertices2]
                    transfer_gate2.placement_left_vertices = [x_vertices1, y_vertices1]
                    
                    self.rearrange_netlist_gates(netlist_gates, target_width, target_height)
                    
                    transfer_area, transfer_overlap_area, transfer_wiring_cost = self.placement_cost(netlist_gates, io_pads, target_width, target_height)
                    transfer_cost = transfer_wiring_cost + lambda_overlap_cost*transfer_overlap_area
                            
                    delta_cost = current_placement_cost - transfer_cost
                    #self.plotting_placement(netlist_gates, io_pads, target_width, target_height)

                    
                    if transfer_cost < best_placement_cost:
                        best_placement_vertices = []
                        best_placement_orientation = []
                        for x in netlist_gates:
                            best_placement_vertices.append([x.placement_left_vertices[0], x.placement_left_vertices[1]])
                            best_placement_orientation.append(x.placement_orientation)
                        best_placement_cost = transfer_cost
                        
                    if delta_cost > 0:
                        print('Move Accepted')
                        current_placement_cost = transfer_cost
                        
                    elif math.exp(delta_cost/curr_temp) > random.random()/2:
                        print('Uphill Move Accepted')
                        current_placement_cost = transfer_cost
                        
                    else:
                        print('Move Rejected')
                        
                        transfer_gate1.placement_left_vertices = [x_vertices1, y_vertices1]
                        transfer_gate2.placement_left_vertices = [x_vertices2, y_vertices2]
                    
                    print('Best Placement Cost:', best_placement_cost)
                    print('Transfer Cost:', transfer_cost)
                    print('Delta Cost', delta_cost)


                iteration = iteration + 1
                
            
            curr_temp = r * curr_temp
            if curr_temp < Tf:
                frozen = True
        
        print('\n\n\nBest Placement')

        for i in range(len(netlist_gates)):
            netlist_gates[i].placement_left_vertices = best_placement_vertices[i]
            netlist_gates[i].placement_orientation = best_placement_orientation[i]
                
        self.plotting_placement(netlist_gates, io_pads, target_width, target_height)
        best_placement_area, best_placement_overlap_area, best_placement_wiring_cost = self.placement_cost(netlist_gates, io_pads, target_width, target_height)
        best_placement_cost = best_placement_wiring_cost + lambda_overlap_cost * best_placement_overlap_area
        
        print('Placement Area: ', best_placement_area)
        print('Placement Density: ', best_placement_area/(target_width * target_height))
        print('Overlap Area: ', best_placement_overlap_area)
        print('Cost: ', best_placement_cost)
        print('No of Gates:', len(netlist_gates))
        
        self.routing_core = self.generate_routing_core(netlist_gates, io_pads, target_width, target_height, routing_per_cell)    
    
    def placement_cost(self, netlist_gates, io_pads, target_width, target_height):
        total_area = 0
        total_overlap_area = 0
        total_wiring_length = 0
        total_wiring_cost = 0
        
        placement_core = numpy.zeros((target_height, target_width))
        
        for i in range(len(netlist_gates)):
            x_vertex = netlist_gates[i].placement_left_vertices[0]
            y_vertex = netlist_gates[i].placement_left_vertices[1]
            
            if netlist_gates[i].placement_orientation == 0:
                x_offset = netlist_gates[i].placement_dimensions[0]
                y_offset = netlist_gates[i].placement_dimensions[1]
            else:
                x_offset = netlist_gates[i].placement_dimensions[1]
                y_offset = netlist_gates[i].placement_dimensions[0]
            
            for j in range(y_vertex, y_vertex + y_offset):
                for i in range(x_vertex, x_vertex + x_offset):
                    placement_core[j, i] = placement_core[j, i] + 1
        
        total_area = numpy.sum(placement_core)
        for j in range(target_height):
            for i in range(target_width):
                if placement_core[j][i] == 0:
                    pass
                else:
                    total_overlap_area = total_overlap_area + placement_core[j][i] - 1
        
        for curr_gate in netlist_gates:
            for i in range(len(curr_gate.adjacent)):
                x_distance = abs(curr_gate.placement_left_vertices[0] + curr_gate.placement_dimensions[0]/2 - (curr_gate.adjacent[i].placement_left_vertices[0] + curr_gate.adjacent[i].placement_dimensions[0]/2))
                y_distance = abs(curr_gate.placement_left_vertices[1] + curr_gate.placement_dimensions[1]/2 - (curr_gate.adjacent[i].placement_left_vertices[1] + curr_gate.adjacent[i].placement_dimensions[1]/2))
                total_wiring_length = total_wiring_length + x_distance + y_distance
                total_wiring_cost = total_wiring_cost + curr_gate.adjacent_distance[i] * (x_distance + y_distance)
               
        return total_area, total_overlap_area, total_wiring_cost
    
    def plotting_placement(self, netlist_gates, io_pads, target_width, target_height):
        best_image_width = 1080
        best_image_height = 720
        
        width_factor = int(best_image_width/target_width)
        height_factor = int(best_image_height/target_height)
        
        w, h = (target_width + 5)*width_factor , target_height*height_factor
        img = Image.new("RGB", (w, h), color = (255, 255, 255))
        img1 = ImageDraw.Draw(img)

        shape = [(2*width_factor, 0), ((target_width+2)*width_factor, (target_height)*height_factor)]
        img1.rectangle(shape, fill ="#ffffff", outline ="black")
        for x in netlist_gates:
            if x.placement_orientation == 1:
                x_offset = x.placement_dimensions[1]
                y_offset = x.placement_dimensions[0]
            else:
                x_offset = x.placement_dimensions[0]
                y_offset = x.placement_dimensions[1]

            shape = [((x.placement_left_vertices[0] + 2)*width_factor, x.placement_left_vertices[1]*height_factor), ((x.placement_left_vertices[0] + x_offset + 2)*width_factor, (x.placement_left_vertices[1] + y_offset)*height_factor)]
            img1.rectangle(shape, fill ="#ffff33", outline ="black")
        
        for x in io_pads:
            if x.node_type == 'INPUT':
                shape = [(1*width_factor, x.placement_left_vertices[1]*height_factor), (2*width_factor, (x.placement_left_vertices[1]+1)*height_factor)]
            if x.node_type == 'OUTPUT':
                shape = [((x.placement_left_vertices[0]+3)*width_factor, x.placement_left_vertices[1]*height_factor), ((x.placement_left_vertices[0] + 4)*width_factor, (x.placement_left_vertices[1] + 1)*height_factor)]
            img1.rectangle(shape, fill ="#0000ff", width = 2)
 
        img.show()
        
        
    def rearrange_io_pads(self, io_pads, target_width, target_height):
        no_inputs = 0
        no_outputs = 0
        inputs = []
        outputs = []
        
        for x in io_pads:
            if x.node_type == 'INPUT':
                no_inputs = no_inputs + 1
                inputs.append(x)
            elif x.node_type == 'OUTPUT':
                no_outputs = no_outputs + 1
                outputs.append(x)
        
        input_pad_spacing = int(target_height/(no_inputs + 1))
        output_pad_spacing = int(target_height/(no_outputs + 1))
        
        input_place = input_pad_spacing
        
        for x in inputs:
            x.placement_left_vertices = [0, input_place]
            input_place = input_place + input_pad_spacing
        
        output_place = output_pad_spacing
        
        for x in outputs:
            x.placement_left_vertices = [target_width-1, output_place]
            output_place = output_place + output_pad_spacing
        
        
    def rearrange_netlist_gates(self, netlist_gates, target_width, target_height):
        for curr_gate in netlist_gates:
            x_vertex = curr_gate.placement_left_vertices[0]
            y_vertex = curr_gate.placement_left_vertices[1]
            
            if curr_gate.placement_orientation == 0:
                x_offset = curr_gate.placement_dimensions[0]
                y_offset = curr_gate.placement_dimensions[1]
            else:
                x_offset = curr_gate.placement_dimensions[1]
                y_offset = curr_gate.placement_dimensions[0]
            
            if x_vertex + x_offset >= target_width:
                x_vertex = target_width - x_offset - 1
            
            if y_vertex + y_offset >= target_height:
                y_vertex = target_height - y_offset
                
            curr_gate.placement_left_vertices = [x_vertex, y_vertex]
                       
            
    def generate_routing_core(self, netlist_gates, io_pads, target_width, target_height, routing_per_cell = 4):
        routing_core_width = target_width * routing_per_cell + 1
        routing_core_height = target_height * routing_per_cell + 1
        
        routing_core1 = []
        for j in range(routing_core_height):
            routing_core_row = []
            for i in range(routing_core_width):
                new_node = routing_node([i,j,0])
                routing_core_row.append(new_node)
            routing_core1.append(routing_core_row)
        
        routing_core2 = []
        for j in range(routing_core_height):
            routing_core_row = []
            for i in range(routing_core_width):
                new_node = routing_node([i,j,1])
                routing_core_row.append(new_node)
            routing_core2.append(routing_core_row)

        routing_layers = [routing_core1, routing_core2]
        
        for layer in range(2):
            for j in range(routing_core_height):
                for i in range(routing_core_width):
                    curr_node = routing_layers[layer][j][i]
                
                    if layer == 0:
                        curr_node.adjacent.append(routing_layers[1][j][i])
                        if i < routing_core_width - 1:
                            curr_node.adjacent.append(routing_layers[0][j][i+1])
                        if i>0:
                            curr_node.adjacent.append(routing_layers[0][j][i-1])
                    
                    else:
                        curr_node.adjacent.append(routing_layers[0][j][i])
                        if j < routing_core_height - 1:
                            curr_node.adjacent.append(routing_layers[1][j+1][i])
                        if j>0:
                            curr_node.adjacent.append(routing_layers[1][j-1][i])
        
        for x in netlist_gates:
            no_inputs = 0
            no_outputs = 0
            if x.node_type == 'DFF':
                no_inputs = 1
                no_outputs = 2
            elif x.node_type == 'NOT':
                no_inputs = 1
                no_outputs = 1
            else:
                no_inputs = int(x.node_type.strip()[-1])
                no_outputs = 1
            
            max_dimension = 0
            max_edge = x.placement_dimensions[1]
            if x.placement_dimensions[0] > x.placement_dimensions[1]:
                max_dimension = 1
                max_edge = x.placement_dimensions[0]
                
            
            input_distances = int(routing_per_cell*max_edge/(no_inputs+1))
            init_input_offset = input_distances-1
            input_position_array = []
            
            output_distances = int(routing_per_cell*max_edge/(no_outputs + 1))
            init_output_offset = output_distances - 1
            output_position_array = []
            
            if x.placement_orientation == 1:
                x_offset = int(routing_per_cell/2)*x.placement_dimensions[1]
                y_offset = int(routing_per_cell/2)*x.placement_dimensions[0]
                
                if max_dimension == 0:
                    for m in range(no_inputs):
                        input_position_array.append([x.placement_left_vertices[0]*routing_per_cell + init_input_offset, x.placement_left_vertices[1]*routing_per_cell, 0])
                        init_input_offset = init_input_offset + input_distances
                    
                    for m in range(no_outputs):
                        output_position_array.append([x.placement_left_vertices[0]*routing_per_cell + init_output_offset, (x.placement_left_vertices[1] + x.placement_dimensions[1])*routing_per_cell, 0])
                        init_output_offset = init_output_offset + output_distances
                
                else:
                    for m in range(no_inputs):
                        input_position_array.append([x.placement_left_vertices[0]*routing_per_cell, x.placement_left_vertices[1]*routing_per_cell + init_input_offset, 0])
                        init_input_offset = init_input_offset + input_distances
                    
                    for m in range(no_outputs):
                        output_position_array.append([(x.placement_left_vertices[0]+x.placement_dimensions[1])*routing_per_cell, (x.placement_left_vertices[1])*routing_per_cell + init_output_offset, 0])
                        init_output_offset = init_output_offset + output_distances
                    
            else:
                x_offset = int(routing_per_cell/2)*x.placement_dimensions[0]
                y_offset = int(routing_per_cell/2)*x.placement_dimensions[1]
                if max_dimension == 0:
                    for m in range(no_inputs):
                        input_position_array.append([x.placement_left_vertices[0]*routing_per_cell, x.placement_left_vertices[1]*routing_per_cell + init_input_offset, 0])
                        init_input_offset = init_input_offset + input_distances
                    
                    for m in range(no_outputs):
                        output_position_array.append([(x.placement_left_vertices[0] + x.placement_dimensions[0])*routing_per_cell, (x.placement_left_vertices[1])*routing_per_cell + init_output_offset, 0])
                        init_output_offset = init_output_offset + output_distances
                
                else:
                    for m in range(no_inputs):
                        input_position_array.append([x.placement_left_vertices[0]*routing_per_cell + init_input_offset, x.placement_left_vertices[1]*routing_per_cell, 0])
                        init_input_offset = init_input_offset + input_distances
                    
                    for m in range(no_outputs):
                        output_position_array.append([x.placement_left_vertices[0]*routing_per_cell + init_output_offset, (x.placement_left_vertices[1] + x.placement_dimensions[1])*routing_per_cell, 0])
                        init_output_offset = init_output_offset + output_distances

            x.routing_vertices = [routing_per_cell*x.placement_left_vertices[0] + x_offset, routing_per_cell*x.placement_left_vertices[1] + y_offset, 0]
            x.routing_input_positions = input_position_array
            x.routing_output_positions = output_position_array

        for x in io_pads:            
            x.routing_vertices = [routing_per_cell*x.placement_left_vertices[0], routing_per_cell*x.placement_left_vertices[1], 0]

        return routing_layers
    
    
# Maze Routing Process

    def maze_routing(self, routing_per_cell = 4):
        total_routing_length = 0
        max_routing_length = 0
        self.all_routing_lines = []
        self.all_routing_positions = []

        print('Height', len(self.routing_core[0]), 'Width', len(self.routing_core[0][0]))
        self.generate_routing_points(routing_per_cell)
       
        for routing_wire in self.all_routing_positions:
            for layer in self.routing_core:
                for y in layer:
                    for x in y:
                        x.cost = 65535
                        x.routing_blocked = False
                        x.routing_prev_dir = 'X'
                        x.source_node = False

            for x in self.all_routing_positions:
                for y in x:
                    self.routing_core[0][y[1]][y[0]].routing_blocked == True
                    self.routing_core[1][y[1]][y[0]].routing_blocked == True
                
            source_vertices  = routing_wire[0]
            print('\n\n\nSource', source_vertices)
            
            target_vertices = routing_wire[1:]
            for target_vertex in target_vertices:
                self.routing_core[0][target_vertex[1]][target_vertex[0]].routing_blocked = False
                self.routing_core[1][target_vertex[1]][target_vertex[0]].routing_blocked = False
                self.routing_core[0][target_vertex[1]][target_vertex[0]].cost = 65535
                self.routing_core[1][target_vertex[1]][target_vertex[0]].cost = 65535
                
            print('Target', target_vertices)
            
            wavefront = []
            wavefront.append(self.routing_core[source_vertices[2]][source_vertices[1]][source_vertices[0]])
            self.routing_core[source_vertices[2]][source_vertices[1]][source_vertices[0]].cost = 0
            self.routing_core[1][source_vertices[1]][source_vertices[0]].routing_blocked = False
            self.routing_core[0][source_vertices[1]][source_vertices[0]].routing_blocked = False
            self.routing_core[source_vertices[2]][source_vertices[1]][source_vertices[0]].routing_prev_dir = 'X'
            
            self.routing_expand(source_vertices, wavefront, target_vertices)
            
            self.routing_clear()
            
        self.plot_routing(routing_per_cell)
        
        
    def routing_expand(self, source_vertices, wavefront, target_vertices, bend_penalty = 2):
        
        target_reached = False
        iteration = 0
        
        while not target_reached:

            if wavefront != []:
                                
                block = wavefront.pop(0)

                for adjacent_block in block.adjacent:
                    if adjacent_block.routing_blocked == True:
                        pass
                    else:        
                        init_position = block.position
                        next_position = adjacent_block.position
                        init_dir = block.routing_prev_dir
                        current_dir = ''
                        if init_position[0] > next_position[0]:
                            current_dir = 'E'
                        elif init_position[0] < next_position[0]:
                            current_dir = 'W'
                        elif init_position[1] < next_position[1]:
                            current_dir = 'S'
                        elif init_position[1] > next_position[1]:
                            current_dir = 'N'
                        elif init_position[2] > next_position[2]:
                            current_dir = 'U'
                        elif init_position[2] < next_position[2]:
                            current_dir = 'D'
                                
                            
                        bend_cost = 0
                        if init_dir == current_dir:
                            bend_cost = 0
                        else:
                            bend_cost = bend_penalty
                            
                        if adjacent_block.cost > block.cost + 1 + bend_cost:
                            adjacent_block.cost = block.cost + 1 + bend_cost
                            adjacent_block.routing_prev_dir = current_dir
                            if adjacent_block not in wavefront:
                                wavefront.append(adjacent_block)
                            
                    for pos in range(len(target_vertices)):
                        if adjacent_block.position == target_vertices[pos]:
                            current_target = target_vertices.pop(pos)
                            print('Target Reached', source_vertices, current_target)
                            wavefront = self.expand_source_blocks(source_vertices, wavefront, current_target)
                            iteration = 0
                            break
                         
            #print('\nWavefront')
            #for x in wavefront:
            #    print(x.position, end='')
            #print()
            
            if target_vertices == []:
                target_reached = True
                
            iteration = iteration + 1
            if iteration > 2*len(self.routing_core[0][0])*len(self.routing_core[0]):
                target_reached = True
                print('Target Not Reached')
    
    def expand_source_blocks(self, source_vertices, wavefront, target_vertices):
        
        target_block = self.routing_core[target_vertices[2]][target_vertices[1]][target_vertices[0]]
        routing_cost = target_block.cost
        
        wavefront = []
        wavefront.append(self.routing_core[source_vertices[2]][source_vertices[1]][source_vertices[0]])
        
        wavepoints = []
                
        while (target_block.position[0] != source_vertices[0] or target_block.position[1] != source_vertices[1] or target_block.position[2] != source_vertices[2]):
            
            wavefront.append(target_block)
            target_block.routing_blocked = True
            target_block.cost = 0
            target_block.source_node = True
            wavepoints.append(target_block.position)
            
            if target_block.routing_prev_dir == 'E':
                target_block = self.routing_core[target_block.position[2]][target_block.position[1]][target_block.position[0] + 1]
            elif target_block.routing_prev_dir == 'W':
                target_block = self.routing_core[target_block.position[2]][target_block.position[1]][target_block.position[0] - 1]
            elif target_block.routing_prev_dir == 'N':
                target_block = self.routing_core[target_block.position[2]][target_block.position[1] + 1][target_block.position[0]]
            elif target_block.routing_prev_dir == 'S':
                target_block = self.routing_core[target_block.position[2]][target_block.position[1] - 1][target_block.position[0]]
            elif target_block.routing_prev_dir == 'U':
                target_block = self.routing_core[target_block.position[2]+1][target_block.position[1]][target_block.position[0]]
            elif target_block.routing_prev_dir == 'D':
                target_block = self.routing_core[target_block.position[2]-1][target_block.position[1]][target_block.position[0]]
            
           
        wavepoints.append(target_block.position)
        
        self.routing_core[source_vertices[2]][source_vertices[1]][source_vertices[0]].routing_blocked = True
        
        self.all_routing_lines.append(wavepoints)
        print('Routing Cost: ', routing_cost)

        return wavefront
    
    def routing_clear(self):
        for layer in self.routing_core:
            for y in layer:
                for x in y:
                    if x.routing_blocked == False:
                        x.cost = 65535
                        x.routing_prev_dir = 'X'
                    
                    
    def generate_routing_points(self, routing_per_cell):
        self.all_routing_positions = []
        for curr_gate in self.all_nodes:
            source_target_list = []                
            if curr_gate.node_type == 'INPUT':
                source_target_list.append(curr_gate.routing_vertices)
            else:
                source_target_list.append(curr_gate.routing_output_positions[curr_gate.routing_output_position_number])
                curr_gate.routing_output_position_number = curr_gate.routing_output_position_number + 1
                if curr_gate.routing_output_position_number == len(curr_gate.routing_output_positions):
                    curr_gate.routing_output_position_number = 0
            
            if curr_gate.output == True:
                source_target_list.append([len(self.routing_core[0][0])-1, curr_gate.routing_vertices[1], 0])

            for adjacent_gate in curr_gate.output_nodes:
                source_target_list.append(adjacent_gate.routing_input_positions[adjacent_gate.routing_input_position_number])
                adjacent_gate.routing_input_position_number = adjacent_gate.routing_input_position_number + 1
                if adjacent_gate.routing_input_position_number == len(adjacent_gate.routing_input_positions):
                    adjacent_gate.routing_input_position_number = 0
            
            self.all_routing_positions.append(source_target_list)                
        
        
    def plot_routing(self, routing_per_cell=4):
        best_image_width = 1080
        best_image_height = 720
        
        width_factor = int(best_image_width/len(self.routing_core[0][0]))
        height_factor = int(best_image_height/len(self.routing_core[0]))
        
        w, h = len(self.routing_core[0][0])*width_factor , len(self.routing_core[0])*height_factor
        img = Image.new("RGB", (w, h), color = (255, 255, 255))
        img1 = ImageDraw.Draw(img)

        shape = [(0, 0), ((len(self.routing_core[0][0])-1)*width_factor, (len(self.routing_core[0])-1)*height_factor)]
        img1.rectangle(shape, fill ="#ffffff", outline ="black")
        
        for x in self.all_gates:
            if x.placement_orientation == 0:
                x_offset = x.placement_dimensions[0]*int(routing_per_cell/2)
                y_offset = x.placement_dimensions[1]*int(routing_per_cell/2)
            else:
                x_offset = x.placement_dimensions[1]*int(routing_per_cell/2)
                y_offset = x.placement_dimensions[0]*int(routing_per_cell/2)
                
            shape = [((x.routing_vertices[0] - x_offset)*width_factor, (x.routing_vertices[1]-y_offset)*height_factor), ((x.routing_vertices[0]+x_offset)*width_factor, (x.routing_vertices[1]+y_offset)*height_factor)]
            img1.rectangle(shape, fill ="#ffff33", outline ="black")
                
        for routing_line in self.all_routing_lines:
            shape = []
            for point in routing_line:
                if point[2] == 0:
                    x_point = point[0]*width_factor
                    y_point = point[1]*height_factor
                    shape.append((x_point, y_point))
                else:
                    if shape != []:
                        img1.line(shape, fill ="#0000ff")
                    shape = []
        img.show()

        w, h = len(self.routing_core[0][0])*width_factor , len(self.routing_core[0])*height_factor
        img = Image.new("RGB", (w, h), color = (255, 255, 255))
        img1 = ImageDraw.Draw(img)

        shape = [(0, 0), ((len(self.routing_core[0][0])-1)*width_factor, (len(self.routing_core[0]))*height_factor)]
        img1.rectangle(shape, fill ="#ffffff", outline ="black")

        for x in self.all_gates:
            if x.placement_orientation == 0:
                x_offset = x.placement_dimensions[0]*int(routing_per_cell/2)
                y_offset = x.placement_dimensions[1]*int(routing_per_cell/2)
            else:
                x_offset = x.placement_dimensions[1]*int(routing_per_cell/2)
                y_offset = x.placement_dimensions[0]*int(routing_per_cell/2)
                
            shape = [((x.routing_vertices[0] - x_offset)*width_factor, (x.routing_vertices[1]-y_offset)*height_factor), ((x.routing_vertices[0]+x_offset)*width_factor, (x.routing_vertices[1]+y_offset)*height_factor)]
            img1.rectangle(shape, fill ="#ffff33", outline ="black")
                
        for routing_line in self.all_routing_lines:
            shape = []
            for point in routing_line:
                if point[2] == 1:
                    x_point = point[0]*width_factor
                    y_point = point[1]*height_factor
                    shape.append((x_point, y_point))
                else:
                    if shape!= []:
                        img1.line(shape, fill ="#ff0000")
                    shape = []
            
        img.show()
