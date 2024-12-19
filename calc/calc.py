import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

# Function to create a linkage tree diagram using NetworkX

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 (SimHei)
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# Initialize data for plots
individuals = ['个体1', '个体2', '个体3', '个体4', '个体5']
fitness_values = [0.75, 0.60, 0.50, 0.85, 0.70]

fronts = {
    'Front 1': [(1, 4), (2, 3)],
    'Front 2': [(3, 2), (4, 1)],
    'Front 3': [(5, 0)]
}

selected_individuals = ['父代1', '父代2', '供体1', '供体2']
selection_sizes = [30, 30, 20, 20]

genes = ['S1', 'S2', 'S3', 'S4']
parent1 = [1, 1, 0, 0]
parent2 = [1, 0, 1, 1]
donor1 = [0, 1, 1, 0]
donor2 = [1, 0, 0, 1]
offspring = [1, 1, 1, 0]

mutation_before = [1, 1, 1, 0]
mutation_after = [1, 1, 0, 1]

generations = ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5']
population = [5, 6, 7, 8, 9]

environment_labels = ['Selected', 'Not Selected']
environment_sizes = [80, 20]

# Plot Fitness Evaluation
plt.figure(figsize=(10, 6))
plt.bar(individuals, fitness_values, color='skyblue')
plt.xlabel('Individuals')
plt.ylabel('Fitness Value')
plt.title('Fitness Evaluation')
plt.ylim(0, 1)
plt.show()

# Plot Non-dominated Sorting
colors = ['yellow', 'red', 'green', 'blue']
plt.figure(figsize=(10, 6))
for idx, (front, points) in enumerate(fronts.items()):
    x, y = zip(*points)
    plt.scatter(x, y, color=colors[idx], label=front)
plt.xlabel('Objective 1')
plt.ylabel('Objective 2')
plt.title('Non-dominated Sorting')
plt.legend()
plt.show()

# Plot Selection Process
plt.figure(figsize=(8, 8))
plt.pie(selection_sizes, labels=selected_individuals,
        colors=colors[:4], autopct='%1.1f%%', startangle=140)
plt.title('Selection Process')
plt.show()

# Plot Linkage-based Recombination
ind = np.arange(len(genes))
width = 0.15
plt.figure(figsize=(12, 6))
plt.bar(ind - 2*width, parent1, width, label='Parent 1', color='gold')
plt.bar(ind - width, parent2, width, label='Parent 2', color='yellowgreen')
plt.bar(ind, donor1, width, label='Donor 1', color='lightcoral')
plt.bar(ind + width, donor2, width, label='Donor 2', color='lightskyblue')
plt.bar(ind + 2*width, offspring, width, label='Offspring', color='purple')
plt.xlabel('Genes')
plt.ylabel('Presence (1) / Absence (0)')
plt.title('Linkage-based Recombination')
plt.xticks(ind, genes)
plt.legend()
plt.show()

# Plot Mutation Process
plt.figure(figsize=(10, 6))
plt.bar(ind - width/2, mutation_before, width,
        label='Before Mutation', color='skyblue')
plt.bar(ind + width/2, mutation_after, width,
        label='After Mutation', color='lightcoral')
plt.xlabel('Genes')
plt.ylabel('Presence (1) / Absence (0)')
plt.title('Mutation Process')
plt.xticks(ind, genes)
plt.legend()
plt.show()

# Plot Population Update
plt.figure(figsize=(10, 6))
plt.bar(generations, population, color='lightgreen')
plt.xlabel('Generations')
plt.ylabel('Population Size')
plt.title('Population Update')
plt.show()

# Plot Environmental Selection
plt.figure(figsize=(8, 8))
plt.pie(environment_sizes, labels=environment_labels, colors=[
        'gold', 'lightcoral'], autopct='%1.1f%%', startangle=140)
plt.title('Environmental Selection')
plt.show()


def plot_separated_linkage_tree():
    G = nx.DiGraph()

    # Define nodes and edges to create a linkage tree example
    nodes = [
        ("root", {"label": "S1, S2, S3, S4"}),
        ("S1,S2", {"label": "S1, S2"}),
        ("S3,S4", {"label": "S3, S4"}),
        ("S1", {"label": "S1"}),
        ("S2", {"label": "S2"}),
        ("S3", {"label": "S3"}),
        ("S4", {"label": "S4"})
    ]

    edges = [
        ("root", "S1,S2"),
        ("root", "S3,S4"),
        ("S1,S2", "S1"),
        ("S1,S2", "S2"),
        ("S3,S4", "S3"),
        ("S3,S4", "S4")
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = {
        "root": (0, 1),
        "S1,S2": (-1, 0.5),
        "S3,S4": (1, 0.5),
        "S1": (-1.5, 0),
        "S2": (-0.5, 0),
        "S3": (0.5, 0),
        "S4": (1.5, 0)
    }

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=3000, node_color="lightblue",
            font_size=10, font_weight="bold", font_color="black", edge_color="gray", width=2, arrowsize=20)
    plt.title("Linkage Tree Model Example", fontsize=15)
    plt.show()


# Plotting the improved linkage tree model example
plot_separated_linkage_tree()


def plot_linkage_tree():
    G = nx.Graph()

    # Adding nodes and edges to the graph to create a linkage tree
    nodes = [
        ("root", {"label": "S1, S2, S3, S4"}),
        ("S1,S2", {"label": "S1, S2"}),
        ("S3,S4", {"label": "S3, S4"}),
        ("S1", {"label": "S1"}),
        ("S2", {"label": "S2"}),
        ("S3", {"label": "S3"}),
        ("S4", {"label": "S4"})
    ]

    edges = [
        ("root", "S1,S2"),
        ("root", "S3,S4"),
        ("S1,S2", "S1"),
        ("S1,S2", "S2"),
        ("S3,S4", "S3"),
        ("S3,S4", "S4")
    ]

    G.add_nodes_from(nodes)
    G.add_edges_from(edges)

    pos = nx.spring_layout(G, seed=42)

    plt.figure(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, 'label'), node_size=3000,
            node_color="skyblue", font_size=10, font_weight="bold", font_color="black")
    plt.title("Linkage Tree Model", fontsize=15)
    plt.show()

# Function to plot fitness evaluation as a bar chart


def plot_fitness_evaluation():
    individuals = ['Ind1', 'Ind2', 'Ind3', 'Ind4', 'Ind5']
    fitness_values = [0.75, 0.60, 0.90, 0.85, 0.80]

    plt.figure(figsize=(10, 6))
    plt.bar(individuals, fitness_values, color='skyblue')
    plt.xlabel('Individuals')
    plt.ylabel('Fitness Value')
    plt.title('Fitness Evaluation')
    plt.ylim(0, 1)
    plt.show()

# Function to plot non-dominated sorting as a scatter plot


def plot_non_dominated_sorting():
    fronts = {
        'Front 1': [(1, 5), (2, 4)],
        'Front 2': [(3, 3), (4, 2)],
        'Front 3': [(5, 1)]
    }

    colors = ['red', 'green', 'blue']
    plt.figure(figsize=(10, 6))

    for idx, (front, points) in enumerate(fronts.items()):
        x, y = zip(*points)
        plt.scatter(x, y, color=colors[idx], label=front)

    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.title('Non-dominated Sorting')
    plt.legend()
    plt.show()

# Function to plot selection process as a pie chart


def plot_selection_process():
    labels = ['Parent 1', 'Parent 2', 'Donor 1', 'Donor 2']
    sizes = [30, 30, 20, 20]
    colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Selection Process')
    plt.show()

# Function to plot linkage-based recombination as a diagram


def plot_linkage_based_recombination():
    labels = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4', 'Gene 5']
    parent1 = [1, 1, 0, 0, 1]
    parent2 = [0, 1, 1, 0, 0]
    donor1 = [1, 0, 1, 1, 0]
    donor2 = [0, 0, 1, 1, 1]
    offspring = [1, 1, 1, 1, 0]

    ind = np.arange(len(labels))
    width = 0.15

    plt.figure(figsize=(12, 6))
    plt.bar(ind - 2*width, parent1, width, label='Parent 1', color='gold')
    plt.bar(ind - width, parent2, width, label='Parent 2', color='yellowgreen')
    plt.bar(ind, donor1, width, label='Donor 1', color='lightcoral')
    plt.bar(ind + width, donor2, width, label='Donor 2', color='lightskyblue')
    plt.bar(ind + 2*width, offspring, width, label='Offspring', color='purple')

    plt.xlabel('Genes')
    plt.ylabel('Presence (1) / Absence (0)')
    plt.title('Linkage-based Recombination')
    plt.xticks(ind, labels)
    plt.legend()
    plt.show()

# Function to plot mutation process as a before and after comparison


def plot_mutation_process():
    labels = ['Gene 1', 'Gene 2', 'Gene 3', 'Gene 4', 'Gene 5']
    before_mutation = [1, 0, 1, 0, 1]
    after_mutation = [1, 1, 1, 0, 0]

    ind = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(ind - width/2, before_mutation, width,
            label='Before Mutation', color='skyblue')
    plt.bar(ind + width/2, after_mutation, width,
            label='After Mutation', color='lightcoral')

    plt.xlabel('Genes')
    plt.ylabel('Presence (1) / Absence (0)')
    plt.title('Mutation Process')
    plt.xticks(ind, labels)
    plt.legend()
    plt.show()

# Function to plot population update as a bar chart


def plot_population_update():
    generations = ['Gen 1', 'Gen 2', 'Gen 3', 'Gen 4', 'Gen 5']
    population = [50, 55, 60, 65, 70]

    plt.figure(figsize=(10, 6))
    plt.bar(generations, population, color='lightgreen')
    plt.xlabel('Generations')
    plt.ylabel('Population Size')
    plt.title('Population Update')
    plt.show()

# Function to plot environmental selection as a pie chart


def plot_environmental_selection():
    labels = ['Selected', 'Not Selected']
    sizes = [80, 20]
    colors = ['gold', 'lightcoral']

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, colors=colors,
            autopct='%1.1f%%', startangle=140)
    plt.title('Environmental Selection')
    plt.show()


# Plotting all required diagrams
plot_linkage_tree()
plot_fitness_evaluation()
plot_non_dominated_sorting()
plot_selection_process()
plot_linkage_based_recombination()
plot_mutation_process()
plot_population_update()
plot_environmental_selection()
