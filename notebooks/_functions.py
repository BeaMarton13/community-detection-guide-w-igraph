#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Install matplotlib (usually comes with Anaconda/Jupyter, but good to ensure)
# !pip install matplotlib --quiet

# Install numpy (usually comes with Anaconda/Jupyter, but good to ensure)
# !pip install numpy --quiet

# Install igraph
# !pip install igraph --quiet

# Install itables for community detection table
# !pip install itables --quiet


# In[8]:


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import numpy as np
import os 
import igraph as ig
import time
import multiprocessing
import random
import itertools
import pandas as pd
import io
import requests
from itables import init_notebook_mode, show
import pickle


# In[3]:


def load_graph_from_gml_file(graph_file: str, weight_attribute_name: str = "weight"):
    # Check if the graph file exists
    if not os.path.exists(graph_file):
        print(f"Error: Graph file '{graph_file}' not found.")
        print(f"Please ensure '{graph_file}' is a valid path to your 'lesmis.gml' file.")
        print("You can typically find this file by searching for 'lesmis.gml network dataset'.")
        return # Exit the function if file is not found

    try:
        # Load network from GML file
        # igraph.Graph.Read_GML will automatically load edge attributes like 'value'
        # if they are present in the GML file.
        graph = ig.Graph.Read_GML(graph_file)
        
        # Check if the graph has the correct weight attribute name
        if weight_attribute_name not in graph.edge_attributes():
            print(f"Warning: Graph '{graph_file}' does not have a '{weight_attribute_name}' attribute. "
                  "Community detection will proceed without explicit weights, or if the algorithm "
                  "expects them, it might use default uniform weights.")
            # If no 'value' attribute, assign a default uniform weight for visualization purposes
            graph.es[weight_attribute_name] = 1 

        return graph


    except Exception as e:
        print(f"An error occurred while loading or processing the graph: {e}")
        return

def community_detection(graph: ig.Graph, community_detection_method: str = "multilevel", weight_attribute_name: str = "weight", 
                        params: dict = None):
    if community_detection_method == "multilevel":
        return graph.community_multilevel(weights=weight_attribute_name if weight_attribute_name in graph.edge_attributes() else None)
    elif community_detection_method == "leiden":
        if params is None:
            return graph.community_leiden(weights=weight_attribute_name if weight_attribute_name in graph.edge_attributes() else None)
        else:
            params["weights"] = weight_attribute_name if weight_attribute_name in graph.edge_attributes() else None
            return graph.community_leiden(**params)
    elif community_detection_method == "fastgreedy":
        return graph.community_fastgreedy(weights=weight_attribute_name if weight_attribute_name in graph.edge_attributes() else None).as_clustering()




# # Functions useful to test community structure

# In[4]:


def get_modularity_on_clustering(graph: ig.Graph, community_detection_method: str = "multilevel", params: dict = None):
    partition = community_detection(graph, community_detection_method, weight_attribute_name=None, params=params)
    return partition.modularity

def rewire(graph: ig.Graph, community_detection_method: str = "multilevel", params: dict = None):
    num_randomizations = 500  # Number of randomized networks to generate
    modularity_random_networks = []
    
    num_swaps_for_randomization = graph.ecount() * 10
    
    for i in range(num_randomizations):
        # G.rewire() modifies the graph in-place, so we must work on a copy.
        graph_random = graph.copy()
    
        graph_random.rewire(n=num_swaps_for_randomization)
    
        modularity_random_networks.append(get_modularity_on_clustering(graph_random))

    return modularity_random_networks

def plot_histogram(modularity_original: float, modularity_random_networks: list[float], graph_name: str="Karate Club Network"):
    plt.figure(figsize=(10, 6))
    plt.hist(modularity_random_networks, bins=30, alpha=0.7, color='lightgreen',
             edgecolor='black', label='Modularity of Randomized Networks')
    
    # Set x-axis limits from 0 to 1
    plt.xlim(0, 1)
    
    # Plot a vertical line for the original network's modularity
    plt.axvline(modularity_original, color='red', linestyle='dashed', linewidth=2,
                label=f'Original Network Modularity ({modularity_original:.4f})')
    
    plt.title(f'Modularity of Original vs. Randomized {graph_name} (igraph)')
    plt.xlabel('Modularity Score')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.show()


def test_community_structure(graph: ig.Graph, graph_name: str = "Karate Club Network", community_detection_method: str = "multilevel", 
                             params: dict = None):
    modularity_orig = get_modularity_on_clustering(graph, community_detection_method, params)
    modularity_random_networks = rewire(graph, community_detection_method, params)
    plot_histogram(modularity_orig, modularity_random_networks, graph_name)


# In[5]:


def check_key_existence(keys, params):
    return all(key in params for key in keys)
    

def build_community_detection_method(graph, community_detection_method, params):
    community_detection = None
    if community_detection_method == "leiden":
        if params is None:
            raise ValueError("params must not be None")
        if not check_key_existence(["resolution"], params):
            raise KeyError("Key not found in params")
        resolution_list = params["resolution"]
        leiden_other_params = {k: v for k, v in params.items() if k != "resolution"}
        community_detection = lambda seed_idx: graph.community_leiden(resolution=resolution_list[seed_idx], **leiden_other_params)
    elif community_detection_method == "multilevel":
        community_detection = lambda _: graph.community_multilevel(**(params if params else {})) # Pass original params if any, or empty dict
    else:
        raise ValueError(f"Unknown community detection method: {community_detection_method}")

    if community_detection is None:
        raise RuntimeError("Failed to set up community_detector_executor.")

    return community_detection


# A helper function to run a target function with a timeout using multiprocessing
def _run_with_timeout(func, args=(), kwargs={}, timeout_seconds=60):
    """
    Runs a function in a separate process with a timeout.
    Returns (result, True) if successful, (None, False) if timeout occurs.
    """
    # Use a multiprocessing.Queue to get the result from the child process
    q = multiprocessing.Queue()
    
    def target():
        try:
            res = func(*args, **kwargs)
            q.put((res, None)) # Put result and no exception
        except Exception as e:
            q.put((None, e)) # Put no result and the exception

    process = multiprocessing.Process(target=target)
    process.start()
    process.join(timeout=timeout_seconds)

    if process.is_alive():
        # If the process is still alive, it means it timed out
        print(f"Warning: Function '{func.__name__}' timed out after {timeout_seconds} seconds. Terminating process.")
        process.terminate() # Forcefully terminate the process
        process.join() # Wait for termination
        time.sleep(0.01) # Small delay to allow OS cleanup after termination attempt
        return None, False, None # Return None result, False for success, None for exception
    else:
        # Process finished, check for result or exception
        if not q.empty():
            res, exception = q.get()
            if exception:
                raise exception # Re-raise any exception caught in the process
            return res, True, None # Return result, True for success
        else:
            # This case might happen if process terminates unexpectedly without putting anything
            print(f"Warning: Process for '{func.__name__}' finished but no result was put in queue.")
            return None, False, None


def generate_reference_partition(graph: ig.Graph, optimal_timeout_seconds: int, use_optimal_as_reference: bool = True):
    reference_partition = None
    if use_optimal_as_reference:
        ref_partition_result, success, exception = _run_with_timeout(
            graph.community_optimal_modularity,
            timeout_seconds=optimal_timeout_seconds
        )
        if success and ref_partition_result is not None:
            reference_partition = ref_partition_result
            print(f"Optimal partition found with modularity: {reference_partition.modularity:.4f}")
        else:
            if exception:
                print(f"Optimal partition calculation failed with error: {exception}")
            print("Falling back to a fixed-seed Louvain partition as reference.")
            # Fallback for larger graphs or if optimal fails/times out
            random.seed(42) # Fix seed for a reproducible reference
            reference_partition = graph.community_multilevel()
            random.seed(None) # Unset seed for subsequent stochastic runs
            print(f"Reference Louvain partition (fixed seed) found with modularity: {reference_partition.modularity:.4f}")
    else:
        # Option B: Louvain with fixed seed as reference (for larger graphs)
        print("\nUsing a fixed-seed Louvain partition as reference.")
        random.seed(42) # Fix seed for a reproducible reference
        reference_partition = graph.community_multilevel()
        random.seed(None) # Unset seed for subsequent stochastic runs
        print(f"Reference Louvain partition (fixed seed) found with modularity: {reference_partition.modularity:.4f}")

    
    if reference_partition is None:
        raise("Could not establish a reference partition")

    return reference_partition

def run_stochastic_community_detection(graph, reference_partition: ig.clustering.VertexClustering, num_runs: int, 
                                       community_detection_method: str = "multilevel", return_partitions: bool = False,
                                      params: dict = None):
    nmi_values = []
    all_partitions = []
    print(f"\nRunning Louvain community detection {num_runs} times and calculating NMI...")

    community_detection = build_community_detection_method(graph, community_detection_method, params)

    for i in range(num_runs):
        random.seed()
        current_partition = community_detection(i)
        all_partitions.append(current_partition)

        if not return_partitions:
            # Calculate NMI between the current partition and the reference partition
            # 'method='nmi'' specifies Normalized Mutual Information
            nmi = ig.compare_communities(reference_partition, current_partition, method='nmi')
            nmi_values.append(nmi)

        if (i + 1) % (num_runs // 10 if num_runs >= 10 else 1) == 0:
            print(f"  Processed {i + 1}/{num_runs} runs.")

    if return_partitions:
        return all_partitions
        
    return nmi_values

def plot_nmi_histogram(graph: ig.Graph, nmi_values: list[float], title: str):
    plt.figure(figsize=(10, 6))
    plt.hist(nmi_values, bins=20, edgecolor='black', alpha=0.7, color='lightcoral')
    plt.title(title)
    plt.xlabel('Normalized Mutual Information (NMI) Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    # Set x-axis limits from 0 to 1
    plt.xlim(0, 1)

    # Add a line for the mean NMI
    mean_nmi = np.mean(nmi_values)
    plt.axvline(mean_nmi, color='blue', linestyle='dashed', linewidth=2,
                label=f'Mean NMI: {mean_nmi:.4f}')

    plt.legend()
    plt.tight_layout()
    plt.show()

def calculate_pairwise_nmi(partitions: list[ig.clustering.VertexClustering]):
    """
    Calculates Normalized Mutual Information (NMI) for all unique pairs of partitions.
    """
    pairwise_nmi_values = []
    for p1, p2 in itertools.combinations(partitions, 2):
        nmi = ig.compare_communities(p1, p2, method='nmi')
        pairwise_nmi_values.append(nmi)
    return pairwise_nmi_values


# In[6]:


def create_grid_graph(rows, cols, circular=False):
    """
    Generates a 2D grid graph (lattice) without plotting.
    Returns the graph and its column count for later layout.
    """
    G = ig.Graph.Lattice(dim=[rows, cols], nei=1, circular=circular)
    return G

# --- Final version of cluster_and_plot_leiden with grid_cols ---
def cluster_and_plot_leiden_on_grid(graph, grid_cols, title="Graph with Leiden Communities", plot_size=(8, 8)):
    """
    Clusters a graph using the Leiden algorithm and plots the result
    with vertices colored by their community, specifically for grid layouts.

    Args:
        graph (igraph.Graph): The graph to cluster.
        grid_cols (int): The number of columns the grid graph has. Crucial for layout.
        title (str): Title for the plot.
        plot_size (tuple): Size of the matplotlib figure (width, height).

    Returns:
        igraph.clustering.VertexClustering: The community detection result.
    """
    print(f"Clustering graph with {graph.vcount()} vertices and {graph.ecount()} edges using Leiden algorithm...")

    resolution = 0.15
    communities = graph.community_leiden(objective_function="modularity", resolution=resolution)

    # Assign colors based on community membership
    palette = ig.GradientPalette("red", "blue", n=len(communities)) 
    if len(communities) > 1:
        vertex_colors = [palette.get(membership_id) for membership_id in communities.membership]
    else:
        vertex_colors = ["red"]
    
    # Generate the grid layout using the provided grid_cols
    layout = graph.layout_grid(width=grid_cols)
            
    fig, ax = plt.subplots(figsize=plot_size)

    ig.plot(
        graph,
        target=ax,
        layout=layout,
        vertex_size=10,
        vertex_color=vertex_colors, # Use community colors
        vertex_label=None,
        edge_width=0.8,
        edge_color="gray",
        bbox=(0, 0, 600, 600),
        margin=20
    )
    ax.set_title(title)
    ax.axis('off')
    plt.show()

    return communities, resolution


# # Game of Thrones

# In[7]:


def load_got_network(url: str = None):
    if url is None:
        # URL of the raw CSV data for Season 1 edges
        url = "https://raw.githubusercontent.com/mathbeveridge/gameofthrones/master/data/got-s1-edges.csv"
    
    # Fetch the data from the URL
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)
        season1_edges_data = io.StringIO(response.text)
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from {url}: {e}")
        print("Please ensure you have an active internet connection or download the file manually.")
        exit()
        
    # Load the data into a pandas DataFrame
    df_s1_edges = pd.read_csv(season1_edges_data)
    
    # Create an igraph graph
    # Assuming 'Source' and 'Target' are the columns for the edges
    # And 'Weight' is the column for edge weights (if present)
    
    # Check if 'Weight' column exists
    if 'Weight' in df_s1_edges.columns:
        # Prepare data as list of (source, target, weight) tuples
        edges_for_tuplelist = df_s1_edges[['Source', 'Target', 'Weight']].values.tolist()
    
        g_s1 = ig.Graph.TupleList(edges_for_tuplelist,
                                 directed=False,
                                 weights=True) # This tells igraph the 3rd element in the tuple is the weight
        print("Graph created with edge weights using weights=True.")
    else:
        g_s1 = ig.Graph.TupleList(df_s1_edges[['Source', 'Target']].itertuples(index=False),
                                  directed=False) # Assuming interactions are undirected
        print("\nGraph created without edge weights.")

    return g_s1

    
def graph_summary(g: ig.Graph):
    print(f"\n--- Graph Summary for Season 1 ---")
    print(f"Number of vertices (characters): {g.vcount()}")
    print(f"Number of edges (interactions): {g.ecount()}")
    print(f"Is graph directed? {g.is_directed()}")
    print(f"Graph attributes: {g.attributes()}")
    print(f"Vertex attributes: {g.vs.attributes()}")
    print(f"Edge attributes: {g.es.attributes()}")


def export_graph(g: ig.Graph, filename: str):
    with open(filename, 'wb') as f:
        pickle.dump(g, f)


# # Community detection table

# In[9]:


def show_table():
    # This line enables itables to make all DataFrames interactive by default
    # If you only want specific tables to be interactive, remove this line and use show(df) explicitly
    init_notebook_mode(all_interactive=True)
    
    # Your DataFrame creation code (as before)
    data = {
        'Method': ['Edge Betweenness', 'Fast-Greedy', 'Fluid Communities', 'Infomap', 'Label Propagation', 'Leading Eigenvector', 'Leiden', 'Louvain (Multilevel)', 'Spinglass', 'Walktrap', 'Optimal Modularity', 'Voronoi'],
        'Function in igraph (Python)': ['`Graph.community_edge_betweenness()`', '`Graph.community_fastgreedy()`', '`Graph.community_fluid_communities()`', '`Graph.community_infomap()`', '`Graph.community_label_propagation()`', '`Graph.community_leading_eigenvector()`', '`Graph.community_leiden()`', '`Graph.community_multilevel()`', '`Graph.community_spinglass()`', '`Graph.community_walktrap()`', '`Graph.community_optimal_modularity()`', '`Graph.community_voronoi()`'],
        'Directed Graph Support': ['✅', '❌', '❌', '✅', '❌(?)', '❌', '❌', '✅', '❌', '❌', '❌', '✅'],
        'Weighted Graph Support': ['✅', '✅', '❌ (weights ignored)', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅'],
        'Signed Graph Support': ['❌', '❌', '❌', '❌', '❌', '❌', '❌', '❌', '✅', '❌', '❌', '❌'],
        'Sparse Graph Performance': ['✅', '✅ (Very efficient)', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '✅', '❌ (Small graphs only)', '✅'],
        'Dense Graph Performance': ['❌ (Slow for large)', '✅ (Can handle)', '✅', '✅', '✅', '✅', '✅', '✅', '❌ (Slower)', '✅', '❌ (Small graphs only)', '✅'],
        'Deterministic': ['✅', '❌', '❌', '❌', '❌', '✅', '❌', '❌', '❌', '✅', '✅', '✅'],
        'Notes': [
            'Divisive hierarchical method. Good for small to medium graphs. Returns a dendrogram. Modularity-based partition can be extracted. The underlying modularity is typically for undirected graphs.',
            'Agglomerative, modularity-maximization method. Returns a dendrogram. Efficient for large sparse graphs. Suffers from resolution limit. (Non-deterministic due to greedy choices / tie-breaking)',
            'Propagation-based. Requires `k` (number of communities) as input. Stochastic. Very fast and scalable. Primarily for unweighted, undirected graphs. (Non-deterministic due to random seeds/updates)',
            'Based on information theory (minimizing description length of random walks). Can handle directed and weighted graphs. (Non-deterministic due to random walks)',
            'Fast, propagation-based. (Non-deterministic due to random initialization/tie-breaking). Uses `directed` parameter to respect edge direction.',
            'Modularity-maximization (spectral method). Finds highest modularity partition. Can be slow for very large graphs. For undirected, weighted graphs.',
            'Improvement over Louvain. Guarantees well-connected communities. Usually higher modularity and more stable than Louvain. Can use resolution parameter. Highly recommended for general use. (Non-deterministic due to local moves/tie-breaking)',
            'Greedy, iterative modularity optimization. Fast and scalable. Returns hierarchical partition. Can suffer from resolution limit and potentially disconnected communities. (Non-deterministic due to local moves/tie-breaking)',
            'Based on statistical mechanics (Ising model and simulated annealing). Can handle negative weights (as "frustration"). Can be computationally intensive, especially for dense or large graphs. Allows for fixed number of spins (`k`). (Stochastic by design)',
            'Based on random walks. Merges communities that random walks tend to stay within. Hierarchical output (dendrogram). Efficient for sparse graphs.',
            'Finds the **exact** highest modularity partition. Computationally very expensive. Only practical for very small graphs (dozens to ~100 nodes).',
            'Partitions nodes into "cells" based on proximity (e.g., shortest path distance) to pre-defined "seed" nodes. Not a direct community *discovery* algorithm; it *assigns* based on input seeds. Used when node coordinates or influence regions are relevant.'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # To display the DataFrame with interactive features, including sticky headers AND frozen first column
    show(df, scrollY="300px", scrollCollapse=True, fixedColumns=True, pageLength=-1)


# In[ ]:




