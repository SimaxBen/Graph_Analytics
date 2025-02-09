import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import SpectralClustering
import community.community_louvain as community_louvain
from networkx.algorithms import community as nx_comm
from scipy.stats import gaussian_kde
import time
import sys
from collections import Counter

class CommunityDetector:
    def __init__(self, dataset_path):
        """Initialize the detector with dataset path."""
        self.G = nx.read_edgelist(dataset_path, nodetype=int)
        self.print_basic_stats()
        self.compute_additional_stats()
    
    def print_basic_stats(self):
        """Print basic network statistics."""
        print(f"Number of nodes: {self.G.number_of_nodes()}")
        print(f"Number of edges: {self.G.number_of_edges()}")
        print(f"Average clustering: {nx.average_clustering(self.G):.4f}")
    
    def compute_additional_stats(self):
        """Compute additional network statistics."""
        self.avg_degree = np.mean([d for n, d in self.G.degree()])
        self.density = nx.density(self.G)
        print(f"Average Degree: {self.avg_degree:.2f}")
        print(f"Graph Density: {self.density:.4f}")
    
    def analyze_degree_distribution(self):
        """Analyze the degree distribution in detail."""
        degrees = [d for n, d in self.G.degree()]
        degree_count = Counter(degrees)
        
        # Calculate statistics
        max_degree = max(degrees)
        min_degree = min(degrees)
        median_degree = np.median(degrees)
        std_degree = np.std(degrees)
        
        print("\nDegree Distribution Statistics:")
        print(f"Maximum Degree: {max_degree}")
        print(f"Minimum Degree: {min_degree}")
        print(f"Median Degree: {median_degree:.2f}")
        print(f"Standard Deviation: {std_degree:.2f}")
        
        return degrees, degree_count

    def plot_degree_distribution(self, save=True, display=True):
        """Plot enhanced degree distribution with statistics."""
        degrees, degree_count = self.analyze_degree_distribution()
        
        plt.figure(figsize=(10, 6))
        
        # Create histogram
        plt.hist(degrees, bins='auto', density=True, alpha=0.7, 
                color='skyblue', edgecolor='black')
        
        # Add a kernel density estimate
        density = gaussian_kde(degrees)
        xs = np.linspace(min(degrees), max(degrees), 200)
        plt.plot(xs, density(xs), 'r-', lw=2, label='KDE')
        
        plt.title("Node Degree Distribution")
        plt.xlabel("Degree")
        plt.ylabel("Density")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if save:
            plt.savefig('degree_distribution.png', dpi=300, bbox_inches='tight')
        if display:
            plt.show()
        else:
            plt.close()
    
    def compare_with_random(self):
        """Compare with random graph of same size and density."""
        random_graph = nx.erdos_renyi_graph(
            n=self.G.number_of_nodes(), 
            p=self.density, 
            seed=42
        )
        partition_random = community_louvain.best_partition(random_graph)
        communities_random = [{n for n in partition_random if partition_random[n] == c} 
                            for c in set(partition_random.values())]
        mod_random = nx_comm.modularity(random_graph, communities_random)
        print(f"Random Graph Modularity: {mod_random:.4f}")
        return mod_random
    
    def louvain_detection(self):
        """Implement Louvain community detection."""
        start_time = time.time()
        partition = community_louvain.best_partition(self.G)
        communities = [{n for n in partition if partition[n] == c} 
                      for c in set(partition.values())]
        modularity = nx_comm.modularity(self.G, communities)
        execution_time = time.time() - start_time
        return partition, modularity, execution_time
    
    def spectral_clustering(self, n_clusters=5):
        """Implement Spectral Clustering."""
        start_time = time.time()
        adj_matrix = nx.to_numpy_array(self.G)
        sc = SpectralClustering(n_clusters=n_clusters, 
                               affinity='precomputed', 
                               random_state=42)
        labels = sc.fit_predict(adj_matrix)
        communities = [{n for n, lbl in enumerate(labels) if lbl == c} 
                      for c in set(labels)]
        modularity = nx_comm.modularity(self.G, communities)
        execution_time = time.time() - start_time
        return labels, modularity, execution_time
    
    def label_propagation(self):
        """Implement Label Propagation."""
        start_time = time.time()
        communities = list(nx_comm.asyn_lpa_communities(self.G))
        modularity = nx_comm.modularity(self.G, communities)
        execution_time = time.time() - start_time
        return communities, modularity, execution_time
    
    def visualize_communities(self, partition, title):
        """Visualize detected communities."""
        plt.figure(figsize=(10, 8))
        colors = [partition[node] for node in self.G.nodes()]
        nx.draw_spring(self.G, node_color=colors, 
                      node_size=50, edge_color="gray", 
                      alpha=0.7)
        plt.title(title)
        plt.show()

def main():
    detector = CommunityDetector("facebook_combined.txt")
    
    # Analyze and plot degree distribution
    detector.plot_degree_distribution(save=True, display=True)
    
    # Compare with random graph
    random_mod = detector.compare_with_random()
    
    # Run all algorithms and collect results
    partition, mod_louvain, time_louvain = detector.louvain_detection()
    labels, mod_spectral, time_spectral = detector.spectral_clustering()
    comm_label, mod_label, time_label = detector.label_propagation()
    
    # Print results
    print("\nResults:")
    print(f"Louvain - Modularity: {mod_louvain:.4f}, Time: {time_louvain:.2f}s")
    print(f"Spectral - Modularity: {mod_spectral:.4f}, Time: {time_spectral:.2f}s")
    print(f"Label Propagation - Modularity: {mod_label:.4f}, Time: {time_label:.2f}s")
    
    # Visualize Louvain communities
    detector.visualize_communities(partition, "Communities Detected by Louvain Method")

if __name__ == "__main__":
    main()
