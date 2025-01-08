import pandas as pd
import networkx as nx
import os
import numpy as np
from Arguments import parser

def compute_dot_products(tf_embeddings, target_embeddings):
    dot_products = np.dot(tf_embeddings.values, target_embeddings.values.T)
    probabilities = 1 / (1 + np.exp(-dot_products))
    return probabilities

def con_net(args):
    Channel1_dir = os.path.join('Result', 'B 500', 'Channel1.csv')
    Channel2_dir = os.path.join('Result', 'B 500', 'Channel2.csv')
    Network_dir = os.path.join('dataset', 'hTFTarget', 'B', 'TFs+500', 'BL--Network.csv')
    save_dir = os.path.join('Result', 'B 500', 'B_out.csv')

    tf_embeddings = pd.read_csv(Channel1_dir, index_col=0)
    target_embeddings = pd.read_csv(Channel2_dir, index_col=0)
    network_data = pd.read_csv(Network_dir)

    probabilities = compute_dot_products(tf_embeddings, target_embeddings)

    tf_names = tf_embeddings.index
    target_names = target_embeddings.index

    gold_tf_indices = [tf_embeddings.index.get_loc(tf_name) for tf_name in network_data['TF'] if
                       tf_name in tf_embeddings.index]

    selected_probabilities = probabilities[gold_tf_indices, :]

    threshold = np.percentile(selected_probabilities, args.density)

    print(threshold)

    predicted_edges = set()

    gold_tf_names = set(network_data['TF'])

    for tf_name in gold_tf_names:
        if tf_name in tf_embeddings.index:
            i = tf_embeddings.index.get_loc(tf_name)
            for j, target_name in enumerate(target_names):
                if tf_name != target_name and probabilities[i, j] > threshold:
                    predicted_edges.add((tf_name, target_name))

    gold_edges = set(zip(network_data['TF'], network_data['Target']))

    retained = gold_edges & predicted_edges
    new_predicted = predicted_edges - gold_edges

    G = nx.DiGraph()
    G.add_nodes_from(tf_embeddings.index.tolist() + target_embeddings.index.tolist())

    for edge in retained:
        G.add_edge(*edge, color='green')
    for edge in new_predicted:
        G.add_edge(*edge, color='blue')

    edges_list = []

    for edge in G.edges():
        color = G.edges[edge]['color']
        edges_list.append((edge[0], edge[1], color))

    edges_df = pd.DataFrame(edges_list, columns=['Source', 'Target', 'Color'])

    edges_df.to_csv(save_dir, index=False)

    print(f"Saved the predicted edges to {save_dir}")

if __name__ == '__main__':
    args = parser.parse_args()
    con_net(args)

