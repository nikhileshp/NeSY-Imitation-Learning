import os
import re
import sys
import matplotlib.pyplot as plt
import networkx as nx
import argparse

# --- Tree Parsing ---

class Node:
    def __init__(self, label, depth, edge_label=None):
        self.label = label
        self.depth = depth
        self.edge_label = edge_label
        self.children = []
        self.is_diff = False
        self.id = None # Assigned later for graph

    def __repr__(self):
        return f"Node({self.label}, depth={self.depth})"

def parse_tree_file(file_path):
    if not os.path.exists(file_path):
        return None

    with open(file_path, 'r') as f:
        lines = f.readlines()

    # Find the ASCII tree section
    start_index = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("% FOR action"):
            start_index = i
            break
    
    if start_index == -1:
        return None

    root = None
    stack = [] # (node, depth)
    
    # Regex for parsing lines
    # Examples:
    # %   if ( facingleft(A) )
    # %   then if ( diversnotfull(A) )
    # %   | then if ( visiblediver(A, C), samelevelasdiver(A, C) )
    # %   | | then return 0.729...
    # %   | | else return 0.464...
    
    for i in range(start_index + 1, len(lines)):
        line = lines[i].strip()
        if not line.startswith("%"):
            break
        
        content = line[1:].strip() # Remove %
        if not content:
            continue
            
        # Count bars for depth
        bars = content.count('|')
        clean_content = content.replace('|', '').strip()
        
        node_label = ""
        edge_label = ""
        
        if clean_content.startswith("if"):
            match = re.search(r"if \((.*)\)", clean_content)
            if match:
                node_label = match.group(1).strip()
                edge_label = "if"
        elif clean_content.startswith("then if"):
             match = re.search(r"then if \((.*)\)", clean_content)
             if match:
                 node_label = match.group(1).strip()
                 edge_label = "True"
        elif clean_content.startswith("else if"):
             match = re.search(r"else if \((.*)\)", clean_content)
             if match:
                 node_label = match.group(1).strip()
                 edge_label = "False"
        elif clean_content.startswith("then return"):
             match = re.search(r"then return ([\d\.\-E]+)", clean_content)
             if match:
                 node_label = f"{float(match.group(1)):.3f}"
                 edge_label = "True"
        elif clean_content.startswith("else return"):
             match = re.search(r"else return ([\d\.\-E]+)", clean_content)
             if match:
                 node_label = f"{float(match.group(1)):.3f}"
                 edge_label = "False"
        elif clean_content.startswith("return"):
             match = re.search(r"return ([\d\.\-E]+)", clean_content)
             if match:
                 node_label = f"{float(match.group(1)):.3f}"
                 edge_label = ""
        
        if not node_label:
            continue

        new_node = Node(node_label, bars, edge_label)
        
        if root is None:
            root = new_node
            stack.append(new_node)
        else:
            target_parent_index = bars
            
            if clean_content.startswith("else"):
                # We want to pop the sibling, so we need stack size to be target + 2
                while len(stack) > target_parent_index + 2:
                    stack.pop()
                # Now pop the sibling
                if len(stack) > 1:
                    stack.pop()
            else:
                # We want to add to parent, so we need stack size to be target + 1
                while len(stack) > target_parent_index + 1:
                    stack.pop()
            
            if stack:
                parent = stack[-1]
                parent.children.append(new_node)
                stack.append(new_node)
            else:
                # Should not happen if logic is correct and root exists
                print(f"Warning: Stack empty when processing: {clean_content}")

    return root

# --- Tree Comparison ---

def compare_trees(node1, node2):
    if node1 is None and node2 is None:
        return
    
    if node1 is None or node2 is None:
        if node2: node2.is_diff = True
        return

    # Compare labels (fuzzy float comparison for returns)
    match = True
    # Check if both are numbers (leaves)
    try:
        val1 = float(node1.label)
        val2 = float(node2.label)
        if abs(val1 - val2) > 1e-4:
            match = False
    except ValueError:
        # Not numbers, compare strings
        if node1.label != node2.label:
            match = False
    
    if not match:
        node2.is_diff = True
    
    # Compare children
    len1 = len(node1.children)
    len2 = len(node2.children)
    
    for i in range(max(len1, len2)):
        c1 = node1.children[i] if i < len1 else None
        c2 = node2.children[i] if i < len2 else None
        compare_trees(c1, c2)

import textwrap

# --- Metadata Extraction ---

def get_metadata(model_dir, tree_file_path):
    meta = {
        "ratio": "Unknown",
        "depth": "Unknown",
        "alpha": "None",
        "sampling": "False"
    }
    
    # Extract from directory name
    dir_name = os.path.basename(model_dir)
    # negpos_2_trees_1_depth_3_grounding_penalty_0.1_new
    
    match_alpha = re.search(r"grounding_penalty_([0-9\.]+)", dir_name)
    if match_alpha:
        meta["alpha"] = match_alpha.group(1)
    elif "grounding_penalty" not in dir_name:
        meta["alpha"] = "None"
        
    if "_new" in dir_name:
        meta["sampling"] = "True (100)"
        
    # Extract from tree file
    if os.path.exists(tree_file_path):
        with open(tree_file_path, 'r') as f:
            for line in f:
                if "negPosRatio" in line:
                    match = re.search(r"=\s*([\d\.]+)", line)
                    if match: meta["ratio"] = match.group(1)
                if "maxTreeDepthInNodes" in line:
                    match = re.search(r"=\s*(\d+)", line)
                    if match: meta["depth"] = match.group(1)
                if meta["ratio"] != "Unknown" and meta["depth"] != "Unknown":
                    break
    
    return meta

# --- Visualization ---

def hierarchy_pos(G, root=None, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5):
    '''
    From Joel's answer at https://stackoverflow.com/questions/29586520/can-one-get-hierarchical-graphs-from-networkx-with-python-3/29597209#29597209
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap = 0.2, vert_loc = 0, xcenter = 0.5, pos = None, parent = None):
        if pos is None:
            pos = {root:(xcenter,vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)  
        if len(children)!=0:
            dx = width/len(children) 
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G,child, width = dx, vert_gap = vert_gap, 
                                    vert_loc = vert_loc-vert_gap, xcenter=nextx,
                                    pos=pos, parent = root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)

def build_graph(node, G, parent_id=None, node_id_counter=None):
    if node_id_counter is None:
        node_id_counter = [0]
        
    current_id = node_id_counter[0]
    node.id = current_id
    node_id_counter[0] += 1
    
    # Determine type for visualization
    # Leaf if label is a number
    try:
        float(node.label)
        is_leaf = True
    except ValueError:
        is_leaf = False
    
    G.add_node(current_id, label=node.label, is_diff=node.is_diff, is_leaf=is_leaf)
    
    if parent_id is not None:
        G.add_edge(parent_id, current_id, label=node.edge_label, is_diff=node.is_diff)
    
    for child in node.children:
        build_graph(child, G, current_id, node_id_counter)

def draw_tree(ax, root, meta, action):
    if root is None:
        ax.text(0.5, 0.5, "No Tree", ha='center')
        ax.axis('off')
        return

    G = nx.DiGraph()
    build_graph(root, G, node_id_counter=[0])
    
    pos = hierarchy_pos(G, 0)
    
    # Separate nodes by type for drawing
    leaves = [n for n, d in G.nodes(data=True) if d.get('is_leaf')]
    internals = [n for n, d in G.nodes(data=True) if not d.get('is_leaf')]
    
    # Internal nodes (Ovals)
    node_colors_int = []
    for n in internals:
        if G.nodes[n].get('is_diff', False):
            node_colors_int.append('mistyrose')
        else:
            node_colors_int.append('lightblue')
            
    nx.draw_networkx_nodes(G, pos, nodelist=internals, node_shape='o', node_color=node_colors_int, 
                           node_size=6000, ax=ax, edgecolors=['red' if c == 'mistyrose' else 'black' for c in node_colors_int])

    # Leaf nodes (Boxes/Squares)
    node_colors_leaf = []
    for n in leaves:
        if G.nodes[n].get('is_diff', False):
            node_colors_leaf.append('mistyrose')
        else:
            node_colors_leaf.append('lightgreen') # Different color for leaves
            
    nx.draw_networkx_nodes(G, pos, nodelist=leaves, node_shape='s', node_color=node_colors_leaf, 
                           node_size=4000, ax=ax, edgecolors=['red' if c == 'mistyrose' else 'black' for c in node_colors_leaf])

    # Edges
    edge_colors = []
    for e in G.edges(data=True):
        if e[2].get('is_diff', False):
            edge_colors.append('red')
        else:
            edge_colors.append('black')
    
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color=edge_colors, arrows=True)
    
    # Labels with wrapping
    labels = {}
    for n in G.nodes:
        text = G.nodes[n]['label']
        # Wrap text
        labels[n] = textwrap.fill(text, width=15)
            
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)
    
    # Edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, ax=ax, font_size=9)
    
    # Title
    title_text = (
        f"Neg Pos Ratio : {meta['ratio']}\n"
        f"Depth : {meta['depth']}\n"
        f"Grounding Alpha : {meta['alpha']}\n"
        f"Sampling : {meta['sampling']}\n"
        f"Action : {action}"
    )
    ax.set_title(title_text, loc='left', fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    ax.axis('off')

# --- Metrics ---

def get_metrics(model_dir, action, seed="seed_42"):
    f1 = 0.0
    auc_pr = 0.0
    auc_roc = 0.0
    
    log_path = os.path.join(model_dir, action, seed, f"test_infer_{seed}.log")
    
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            
            # Extract metrics using regex
            roc_match = re.search(r"%\s+AUC ROC\s+=\s+([\d\.]+)", content)
            pr_match = re.search(r"%\s+AUC PR\s+=\s+([\d\.]+)", content)
            f1_match = re.search(r"%\s+F1\s+=\s+([\d\.]+)", content)
            
            if roc_match: auc_roc = float(roc_match.group(1))
            if pr_match: auc_pr = float(pr_match.group(1))
            if f1_match: f1 = float(f1_match.group(1))
            
    return f1, auc_pr, auc_roc

# --- Main ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir1", help="Path to first model directory")
    parser.add_argument("dir2", help="Path to second model directory")
    args = parser.parse_args()
    
    actions = ["down", "fire", "left", "noop", "right", "up"]
    seed = "seed_42"
    
    # Create output directory
    name1 = os.path.basename(args.dir1.rstrip('/'))
    name2 = os.path.basename(args.dir2.rstrip('/'))
    output_dir = f"plots/compare_m1_{name1}_m2_{name2}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to: {output_dir}")
    
    for action in actions:
        print(f"Processing action: {action}")
        
        # Paths
        tree1_path = os.path.join(args.dir1, action, seed, "WILLtheories", "action_learnedWILLregressionTrees.txt")
        tree2_path = os.path.join(args.dir2, action, seed, "WILLtheories", "action_learnedWILLregressionTrees.txt")
        
        # Parse
        root1 = parse_tree_file(tree1_path)
        root2 = parse_tree_file(tree2_path)
        
        # Compare
        compare_trees(root1, root2)
        
        # Metadata
        meta1 = get_metadata(args.dir1, tree1_path)
        meta2 = get_metadata(args.dir2, tree2_path)
        
        # Metrics
        m1 = get_metrics(args.dir1, action)
        m2 = get_metrics(args.dir2, action)
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(40, 16)) # Increased width for better visibility
        
        draw_tree(axes[0], root1, meta1, action)
        draw_tree(axes[1], root2, meta2, action)
        
        # Add metrics text
        diff_f1 = m2[0] - m1[0]
        diff_pr = m2[1] - m1[1]
        diff_roc = m2[2] - m1[2]
        
        metrics_text = (
            f"Metrics Difference (Right - Left):\n"
            f"F1: {m2[0]:.4f} - {m1[0]:.4f} = {diff_f1:+.4f}\n"
            f"AUC-PR: {m2[1]:.4f} - {m1[1]:.4f} = {diff_pr:+.4f}\n"
            f"AUC-ROC: {m2[2]:.4f} - {m1[2]:.4f} = {diff_roc:+.4f}"
        )
        
        plt.figtext(0.5, 0.02, metrics_text, ha="center", fontsize=14, bbox={"facecolor":"white", "alpha":0.8, "pad":10})
        
        output_file = os.path.join(output_dir, f"comparison_{action}.png")
        plt.savefig(output_file, dpi=150)
        print(f"Saved {output_file}")
        plt.close()

if __name__ == "__main__":
    main()
