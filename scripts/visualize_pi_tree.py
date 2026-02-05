import argparse
import os
import glob
import re
import subprocess

def parse_clause(clause_str):
    """
    Parses a Prolog clause string into head and body literals.
    Example: action(A, fire) :- lit1(A), lit2(A, B), action(A, 0.5).
    """
    clause_str = clause_str.strip()
    if not clause_str.endswith('.'):
        return None
    clause_str = clause_str[:-1] # Remove trailing dot
    
    if ':-' in clause_str:
        head_str, body_str = clause_str.split(':-', 1)
        
        # Remove comments /* ... */
        body_str = re.sub(r'/\*.*?\*/', '', body_str)
        
        # Parse body literals
        # This is a naive split by comma, might fail with nested terms but suffice for these BKs
        # Detailed parsing might be needed if complex terms are used
        body_lits = []
        current_lit = ""
        depth = 0
        for char in body_str:
            if char == ',' and depth == 0:
                l = current_lit.strip()
                if l and l != '!': # Ignore empty literals and cuts
                    body_lits.append(l)
                current_lit = ""
            else:
                current_lit += char
                if char == '(': depth += 1
                if char == ')': depth -= 1
        
        l = current_lit.strip()
        if l and l != '!':
            body_lits.append(l)
            
        return head_str.strip(), body_lits
    else:
        # Fact
        return clause_str.strip(), []

def build_trie(clauses):
    """
    Builds a trie structure from the clauses.
    To generate a proper decision tree, we need to handle the structure carefully.
    Clauses sharing a prefix share the same path.
    """
    root = {'children': {}, 'value': None, 'count': 0}
    
    for head, body in clauses:
        node = root
        node['count'] += 1
        
        path = body
        
        for lit in path:
            if lit not in node['children']:
                node['children'][lit] = {'children': {}, 'value': None, 'count': 0}
            node = node['children'][lit]
            node['count'] += 1
            
        # Extract value from head
        match = re.search(r'([0-9.-]+)\)$', head)
        if match:
            node['value'] = match.group(1)
        else:
            node['value'] = head # fallback
            
    return root

class DotGenerator:
    def __init__(self, name):
        self.lines = [f'digraph "{name}" {{', 'rankdir=LR;']
        self.nodes = set()
        
    def node(self, name, label, shape='box', style='', color=''):
        if name in self.nodes: return
        self.nodes.add(name)
        attr = f'label="{label}"'
        if shape: attr += f', shape={shape}'
        if style: attr += f', style={style}'
        if color: attr += f', color={color}'
        self.lines.append(f'"{name}" [{attr}];')
        
    def edge(self, src, dst, label=""):
        attr = f'label="{label}"' if label else ""
        self.lines.append(f'"{src}" -> "{dst}" [{attr}];')
        
    def save(self, path):
        with open(path, 'w') as f:
            f.write('\n'.join(self.lines) + '\n}')

def dict_to_graph(node, graph, parent_id="root", edge_label=""):
    node_id = str(id(node))
    
    label = ""
    # Check if leaf
    if not node['children']:
        label = f"Leaf: {node['value']}"
        graph.node(node_id, label=label, shape='box', style='filled', color='lightblue')
    else:
        # Inner node
        graph.node(node_id, label="", shape='point')

    if parent_id != "root":
        graph.edge(parent_id, node_id, label=edge_label)
        
    for lit, child in node['children'].items():
        dict_to_graph(child, graph, node_id, edge_label=lit)

def visualize_tree_file(tree_file, output_path):
    print(f"Processing {tree_file}...")
    with open(tree_file, 'r') as f:
        content = f.read()
        
    raw_clauses = re.split(r'\.\s+', content)
    
    parsed_clauses = []
    for rc in raw_clauses:
        if not rc.strip(): continue
        parsed = parse_clause(rc + '.')
        if parsed:
            parsed_clauses.append(parsed)
            
    root = build_trie(parsed_clauses)
    
    dot = DotGenerator(os.path.basename(tree_file))
    
    # Root node
    dot.node("root", "Root", shape='oval')
    
    for lit, child in root['children'].items():
        dict_to_graph(child, dot, "root", lit)
        
    dot_path = output_path + ".dot"
    png_path = output_path + ".png"
    
    dot.save(dot_path)
    print(f"Saved .dot file to {dot_path}")
    
    # Try running dot command
    try:
        subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True)
        print(f"Saved visualization to {png_path}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Warning: 'dot' command not found or failed. Please install Graphviz to generate PNG.")

def main():
    parser = argparse.ArgumentParser(description='Visualize PI Trees')
    parser.add_argument('--model_dir', type=str, required=True, help='Path to the model directory containing PI_Model/Trees')
    args = parser.parse_args()
    
    pi_trees_dir = os.path.join(args.model_dir, 'PI_Model', 'Trees')
    if not os.path.exists(pi_trees_dir):
        print(f"Error: Directory {pi_trees_dir} does not exist.")
        return

    tree_files = glob.glob(os.path.join(pi_trees_dir, '*.tree'))
    if not tree_files:
        print(f"No .tree files found in {pi_trees_dir}")
        return
        
    output_dir = os.path.join(args.model_dir, 'PI_Model', 'Plots')
    os.makedirs(output_dir, exist_ok=True)
    
    for tree_file in tree_files:
        filename = os.path.basename(tree_file)
        name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, name)
        visualize_tree_file(tree_file, output_path)

if __name__ == "__main__":
    main()
