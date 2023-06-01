import argparse
import copy
import sys
from typing import *

from tree_sitter import Language, Node, Parser, Tree, TreeCursor
import networkx as nx
import numpy as np
import pandas as pd
import pygraphviz as pgv

from graph import Graph as G
from graph import Node as N

Language.build_library(
    'build/my-languages.so',
    ['../tree-sitter-python']
)

PYTHON = Language('build/my-languages.so', 'python')


class ASTFileParser():

    BUILTINS = dir(__builtins__)

    def __init__(self, filepath: str) -> None:
        super().__init__()

        self._parser = Parser()
        self._parser.set_language(PYTHON)

        self._filepath = filepath
        self._tree : Tree = self._get_syntax_tree(self._filepath)
        self._cursor : TreeCursor = self._tree.walk()
        self._root : Node = self._tree.root_node

        self._AST = G()

        self._init_tracking()

    def _init_tracking(self) -> None:
        # track the number of each node type
        self._counts : Dict[str, int] = {}

        # track calls to functions and their locations
        # key: file name
        # value: dict of (function name, node name)
        self._function_calls : Dict[str, Dict[str, str]] = {}

        # track imports and their locations
        # key: file name
        # value: dict of (function name, (node name, import path))
        self._imports : Dict[str, Dict[str, (str, str)]] = {}

        # track function definitions and their locations
        # key: file name
        # value: dict of (function name, node name)
        self._function_definitions : Dict[str, Dict[str, str]] = {}

        # track edges to be added at the end
        # don't add edges right away b/c ruins tree structure and traversal
        # (node_id_to, node_id_from)
        self._edges_to_add : List[Tuple[str, str]] = []

        # track assignments
        # key: file name
        # value: dict of {variable name: (variable type, node name)}
        self._assignments : Dict[str, Dict[str, Tuple[str, str]]] = {}

        # track classes and their attributes
        # key: file name
        # value: dict of (class name, dict of (attribute name, node name))
        self._classes : Dict[str, Dict[str, Dict[str, str]]] = {}

        # track edges for imports from files that have not been read yet
        # value: (node_from_id, file_imported_from, function_imported)
        self._delayed_assignment_edges_to_add : List[Tuple[str, str, str]] = []

        self._delayed_call_edges_to_add : List[Tuple[str, str, str]] = []
        
        # (node_from_id, imported_file, class_type, attribute_name)
        self._delayed_class_attributes_to_add : List[Tuple[str, str, str, str]] = []
    
    def _copy_for_scope(self) -> List[Dict]:
        return [
            copy.deepcopy(self._function_calls),
            copy.deepcopy(self._function_definitions),
            copy.deepcopy(self._assignments),
            copy.deepcopy(self._classes),
        ]

    @property
    def AST(self) -> dict[str, Any]:
        return self._AST

    @AST.setter
    def AST(self, value: dict[str, Any]) -> None:
        raise Exception("AST is read-only. Use parse() instead.")
    
    def __str__(self) -> str:
        if not self._AST:
            raise Exception("AST is empty. Use parse() first.")
        return str(self._AST)
    
    def _get_syntax_tree(self, filepath: str) -> Tree:
        with open (filepath, "r") as myfile:
            file = myfile.read()
        return self._parser.parse(bytes(file, "utf8"))
    
    def parse(self) -> str:
    
        def _parse_node(node: Node, parent: G, last_node: Union[N, None]) -> str:
            # add text if node is terminal
            text = None
            if node.is_named and len(node.children) == 0:
                text = node.text.decode("utf-8")
            if node.type == 'binary_operator':
                text = node.children[1].text.decode("utf-8")
            # add text to attribute nodes
            if node.type == 'attribute':
                text = node.text.decode("utf-8")            
            
            name = node.type if not text else node.type + ' | ' + text

            # TODO: does this make this better or worse?
            # condense dotted attributes
            # if node.type == 'attribute':                
            #     text = node.text.decode('utf-8')
            #     name = 'identifier | ' + text

            # add file name to root node
            if node.type == 'module':
                name = node.type + ' | ' + self._filepath
            else:
                if name not in self._counts:
                    self._counts[name] = 0
                    name = name + '_' + str(self._counts[name])
                else:
                    self._counts[name] += 1
                    name = name + '_' + str(self._counts[name])
            
            n_ = N(name, node.start_point, node.end_point, type = node.type, parent = last_node)
            if text:
                n_.text = text

            # if node.type == 'attribute':
            #     n_.type = 'identifier'
            #     id = parent.add_vertex(n_)
            #     return id

            # add the node to the graph
            id = parent.add_vertex(n_)

            # track variable name for identifier nodes
            if node.type == 'identifier':
                n_.var_name = node.text.decode("utf-8")

            # handle function calls
            if node.type == 'call' and node.children[0].text.decode("utf-8") not in self.BUILTINS:
                self._handle_call(node, parent, name)

            # handle imports
            if node.type == "aliased_import" or \
                (node.type == "dotted_name" and node.parent.type.startswith("import")):
                self._handle_import(node, parent, name)

            # handle function definitions
            if node.type == 'function_definition' or node.type == 'class_definition':
                self._handle_definition(node, parent, name)
            
            for child in node.children:
                # only use named nodes
                if not child.is_named:
                    continue
                to_id_ = _parse_node(child, parent, last_node = n_)
                parent.add_edge(n_.id, to_id_)
            
            return id
    
        root_id = _parse_node(self._root, self._AST, last_node = None)

        # check if this is a file or dir parser
        if type(self) == ASTFileParser:
            self._resolve_imports(self._AST)

        return root_id

    def _handle_call(self, node: Node, parent: G, id: str) -> None:
        # get function name
        function_name = node.children[0].text.decode("utf-8")
        # add function call to dict
        if self._filepath not in self._function_calls:
            self._function_calls[self._filepath] = {function_name: id}
        else:
            self._function_calls[self._filepath][function_name] = id

        # add edge from the call to the import statment if it exists
        self._call_to_import(function_name, parent, id)
    
    def _call_to_import(self, function_call: str, parent: G, id: str) -> None:
        if self._filepath in self._imports and function_call in self._imports[self._filepath]:
            # parent.add_edge(id, self._imports[self._filepath][function_call])
            self._edges_to_add.append((id, self._imports[self._filepath][function_call][0]))
            return
        if '.' in function_call:
            # function_name = function_name if len(function_name.split('.')) <= 1 else function_name.split('.')[0]
            # TODO: fix this to work with attributes
            function_call = function_call[:function_call.rfind('.')]
            self._call_to_import(function_call, parent, id)
        
    def _handle_import(self, node: Node, parent: G, id: str) -> None:
        if node.type == 'aliased_import':
            if node.parent.type == 'import_from_statement':
                import_path = node.parent.children[1].text.decode("utf-8") + '.' + node.children[0].text.decode("utf-8")
            elif node.parent.type == 'import_statement':
                import_path = node.children[0].text.decode("utf-8")
            import_name = [(node.children[2].text.decode("utf-8"), id, import_path)]
        elif node.type == 'dotted_name':
            # skip the first dotted name of the import from
            if node.parent.type == 'import_from_statement' and node.parent.children[1] == node:
                return
            if node.parent.type == 'import_from_statement':
                import_path = node.parent.children[1].text.decode("utf-8")
            elif node.parent.type == 'import_statement':
                # import_path = node.text.decode("utf-8")
                import_path = ""
            import_name = [(node.text.decode("utf-8"), id, import_path)]
            
        # add import to dict
        for import_, id_, import_path_ in import_name:
            # get import location
            if self._filepath not in self._imports:
                self._imports[self._filepath] = {import_: (id_, import_path_)}
            else:
                self._imports[self._filepath][import_] = (id_, import_path_)

    def _handle_definition(self, node: Node, parent: G, id: str) -> None:
        # get function name
        function_name = node.children[1].text.decode("utf-8")
        # add function definition to dict
        if self._filepath not in self._function_definitions:
            self._function_definitions[self._filepath] = {function_name: id}
        else:
            self._function_definitions[self._filepath][function_name] = id

    # TODO: make this work with single files again
    def _resolve_imports(self, parent: G) -> None:
        # connect all function calls to their definitions
        if not self._function_calls:
            return
        for function_name in self._function_calls[self._filepath]:
            # check if function is defined
            if self._function_definitions and function_name in self._function_definitions[self._filepath]:
                # add edge
                # for call_function_name, call_node_name in self._function_calls[self._filepath].items():
                #     for definition_location, definition_node_name in self._function_definitions[function_name]:
                #         if call_location == definition_location:
                parent.add_edge(self._function_calls[self._filepath][function_name], self._function_definitions[self._filepath][function_name])
                parent.add_edge(self._function_definitions[self._filepath][function_name], self._function_calls[self._filepath][function_name])

        # add import edges at the end
        for edge_from, edge_to in self._edges_to_add:
            parent.add_edge(edge_from, edge_to)

    def save_dot_format(self, filepath: str = 'tree.gv') -> str:
        if not self._AST:
            raise Exception("AST is empty. Use parse() first.")
        return self._get_dot_format(filepath)
    
    def _get_dot_format(self, filepath: str) -> str:
        edges = []
        nodes_ : List[str] = self._AST.get_vertices()
        nodes = []

        for node in nodes_:
            n_ : N = self._AST.get_vertex(node)
            nodes.append((n_.id, n_._start, n_._end))
            
            for child in n_.get_connections():
                edges.append((n_.id, child.id))

        real_stdout = sys.stdout
        sys.stdout = open(filepath, 'w')

        # Dump edge list in Graphviz DOT format
        print('strict digraph tree {')
        for row in edges:
            print('    "{0}" -> "{1}";'.format(*row))
        for node in nodes:
            print('    "{0}" [xlabel="{1}->{2}"];'.format(*node))
        print('}')

        sys.stdout.close()
        sys.stdout = real_stdout
    
    def convert_to_graphviz(self) -> pgv.AGraph:
        if not self._AST:
            raise Exception("AST is empty. Use parse() first.")
        return self._convert_to_graphviz()
    
    def _convert_to_graphviz(self) -> pgv.AGraph:
        nodes = self._AST.get_vertices()
        edges = []
        # g = Digraph('G', filename='tree.gv')
        g = pgv.AGraph(strict=True, directed=True)


        for node in nodes:
            n : N = self._AST.get_vertex(node)
            g.add_node(
                n.id,
                xlabel=f'{n._start}->{n._end}',
            )
            edges.extend([(n.id, x.id) for x in n.get_connections()])

        g.add_edges_from(edges)
        g.write('tree.gv')
        return g

    def to_csv(self, nf: str, adj: str) -> None:
        if not self._AST:
            raise Exception("AST is empty. Use parse() first.")
        self._to_csv(nf, adj)

    def _to_csv(self, nf: str, adj: str) -> None:
        g : pgv.AGraph = self.convert_to_graphviz()
        g : nx.DiGraph = nx.nx_agraph.from_agraph(g)

        nodes = [n for n in g.nodes()]
        feats = [feat['xlabel'] for node, feat in dict(g.nodes(data=True)).items()]
        node_feats = pd.DataFrame({'node': nodes, 'feat': feats})
        node_feats.to_csv(nf, index = False)
        print(f'Saved node features to {nf}')
        del node_feats
        del nodes
        del feats
        adj_np = nx.to_numpy_array(g, dtype = np.bool_, weight = None)
        np.savetxt(adj, adj_np, delimiter = ',', fmt = '%.0f')
        print(f'Saved adjacency matrix to {adj}')

    def _to_networkx(self) -> nx.DiGraph:
        g : pgv.AGraph = self.convert_to_graphviz()
        return nx.nx_agraph.from_agraph(g)

    def view_k_neighbors(self,
                         node_id: str,
                         k: int = 10
                        ) -> None:
        g : nx.DiGraph = self._to_networkx()
        g_k = pgv.AGraph(strict=True, directed=True)
        g_k.add_node(node_id)

        depth = 0

        def neighbors(g: nx.DiGraph, node_id: str, depth: int) -> None:
            if depth >= k:
                return
            depth += 1
            for neighbor in g.neighbors(node_id):
                g_k.add_edge(node_id, neighbor)
                neighbors(g, neighbor, depth)

        neighbors(g, node_id, depth)

        g_k.write('tree.gv')

        

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--file", type=str, required=True, help="Path to file to parse")
    args = arg_parser.parse_args()

    ast = ASTFileParser(args.file)
    ast.parse()
    ast.convert_to_graphviz()
    print(ast._imports)
    print(ast._function_calls)
    print(ast._function_definitions)
    # ast.to_csv()

    # import ast
    # print(ast.dump(ast.parse(file), indent = 5))

if __name__ == "__main__":
    main()