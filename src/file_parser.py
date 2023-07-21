import argparse
import copy
import gc
import os
import re
import sys
from typing import *

import fasttext
import fasttext.util
import networkx as nx
import numpy as np
import pandas as pd
import pygraphviz as pgv
import scipy.sparse
from tree_sitter import Language, Node, Parser, Tree

from graph import Graph as G
from graph import Node as N

fasttext.FastText.eprint = lambda x: None

Language.build_library(
    'build/my-languages.so',
    [os.path.expanduser('~/nfsdata/workspace/tree-sitter-python')]
)

PYTHON = Language('build/my-languages.so', 'python')
CONST = 10e-4


class ASTFileParser():

    __slots__ = (
        "_parser",
        "_filepath",
        "_tree",
        "_root",
        "_AST",
        "_counts",
        "_function_calls",
        "_imports",
        "_function_definitions",
        "_edges_to_add",
        "_assignments",
        "_classes",
        "_delayed_assignment_edges_to_add",
        "_delayed_call_edges_to_add",
        "_delayed_class_attributes_to_add",
        "_dim",
        "_ft",
    )

    BUILTINS = dir(__builtins__)

    def __init__(self, filepath: str) -> None:
        super().__init__()

        self._parser = Parser()
        self._parser.set_language(PYTHON)

        self._filepath = filepath
        self._tree : Tree = self._get_syntax_tree(self._filepath)
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

    def _cleanup(self) -> None:
        """
            Delete and clear objects that are no longer needed.
                All that needs to be kept at this point is the AST
        """
        self._tree = None
        self._root = None
        self._counts.clear()
        self._function_calls.clear()
        self._imports.clear()
        self._function_definitions.clear()
        self._edges_to_add.clear()
        self._assignments.clear()
        self._classes.clear()
        self._delayed_assignment_edges_to_add.clear()
        self._delayed_call_edges_to_add.clear()
        self._delayed_class_attributes_to_add.clear()
        gc.collect()

    @property
    def AST(self) -> Dict[str, Any]:
        return self._AST

    @AST.setter
    def AST(self, value: Dict[str, Any]) -> None:
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
    
        def _parse_node(node: Node, parent: G, last_node: Union[N, None], filename: str) -> str:
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
            
            n_ = N(name, node.start_point, node.end_point, filename, type = node.type, parent = last_node)
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
                to_id_ = _parse_node(child, parent, last_node = n_, filename=filename)
                parent.add_edge(n_.id, to_id_)
            
            return id
    
        root_id = _parse_node(self._root, self._AST, last_node = None, filename = self._filepath)

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
                label=n.file,
            )
            edges.extend((n.id, x.id) for x in n.get_connections())

        g.add_edges_from(edges)
        return g

    def to_csv(self, nf: str, adj: str) -> None:
        if not self._AST:
            raise Exception("AST is empty. Use parse() first.")
        self._to_csv(nf, adj)

    def _to_csv(self, nf: str, adj: str) -> None:
        g : pgv.AGraph = self.convert_to_graphviz()
        g : nx.DiGraph = nx.nx_agraph.from_agraph(g)

        nodes = (n for n in g.nodes())
        feats = (feat['xlabel'] for node, feat in dict(g.nodes(data=True)).items())
        files = (feat['label'] for node, feat in dict(g.nodes(data=True)).items())
        node_feats = pd.DataFrame({'node': nodes, 'feat': feats, 'file': files})
        node_feats.to_csv(f"{nf}.csv", index = False)
        print(f'Saved node features to {nf}.csv')
        del node_feats
        del nodes
        del feats
        del files
        adj_sparse = nx.to_scipy_sparse_array(g, dtype = np.bool_, weight = None)
        scipy.sparse.save_npz(adj, adj_sparse)
        print(f'Saved adjacency matrix to {adj}.npz')
        del adj_sparse
        gc.collect()

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

    def csv_features_to_vectors(self, nf: str) -> None:
        # check that the files exist
        if not os.path.exists(f"{nf}.csv"):
            raise Exception(f'File {nf}.csv does not exist.')
        else:
            self._csv_features_to_vectors(nf)
        
    def _csv_features_to_vectors(self, nf: str) -> None:
        df = pd.read_csv(f"{nf}.csv", header = 0)
        if os.path.exists(f'cc.en.{self._dim // 4}.bin'):
            ft = fasttext.load_model(f'cc.en.{self._dim // 4}.bin')
        else:
            fasttext.util.download_model('en', if_exists='ignore')
            ft = fasttext.load_model('cc.en.300.bin')
            fasttext.util.reduce_model(ft, self._dim // 4)
            ft.save_model(f'cc.en.{self._dim // 4}.bin')
        self._ft = ft

        # define the embedding functions
        def location_to_embed(location: str) -> np.ndarray:
            dim = self._dim // 4
            # get number between first parentheses and comma
            x = int(re.search(r"\(([0-9]+),", location).groups()[0])
            y = int(re.search(r",\s([0-9]+)\)", location).groups()[0])
            res = np.zeros(dim)
            i = np.arange(dim // 4)
            res[2*i] = np.sin(x * CONST ** (4 * i / dim))
            res[2*i + 1] = np.cos(x * CONST ** (4 * i / dim))
            res[2*i + dim // 2] = np.sin(y * CONST ** (4 * i / dim))
            res[2*i + dim // 2 + 1] = np.cos(y * CONST ** (4 * i / dim))
            return res

        def type_to_embed(type_: str, ft: fasttext.FastText._FastText) -> np.ndarray:
            return ft.get_word_vector(type_)
        
        def text_to_embed(text: str, ft: fasttext.FastText._FastText) -> np.ndarray:
            return ft.get_word_vector(text)
        
        def embed(start: str, end: str, type_: str, text: str, ft: fasttext.FastText._FastText) -> np.ndarray:
            return np.concatenate([location_to_embed(start), location_to_embed(end), type_to_embed(type_, ft), text_to_embed(text, ft)], axis = 0)

        def get_node_text(node_id: str) -> str:
            return '' if ' | ' not in node_id else (node_id.split(' | ')[1] if not re.match(r"(.*)(_[0-9]+$)", node_id.split(' | ')[1]) else re.match(r"(.*)(_[0-9]+$)", node_id.split(' | ')[1]).groups()[0])
        
        def get_node_type(node_id: str) -> str:
            return node_id[:node_id.rfind('_')] if ' | ' not in node_id else node_id.split(' | ')[0]
        
        # extract features to columns
        df['start'] = df['feat'].apply(lambda x: x.split('->')[0])
        df['end'] = df['feat'].apply(lambda x: x.split('->')[1])
        df['text'] = df['node'].apply(lambda x: get_node_text(x))
        df['type'] = df['node'].apply(lambda x: get_node_type(x))

        feats = df.apply(
            lambda row: embed(row.start, row.end, row.type, row.text, self._ft),
            axis = 1,
            result_type = "expand"
        )
        feats['start'] = df['start']
        feats['end'] = df['end']
        feats['file'] = df['file']
        feats.index = df['node']
        feats.to_csv(f"{nf}.csv")

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