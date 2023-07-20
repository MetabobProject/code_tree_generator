import argparse
import os
from typing import *
import re

from tree_sitter import Language, Node, Parser, Tree, TreeCursor

from file_parser import ASTFileParser
from graph import Graph as G
from graph import Node as N

Language.build_library(
    'build/my-languages.so',
    [os.path.expanduser('~/nfsdata/workspace/tree-sitter-python')]
)

PYTHON = Language('build/my-languages.so', 'python')

class ASTCodebaseParser(ASTFileParser):

    BUILTINS = dir(__builtins__)

    def __init__(self, dir: str, dim: int) -> None:
        self._dir : str = dir
        self._dim : int = dim
        self._relative_files = self.get_files()

        self._parser = Parser()
        self._parser.set_language(PYTHON)

        self._AST = G()

        self._init_tracking()

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
    
    def get_files(self) -> List[str]:
        files = []
        for (dirpath, dirnames, filenames) in os.walk(self._dir):
            files.extend(
                [
                    os.path.relpath(os.path.join(os.path.join(os.getcwd(), dirpath) , x))
                    for x in filenames if x.endswith(".py")
                ]
            )
        return files
    
    def parse_dir(self) -> None:
        roots = []
        # i = 0
        for file in self._relative_files:
            # print(f'done {i}')
            self._filepath = file
            tree = self._get_syntax_tree(file)
            self._root = tree.root_node
            root_id = self.parse()
            roots.append(root_id)
            # i += 1
        # clear assignments, definition and classes
        self._function_definitions = {}
        self._assignments = {}
        self._classes = {}
        # second loop
        # i = 0
        for root in roots:
            # print(f'Second {i}')
            filepath = root.split(' | ')[1]
            self._second_loop(root, self._AST, filepath)
            # i+=1
        self._add_edges(self._AST)
        self._add_delayed_assignment_edges(self._AST)
        self._add_delayed_call_edges(self._AST)
        self._add_delayed_attribute_edges(self._AST)

    def _add_edges(self, parent: G) -> None:
        # connect import edges to their calls
        for edge_from, edge_to in self._edges_to_add:
            parent.add_edge(edge_from, edge_to)

    def _add_delayed_assignment_edges(self, parent: G) -> None:
        # connect import edges to their calls
        for edge_from, edge_to_file, function_name in self._delayed_assignment_edges_to_add:
            if edge_to_file not in self._assignments or function_name not in self._assignments[edge_to_file]:
                continue
            edge_to = self._assignments[edge_to_file][function_name][1]
            parent.add_edge(edge_from, edge_to)
            parent.add_edge(edge_to, edge_from)
    
    def _add_delayed_call_edges(self, parent: G) -> None:
        # connect calls to their definition
        for edge_from, edge_to_file, function_name in self._delayed_call_edges_to_add:
            if edge_to_file not in self._function_definitions or function_name not in self._function_definitions[edge_to_file]:
                continue
            edge_to = self._function_definitions[edge_to_file][function_name]
            parent.add_edge(edge_from, edge_to)
            parent.add_edge(edge_to, edge_from)
    
    def _add_delayed_attribute_edges(self, parent: G) -> None:
        # connect calls to their definition
        for edge_from, edge_to_file, class_name, attribute_name in self._delayed_class_attributes_to_add:
            if edge_to_file in self._classes:
                if class_name in self._classes[edge_to_file]:
                    if attribute_name in self._classes[edge_to_file][class_name]:
                        edge_to = self._classes[edge_to_file][class_name][attribute_name]
                        parent.add_edge(edge_from, edge_to)
                        parent.add_edge(edge_to, edge_from)
                    # add edge to class definition
                    else:
                        class_definition = self._function_definitions[edge_to_file][class_name]
                        parent.add_edge(edge_from, class_definition)
                        parent.add_edge(class_definition, edge_from)

    def _second_loop(self, node_id: str, parent: G, file: str) -> None:
        current_vertex = parent.get_vertex(node_id)
        parent_vertex = parent.get_parent(node_id)

        ### REDO VARIABLE TRACKING ###
        # handle function definitions
        if current_vertex.type == 'function_definition' or current_vertex.type == 'class_definition':
            # get function name
            function_name = list(current_vertex.get_connections())[0].text
            # add function definition to dict
            if file not in self._function_definitions:
                self._function_definitions[file] = {function_name: node_id}
            else:
                self._function_definitions[file][function_name] = node_id
        ### END REDO VARIABLE TRACKING ###

        ### add assignments ###
        if current_vertex.type == 'assignment':
            identifier_node = [n for n in current_vertex.get_descendants() if n.type == 'identifier'][0]
            variable_node = list(current_vertex.get_connections())[1]
            type_ = variable_node.type
            if variable_node.type == 'call':
                variable_node = list(variable_node.get_connections())[0]
                type_ = variable_node.text
            if file not in self._assignments:
                self._assignments[file] = {}
            self._assignments[file][identifier_node.var_name] = (type_, identifier_node.id)
        ### end add assignments ###

        ### add class methods ###
        if current_vertex.type == 'class_definition':
            # add a dictionary entry for the class name
            class_name = list(current_vertex.get_connections())[0].text
            if file not in self._classes:
                self._classes[file] = {class_name: {}}
            else:
                self._classes[file][class_name] = {}

            # traverse the rest of the class definition and add all attributes
            self._class_attribute(file, class_name, parent, node_id)
        ### end add class methods ###

        ### handle other imports (constants) from other files ###
        if file in self._imports:
            if current_vertex.type == 'identifier' and not (parent_vertex.type == 'aliased_import' or parent_vertex.type == 'dotted_name'):
                possible_imports = list(self._imports[file].keys())
                import_ids = [i for _, (i, _) in self._imports[file].items()]
                paths = [p for _, (_, p) in self._imports[file].items()]
                
                # if this is an attribute call, get the parent text instead
                txt = parent.get_parent(node_id).text if parent.get_parent(node_id).type == 'attribute' else current_vertex.text

                if any([re.match(r'(^' + s + r'\.|^' + s + r'$)', txt) for s in possible_imports]):
                    func, import_id, path = [
                        (f, i, p if i.startswith('aliased_import') else p + '.' + f if p else f) for f, i, p in zip(possible_imports, import_ids, paths)
                        if txt.startswith(f)
                    ][0]

                    # if there is an identifier that matches an import, add an edge to the import
                    self._edges_to_add.append((node_id, import_id))

                    func_new = (txt if path not in txt else txt[txt.find(path)+1+len(path):]) \
                        if not import_id.startswith('aliased_import') \
                        else (path[path.rfind('.')+1:] if txt == func else txt[txt.find(func) + len(func) + 1:])
                    path_new = (path[:path.rfind(txt)-1] if len(path) - path.rfind(txt) == len(txt) else path) \
                        if not import_id.startswith('aliased_import') \
                        else (path[:path.rfind('.')] if '.' in path else path)


                    # replace any leading double period in path_new with a double period and a slash
                    if '..' in path_new:
                        path_new = re.sub(r'^\.\.', '../', path_new)
                        path_new = re.sub(r'(?<!\.)\.(?!\.)', '/', path_new)
                        path_new = os.path.normpath(os.path.join(os.path.dirname(file), path_new)) + '.py'
                        imported_from = [f for f in self._relative_files if path_new in f]

                    # replace any leading period in path_new with nothing
                    elif path_new.startswith('.') and '/' not in path_new:
                        path_new = path_new[1:]
                        path_new = re.sub(r'(?<!\.)\.(?!\.)', '/', path_new)
                        path_new = os.path.normpath(os.path.join(os.path.dirname(file), path_new)) + '.py'
                        imported_from = [f for f in self._relative_files if path_new in f]

                    else:
                        # find which file we are importing the constant from
                        imported_from = [f for f in self._relative_files if path_new.replace('.', '/') in f]
                    
                    # # find which file we are importing the constant from
                    # imported_from = [f for f in self._relative_files if path_new.replace('.', '/') in f]
                    if imported_from:
                        imported_from = imported_from[0]
                        if imported_from in self._assignments:
                            if func_new in self._assignments[imported_from]:
                                # add edge
                                self._edges_to_add.append((node_id, self._assignments[imported_from][func_new][1]))
                                self._edges_to_add.append((self._assignments[imported_from][func_new][1], node_id))
                        else:
                            self._delayed_assignment_edges_to_add.append((node_id, imported_from, func_new))
                        
                        if imported_from in self._function_definitions:
                            if func_new in self._function_definitions[imported_from]:
                                # add edge
                                self._edges_to_add.append((node_id, self._function_definitions[imported_from][func_new]))
                                self._edges_to_add.append((self._function_definitions[imported_from][func_new], node_id))
                        else:
                            self._delayed_call_edges_to_add.append((node_id, imported_from, func_new))
        ### end handle other imports (constants) from other files ###

        ### check if function is defined in the current file ###
        if file in self._function_definitions and parent_vertex and parent_vertex.type == 'call':
            func = current_vertex.text
            if func in self._function_definitions[file]:
                # add edge
                self._edges_to_add.append((node_id, self._function_definitions[file][func]))
                self._edges_to_add.append((self._function_definitions[file][func], node_id))
        ### end check if function is defined in the current file ###
        
        ### check if the function is part of an import in the current file ###
        if file in self._imports and parent_vertex and parent_vertex.type == 'call':
            possible_imports = list(self._imports[file].keys())
            import_ids = [i for _, (i, _) in self._imports[file].items()]
            paths = [p for _, (_, p) in self._imports[file].items()]

            txt = current_vertex.text
            if any([re.match(r'(^' + s + r'\.|^' + s + r'$)', txt) for s in possible_imports]):
                func, import_id, path = [
                    (f, i, p if i.startswith('aliased_import') else p + '.' + f if p else f) for f, i, p in zip(possible_imports, import_ids, paths)
                    if txt.startswith(f)
                ][0]

                func_new = (txt if path not in txt else txt[txt.find(path)+1+len(path):]) \
                    if not import_id.startswith('aliased_import') \
                    else (path[path.rfind('.')+1:] if txt == func else txt[txt.find(func) + len(func) + 1:])
                path_new = (path[:path.rfind(txt)-1] if len(path) - path.rfind(txt) == len(txt) else path) \
                    if not import_id.startswith('aliased_import') \
                    else (path[:path.rfind('.')] if '.' in path else path)
                
                # replace any leading double period in path_new with a double period and a slash
                if '..' in path_new:
                    path_new = re.sub(r'^\.\.', '../', path_new)
                    path_new = re.sub(r'(?<!\.)\.(?!\.)', '/', path_new)
                    path_new = os.path.normpath(os.path.join(os.path.dirname(file), path_new)) + '.py'
                    imported_from = [f for f in self._relative_files if path_new in f]

                # replace any leading period in path_new with nothing
                elif path_new.startswith('.') and '/' not in path_new:
                    path_new = path_new[1:]
                    path_new = re.sub(r'(?<!\.)\.(?!\.)', '/', path_new)
                    path_new = os.path.normpath(os.path.join(os.path.dirname(file), path_new)) + '.py'
                    imported_from = [f for f in self._relative_files if path_new in f]

                else:
                    # find which file we are importing the constant from
                    imported_from = [f for f in self._relative_files if path_new.replace('.', '/') in f]
                
                if imported_from:
                    imported_from = imported_from[0]
                    if imported_from in self._function_definitions:
                        if func_new in self._function_definitions[imported_from]:
                            # add edge
                            self._edges_to_add.append((node_id, self._function_definitions[imported_from][func_new]))
                            self._edges_to_add.append((self._function_definitions[imported_from][func_new], node_id))
                    else:
                        self._delayed_call_edges_to_add.append((node_id, imported_from, func_new))
        ### end check if the function is part of an import in the current file ###
        
        ### check if the call is an class attribute and find its definition ### TODO
        # handle already created objects
        grandparent = parent.get_parent(parent_vertex.id) if parent_vertex else None
        if parent_vertex and parent_vertex.type == 'attribute' and grandparent and grandparent.type == 'call':
            txt = current_vertex.text
            
            long_attribute = parent_vertex.text

            # check that the attribute matches the end of the parent string
            if long_attribute.endswith(txt) and txt:
                attribute_call = txt
                attribute_prefix = long_attribute[:long_attribute.rfind(attribute_call)-1]
                
                # check if the prefix matches an object
                if file in self._assignments:
                    if attribute_prefix in self._assignments[file]:
                        object_type = self._assignments[file][attribute_prefix][0]
                        object_node_id = self._assignments[file][attribute_prefix][1]

                        # connect locally if available
                        if file in self._classes:
                            if object_type in self._classes[file]:
                                # connect call to local attribute definition (if it exists)
                                if attribute_call in self._classes[file][object_type]:
                                    self._edges_to_add.append((node_id, self._classes[file][object_type][attribute_call]))
                                    self._edges_to_add.append((self._classes[file][object_type][attribute_call], node_id))
                                # connect call to local class definition (if it exists)
                                elif file in self._function_definitions:
                                    if object_type in self._function_definitions[file]:
                                        self._edges_to_add.append((node_id, self._function_definitions[file][object_type]))
                                        self._edges_to_add.append((self._function_definitions[file][object_type], node_id))
                                # connect call to local assignment (if it exists)
                                else:
                                    self._edges_to_add.append((node_id, object_node_id))
                                    self._edges_to_add.append((object_node_id, node_id))

                        # connect to other files if necessary
                        if file in self._imports:
                            possible_imports = list(self._imports[file].keys())
                            import_ids = [i for _, (i, _) in self._imports[file].items()]
                            paths = [p for _, (_, p) in self._imports[file].items()]
                            type_ = object_type[object_type.rfind('.')+1:]
                            object_import = object_type[:object_type.rfind('.')]

                            if object_import in possible_imports:
                                # find which file we are importing the constant from
                                imported_from = paths[possible_imports.index(object_import)] if paths[possible_imports.index(object_import)] else object_import
                                imported_from = [f for f in self._relative_files if imported_from in f]
                                if imported_from:
                                    imported_from = imported_from[0]
                                    # connect call to imported attribute definition (if it exists)
                                    if imported_from in self._classes:
                                        if type_ in self._classes[imported_from]:
                                            if attribute_call in self._classes[imported_from][type_]:
                                                self._edges_to_add.append((node_id, self._classes[imported_from][type_][attribute_call]))
                                                self._edges_to_add.append((self._classes[imported_from][type_][attribute_call], node_id))
                                            # connect to the class definition
                                            else:
                                                self._edges_to_add.append((node_id, self._function_definitions[imported_from][type_]))
                                                self._edges_to_add.append((self._function_definitions[imported_from][type_], node_id))
                                    else:
                                        self._delayed_class_attributes_to_add.append((node_id, imported_from, type_, attribute_call))


                # check if the prefix follows an import (for inline calls)
                # check the prefix to get all the possible import paths it could come from
                if file in self._imports:
                    possible_imports = list(self._imports[file].keys())
                    import_ids = [i for _, (i, _) in self._imports[file].items()]
                    paths = [p for _, (_, p) in self._imports[file].items()]
                    txt = attribute_prefix
                    while '.' in txt:
                        # first part is import
                        # second part is path and/or class call
                        # third part is the attribute
                        # match the prefix to an import where the 
                        if any([re.match(r'(^' + s + r'\.|^' + s + r'$)', txt) for s in possible_imports]):
                            func, import_id, path = [
                                (f, i, p if i.startswith('aliased_import') else p + '.' + f if p else f) for f, i, p in zip(possible_imports, import_ids, paths)
                                if txt.startswith(f)
                            ][0]
                            # remove import path from the prefix
                            txt = txt[len(func)+1:]
                            if txt:
                                txt = txt[:txt.find('(')] if '(' in txt else txt
                                # use the path to check if the file contains the class definition
                                imported_from = [f for f in self._relative_files if path_new.replace('.', '/') in f]
                                if imported_from:
                                    imported_from = imported_from[0]
                                    # connect call to imported attribute definition (if it exists)
                                    if imported_from in self._classes:
                                        if txt in self._classes[imported_from]:
                                            # connect to the class definition
                                            if attribute_call in self._classes[imported_from][txt]:
                                                self._edges_to_add.append((node_id, self._classes[imported_from][txt][attribute_call]))
                                                self._edges_to_add.append((self._classes[imported_from][txt][attribute_call], node_id))
                                            # connect to the identifier definition
                                            else:
                                                self._edges_to_add.append((node_id, self._function_definitions[imported_from][txt]))
                                                self._edges_to_add.append((self._function_definitions[imported_from][txt], node_id))
                                    # add to edges to add later
                                    else:
                                        self._delayed_class_attributes_to_add.append((node_id, imported_from, txt, attribute_call))
                            
                            break
                        txt = txt[:txt.rfind('.')]
                           
        ### end check if the call is an class attribute and find its definition ###

        ### connect identifiers to their assignments ###
        txt = current_vertex.text
        if parent_vertex and parent_vertex.type != 'assignment':
            if file in self._assignments:
                if txt in self._assignments[file]:
                    # add edge
                    self._edges_to_add.append((node_id, self._assignments[file][txt][1]))
                    # self._edges_to_add.append((self._assignments[file][txt][1], node_id))
        
        # copy all dictionaries for scoping
        if current_vertex.type in ['function_definition', 'class_definition'] or 'comprehension' in current_vertex.type or 'lambda' == current_vertex.type:
            _, fd, a, c = self._copy_for_scope()
        ### end connect identifiers to their assignments ###

        # recurse over neighbors/children
        for neighbor in current_vertex.get_connections():
            self._second_loop(neighbor.id, parent, file)

        # reset the scoping dicts
        if current_vertex.type in ['function_definition', 'class_definition'] or 'comprehension' in current_vertex.type or 'lambda' == current_vertex.type:
            self._function_definitions = fd
            self._assignments = a
            self._classes = c

    def _class_attribute(self, file: str, class_name: str, parent: G, class_root_node_id: str) -> None:
        # record all class attributes
        for child in parent.get_vertex(class_root_node_id).get_connections():
            if child.type == 'function_definition':
                # add a dictionary entry for the function name
                function_name = list(parent.get_vertex(child.id).get_connections())[0].text
                self._classes[file][class_name][function_name] = child.id

            # traverse the rest of the function definition and add all attributes
            self._class_attribute(file, class_name, parent, child.id)
                    
def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--dir", metavar = "Directory", type=str, required=True, help="Path to directory to parse")
    arg_parser.add_argument("--nf", metavar = "Node features", type = str, required = True, help = "File to save node features to")
    arg_parser.add_argument("--adj", metavar = "Adjacency matrix", type = str, required = True, help = "File to save adjacency matrix t0")
    arg_parser.add_argument("--dim", metavar = "Dimension", type = int, required = True, help = "Dimension of the node features")
    arg_parser.add_argument("--save-gv", action = "store_true", help = "Flag to save Graphviz file format of graph")
    arg_parser.add_argument('--neighbors', metavar = "Neighbors", type = int, help = "Number of neighbors to show for a specific node")
    arg_parser.add_argument('--node', metavar = "Node", type = str, help = "Node to start neighbor search at")
    args = arg_parser.parse_args()

    if args.neighbors and not args.node:
        arg_parser.error("--neighbors requires --node")

    ast = ASTCodebaseParser(args.dir, args.dim)
    ast.parse_dir()
    ast.to_csv(args.nf, args.adj)
    ast.csv_features_to_vectors(args.nf)
    
    if args.save_gv:
        ast.convert_to_graphviz()

    if args.neighbors:
        ast.view_k_neighbors(args.node, args.neighbors)

if __name__ == "__main__":
    main()
