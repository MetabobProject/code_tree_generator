import gc
import typing as th


class Node:

    __slots__ = [
        '_id',
        '_start',
        '_end',
        '_file',
        '_text',
        '_type',
        '_var_name',
        '_adjacent',
        '_parent',
    ]

    def __init__(self, 
                 id: str,
                 start: str,
                 end: str,
                 file: str,
                 text: th.Optional[str] = None,
                 type: th.Optional[str] = None,
                 var_name: th.Optional[str] = None,
                 parent: th.Optional['Node'] = None) -> None:
        self._id = id
        self._start = start
        self._end = end
        self._file = file
        self._text = text
        self._type = type
        self._var_name = var_name
        self._adjacent : th.Dict[Node, int] = {}
        self._parent = parent

    @property
    def id(self) -> str:
        return self._id
    
    @id.setter
    def id(self, value: str) -> None:
        raise Exception("id is read-only.")

    @property
    def file(self) -> str:
        return self._file

    @file.setter
    def file(self, value: str) -> None:
        raise Exception("file is read-only.")

    @property
    def text(self) -> str:
        return self._text if self._text else ""

    @text.setter
    def text(self, value: str) -> None:
        self._text = value

    @property
    def type(self) -> str:
        return self._type if self._type else ""
    
    @type.setter
    def type(self, value: str) -> None:
        self._type = value

    @property
    def var_name(self) -> str:
        return self._var_name if self._var_name else ""

    @var_name.setter
    def var_name(self, value: str) -> None:
        self._var_name = value
    
    @property
    def parent(self) -> th.Optional['Node']:
        return self._parent

    @parent.setter
    def parent(self, value: th.Optional['Node']) -> None:
        if value is not None and not isinstance(value, Node):
            raise Exception("parent must be a Node or None for root nodes.")
        self._parent = value

    def __str__(self) -> str:
        return str(self.id) + ' adjacent: ' + str([x.id for x in self._adjacent])

    def add_neighbor(self, neighbor: "Node", weight : float = 1.) -> None:
        self._adjacent[neighbor] = weight

    def get_connections(self) -> th.List["Node"]:
        return self._adjacent.keys()

    def get_weight(self, neighbor: "Node") -> float:
        return self._adjacent[neighbor]
    
    def get_descendants(self) -> th.List["Node"]:
        descendants : th.List["Node"] = []
        for neighbor in self.get_connections():
            descendants.append(neighbor)
            descendants.extend(neighbor.get_descendants())
        return descendants

class Graph:

    __slots__ = [
        'vert_dict',
        'num_vertices',
    ]

    def __init__(self) -> None:
        self.vert_dict : th.Dict[str: Node]= {}
        self.num_vertices : int = 0
    
    def __iter__(self) -> th.Iterator[Node]:
        return iter(self.vert_dict.values())
    
    def __str__(self) -> str:
        return '----------\n' + \
            '\n-\n'.join(str(node) for node in iter(self)) + \
            '\n----------'

    def add_vertex(self, node: Node) -> Node:
        # check that if there is a parent it is in the graph
        if node.parent:
            if node.parent.id not in self.vert_dict:
                raise Exception(f"Parent {node.parent.id} not in graph.")
        self.num_vertices = self.num_vertices + 1
        self.vert_dict[node.id] = node

        return node.id

    def get_vertex(self, id: str) -> Node:
        if id in self.vert_dict:
            return self.vert_dict[id]
        else:
            return None
        
    def add_edge(self, from_: str, to_: str, weight: float = 1, bi: bool = False) -> None:
        if from_ not in self.vert_dict:
            raise Exception(f"Vertex {from_} not in graph.")
        if to_ not in self.vert_dict:
            raise Exception(f"Vertex {to_} not in graph.")
        self.vert_dict[from_].add_neighbor(self.vert_dict[to_], weight)
        if bi:
            self.vert_dict[to_].add_neighbor(self.vert_dict[from_], weight)
    
    def get_vertices(self) -> th.List[str]:
        return list(self.vert_dict.keys())
    
    def get_parent(self, id: str) -> Node:
        return self.vert_dict[id].parent

    def get_highest_attribute(self, id: str) -> Node:
        # find the highest parent of the current node that has a type of attribute
        node: Node = self.vert_dict[id]
        
        if node.parent:
            while node.parent:
                if node.parent.type == 'attribute':
                    node = node.parent
                else:
                    break
            return node
        else:
            return None
        
    def delete_graph(self) -> None:
        for node_id, node in self.vert_dict.items():
            node.parent = None
            node._adjacent.clear()
            del node
        self.vert_dict.clear()
        self.num_vertices = 0
        gc.collect()


if __name__ == "__main__":
    print(Node('a', 'b', 'c'))
    print(Node('a', 'b', 'c').id)
    n = Node('a', 'b', 'c')
    n1 = Node('a', 'b', 'd', parent=n)
    g = Graph()
    g.add_vertex(n)
    g.add_vertex(n1)
    print(g)
