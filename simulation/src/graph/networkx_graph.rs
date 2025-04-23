use graph_simulation::algorithm::simulation::Simulation;
use graph_simulation::graph::base::Graph;
use pyo3::types::PySet;
use pyo3::{prelude::*, types::PyDict};
use graph_simulation::graph::labeled_graph::{Label, Labeled, LabeledAdjacency};
use graph_simulation::graph::base::{Directed, Adjacency, AdjacencyInv, SingleId, IdPair};


use pyo3::prelude::*;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Display;
use std::sync::Arc;

type SharedRustFn = Arc<dyn Fn(&Attributes, &Attributes) -> bool + Send + Sync>;


// 自定义图结构

#[derive(Debug)]
struct Attributes(HashMap<String, Py<PyAny>>);

impl Clone for Attributes {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            let cloned_map = self.0.iter()
                .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                .collect();
            Attributes(cloned_map)
        })
    }
}

impl std::fmt::Display for Attributes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));

        write!(f, "{{")?;
        for (key, value) in entries {
            write!(f, "{}: {}, ", key, value)?;
        }
        write!(f, "}}")
    }
}

impl PartialEq for Attributes {
    fn eq(&self, other: &Self) -> bool {
        // 首先比较长度
        if self.0.len() != other.0.len() {
            return false;
        }

        Python::with_gil(|py| {
            self.0.iter().all(|(key, value)| {
                match other.0.get(key) {
                    Some(other_value) => {
                        match value.call_method1(py, "__eq__", (other_value,)) {
                            Ok(result) => result.extract::<bool>(py).unwrap_or(false),
                            Err(_) => false,
                        }
                    },
                    None => false,
                }
            })
        })
    }
}

impl Eq for Attributes {}

impl Hash for Attributes {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // 确保相同的字典产生相同的哈希值
        let mut entries: Vec<_> = self.0.iter().collect();
        entries.sort_by(|(k1, _), (k2, _)| k1.cmp(k2));

        Python::with_gil(|py| {
            for (key, value) in entries {
                key.hash(state);
                // 使用 Python 对象的 __hash__ 方法
                match value.call_method0(py, "__hash__") {
                    Ok(hash_result) => {
                        if let Ok(hash_value) = hash_result.extract::<isize>(py) {
                            hash_value.hash(state);
                        }
                    },
                    Err(_) => {
                        // 如果对象不可哈希，使用默认值
                        0isize.hash(state);
                    }
                }
            }
        });
    }
}

impl Label for Attributes {
    fn label(&self) -> &str {
        ""
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Node {
    id: usize,
    attributes: Attributes,
}

impl std::fmt::Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Node({})", self.id)
    }
}

impl SingleId for Node {
    fn id(&self) -> usize {
        self.id
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct Edge {
    source: usize,
    target: usize,
    attributes: Attributes,
}

impl IdPair for Edge {
    fn pair(&self) -> (usize, usize) {
        (self.source, self.target)
    }
}

#[pyclass]
pub struct NetworkXGraph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    node_indices: HashMap<String, usize>,
    same_label_fn: Option<Py<PyAny>>,
    same_edge_fn: Option<Py<PyAny>>,
    same_node_edge_fn: Option<Py<PyAny>>,
    same_label_cache: Option<HashSet<(usize, usize)>>
}

impl Clone for NetworkXGraph {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            NetworkXGraph {
                nodes: self.nodes.clone(),
                edges: self.edges.clone(),
                node_indices: self.node_indices.clone(),
                same_label_fn: self.same_label_fn.as_ref().map(|f| f.clone_ref(py)),
                same_edge_fn: self.same_edge_fn.as_ref().map(|f| f.clone_ref(py)),
                same_node_edge_fn: self.same_node_edge_fn.as_ref().map(|f| f.clone_ref(py)),
                same_label_cache: self.same_label_cache.clone(),
            }
        })
    }
}

fn convert_to_string(obj: &PyObject) -> PyResult<String> {
    Python::with_gil(|py| {
        // Try direct conversion first
        obj.call_method0(py, "__str__")?.extract::<String>(py)
            .or_else(|_| {
                // If that fails, try to convert to a string using repr
                obj.call_method0(py, "__repr__")?.extract::<String>(py)
            })
    })
}

#[pymethods]
impl NetworkXGraph {
    #[new]
    fn new() -> Self {
        NetworkXGraph {
            nodes: Vec::new(),
            edges: Vec::new(),
            node_indices: HashMap::new(),
            same_label_fn: None,
            same_edge_fn: None,
            same_node_edge_fn: None, 
            same_label_cache: None,
        }
    }

    // 从NetworkX图转换的静态方法
    #[staticmethod]
    fn from_networkx(nx_graph: &Bound<'_, PyAny>) -> PyResult<Self> {
        let nodes = nx_graph.getattr("nodes")?.call_method1("items", ())?;
        let edges = nx_graph.getattr("edges")?.call_method1("data", ())?;

        let mut graph = NetworkXGraph::new();

        for node in nodes.try_iter()? {
            let node = node?;
            let id = node.call_method1("__getitem__", (0, ))?.extract::<PyObject>()?;
            let id = convert_to_string(&id)?;
            let attrs = node.call_method1("__getitem__", (1, ))?.extract::<HashMap<String, PyObject>>()?;
            graph.add_node(id, attrs);
        }
        for edge in edges.try_iter()? {
            let edge = edge?;
            let source = edge.call_method1("__getitem__", (0, ))?.extract::<PyObject>()?;
            let source = convert_to_string(&source)?;
            let target = edge.call_method1("__getitem__", (1, ))?.extract::<PyObject>()?;
            let target = convert_to_string(&target)?;
            let attrs = edge.call_method1("__getitem__", (2, ))?.extract::<HashMap<String, PyObject>>()?;
            graph.add_edge(source, target, attrs);
        }
        
        Ok(graph)
    }

    // 转回NetworkX图的方法
    fn to_networkx(&self, py: Python) -> PyResult<PyObject> {
        let nx = py.import("networkx")?;
        let graph = nx.getattr("Graph")?.call0()?;

        // 添加节点
        for node in &self.nodes {
            let attrs_dict = PyDict::new(py);
            for (k, v) in &node.attributes.0 {
                attrs_dict.set_item(k, v.clone_ref(py))?;
            }
            graph.call_method1(
                "add_node",
                (node.id.clone(), attrs_dict),
            )?;
        }

        // 添加边
        for edge in &self.edges {
            let attrs_dict = PyDict::new(py);
            for (k, v) in &edge.attributes.0 {
                attrs_dict.set_item(k, v.clone_ref(py))?;
            }
            graph.call_method1(
                "add_edge",
                (
                    edge.source.clone(),
                    edge.target.clone(),
                    attrs_dict,
                ),
            )?;
        }

        Ok(graph.into())
    }

    // 其他有用的方法
    fn add_node(&mut self, id: String, attributes: HashMap<String, PyObject>) {
        let index = self.nodes.len();
        self.node_indices.insert(id.clone(), index);
        let attributes = Attributes(attributes);
        self.nodes.push(Node { id: index, attributes });
    }

    fn add_edge(
        &mut self,
        source: String,
        target: String,
        attributes: HashMap<String, PyObject>,
    ) {
        let attributes = Attributes(attributes);
        let source = *self.node_indices.get(&source).unwrap();
        let target = *self.node_indices.get(&target).unwrap();
        self.edges.push(Edge {
            source,
            target,
            attributes,
        });
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    // 获取节点属性
    fn get_node_attributes(&self, node_id: &str) -> Option<HashMap<String, PyObject>> {
        self.node_indices.get(node_id).map(|&index| {
            Python::with_gil(|py| {
                self.nodes[index].attributes.0.iter()
                    .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                    .collect()
            })
        })
    }

    // 获取边属性
    fn get_edge_attributes(
        &self,
        source: &str,
        target: &str,
    ) -> Option<HashMap<String, PyObject>> {
        self.edges
            .iter()
            .find(|e| e.source == *self.node_indices.get(source).unwrap() 
                            && e.target == *self.node_indices.get(target).unwrap())
            .map(|e| Python::with_gil(|py| {
                e.attributes.0.iter()
                    .map(|(k, v)| (k.clone(), v.clone_ref(py)))
                    .collect()
            }))
    }

    fn register_compare_fn(&mut self, compare: Py<PyAny>) {
        self.same_label_fn = Some(compare);
    }

    fn register_edge_compare_fn(&mut self, compare: Py<PyAny>) {
        self.same_edge_fn = Some(compare);
    }

    fn register_node_edge_compare_fn(&mut self, compare: Py<PyAny>) {
        self.same_node_edge_fn = Some(compare);
    }

    fn build_compare_cache(&mut self, other: &NetworkXGraph) {
        // use rayon::prelude::*;
        // let cache: HashSet<_> = self.nodes.par_iter().flat_map(|node1| {
        //     let local: HashSet<_> = other.nodes.par_iter().filter_map(|node2| {
        //         if self.label_same(node1, node2) {
        //             Some((node1.id, node2.id))
        //         } else {
        //             None
        //         }
        //     }).collect();
        //     local
        // }).collect();

        let cache: HashSet<_> = self.nodes.iter().flat_map(|node1| {
            other.nodes.iter().filter_map(|node2| {
                if self.label_same(node1, node2) {
                    Some((node1.id, node2.id))
                } else {
                    None
                }
            }).collect::<HashSet<_>>()
        }).collect();

        self.same_label_cache = Some(cache);
    }
}

impl<'a> Graph<'a> for NetworkXGraph {
    type Node = Node;

    type Edge = Edge;

    fn nodes(&'a self) -> impl Iterator<Item = &'a Self::Node> {
        self.nodes.iter()
    }

    fn edges(&'a self) -> impl Iterator<Item = &'a Self::Edge> {
        self.edges.iter()
    }

    fn get_edges_pair(&'a self) -> impl Iterator<Item = (&'a Self::Node, &'a Self::Node)> {
        let id_map: HashMap<_, _, std::collections::hash_map::RandomState> = HashMap::from_iter(self.nodes.iter().map(|node| (node.id, node)));
        self.edges.iter().map(|edge| (id_map.get(&edge.source).unwrap().clone(), id_map.get(&edge.target).unwrap().clone()) ).collect::<Vec<_>>().into_iter()
    }

    fn add_node(&mut self, node: Self::Node) {
        let index = self.nodes.len();
        self.node_indices.insert(format!("Node{}.", index), index);
        self.nodes.push(node);
    }

    fn add_edge(&mut self, edge: Self::Edge) {
        self.edges.push(edge);
    }
}

fn test_eq(a: &Py<PyAny>, b: &Py<PyAny>) -> bool {
    Python::with_gil(|py| {
        match a.call_method1(py, "__eq__", (b,)) {
            Ok(result) => result.extract::<bool>(py).unwrap_or(false),
            Err(_) => false
        }
    })
}

fn native_same_label_fn(a: &Attributes, b: &Attributes) -> bool {
    for (k, v) in &a.0 {
        if let Some(other_v) = b.0.get(k) {
            if test_eq(v, other_v) {
                continue;
            }
        } else {
            return false;
        }
    }
    true
}

impl<'a> Labeled<'a> for NetworkXGraph {
    fn label_same(&self, node: &Self::Node, label: &Self::Node) -> bool {

        if let Some(cache) = self.same_label_cache.as_ref() {
            return cache.contains(&(node.id, label.id));
        }

        if let Some(compare_fn) = self.same_label_fn.as_ref() {
            Python::with_gil(|py| {
                let attr1 = node.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let attr2 = label.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                compare_fn.call1(py,(attr1, attr2)).unwrap().extract::<bool>(py).unwrap()
            })
        } else {
            return native_same_label_fn(&node.attributes, &label.attributes);
        }
    }

    fn get_label(&'a self, node: &'a Self::Node) -> &'a impl Label {
        &node.attributes
    }

    fn get_edges_pair_label(&'a self) -> impl Iterator<Item = (&'a Self::Node, &'a Self::Node, &'a impl Label)> {
        let id_map: HashMap<_, _, std::collections::hash_map::RandomState> = HashMap::from_iter(self.nodes.iter().map(|node| (node.id, node)));
        self.edges.iter().map(move |edge| (id_map.get(&edge.source).unwrap().clone(), id_map.get(&edge.target).unwrap().clone(), &edge.attributes)).collect::<Vec<_>>().into_iter()
    }

    fn edge_label_same(&self, edge1: &Self::Edge, edge2: &Self::Edge) -> bool {
        if let Some(compare_fn) = self.same_edge_fn.as_ref() {
            Python::with_gil(|py| {
                let attr1 = edge1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let attr2 = edge2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                compare_fn.call1(py,(attr1, attr2)).unwrap().extract::<bool>(py).unwrap()
            })
        } else {
            native_same_label_fn(&edge1.attributes, &edge2.attributes)
        }
    }

    fn edge_node_label_same(&self, src1: &Self::Node, edge1: &Self::Edge, dst1: &Self::Node, src2: &Self::Node, edge2: &Self::Edge, dst2: &Self::Node) -> bool {
        if let Some(compare_fn) = self.same_node_edge_fn.as_ref() {
            Python::with_gil(|py| {
                let src1_attr = src1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let dst1_attr = dst1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let edge1_attr = edge1.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let src2_attr = src2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let dst2_attr = dst2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                let edge2_attr = edge2.attributes.0.iter().map(|(k, v)| (k.clone(), v.clone_ref(py))).collect::<HashMap<_, _>>();
                compare_fn.call1(py,(src1_attr, edge1_attr, dst1_attr, src2_attr, edge2_attr, dst2_attr)).unwrap().extract::<bool>(py).unwrap()
            })
        } else {
            native_same_label_fn(&src1.attributes, &src2.attributes) 
                && native_same_label_fn(&dst1.attributes, &dst2.attributes) 
                && native_same_label_fn(&edge1.attributes, &edge2.attributes)
        }
    }
}

impl LabeledAdjacency<'_> for NetworkXGraph {}

impl std::fmt::Display for NetworkXGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NetworkXGraph(\n")?;
        write!(f, "Nodes: [\n")?;
        for node in &self.nodes {
            write!(f, "  {},\n", node)?;
        }
        write!(f, "],\n")?;
        write!(f, "Edges: [\n")?;
        for edge in &self.edges {
            write!(f, "{} -> {}, ", edge.source, edge.target)?;
        }
        write!(f, "]\n")?;
        write!(f, ")")
    }
}

fn to_nx_node(py: Python, node: &Node) -> PyResult<PyObject> {
    let attrs_dict = PyDict::new(py);
    for (k, v) in &node.attributes.0 {
        attrs_dict.set_item(k, v.clone_ref(py))?;
    }
    let nx = py.import("networkx")?;
    let node = nx.getattr("Node")?.call1((node.id.clone(), attrs_dict))?;
    Ok(node.into())
}

#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, is_label_cached = false))]
pub fn get_simulation_inter(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, is_label_cached: bool) -> PyResult<PyObject> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    let sim = graph1.get_simulation_inter(&graph2);
    

    // Convert simulation to a list of pairs (i, j) where i is a node in graph1, j is a node in graph2
    Python::with_gil(|py| {
        let map = PyDict::new(py);
        
        for (node, set) in sim.iter() {
            let py_set = PySet::new(py, set.iter().map(|node| to_nx_node(py, node)).collect::<PyResult<Vec<_>>>()?)?;
            map.set_item(to_nx_node(py, node)?, py_set)?;
        }
    
        Ok(map.into())
    })
}

#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, is_label_cached = false))]
pub fn is_simulation_isomorphic(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_inter(&graph2)))
}

#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, compare, is_label_cached = false))]
pub fn get_simulation_inter_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, compare: Py<PyAny>, is_label_cached: bool) -> PyResult<PyObject> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;

    graph1.register_compare_fn(compare);
    
    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    let sim = graph1.get_simulation_inter(&graph2);

    Python::with_gil(|py| {
        let map = PyDict::new(py);
        
        for (node, set) in sim.iter() {
            let py_set = PySet::new(py, set.iter().map(|node| to_nx_node(py, node)).collect::<PyResult<Vec<_>>>()?)?;
            map.set_item(to_nx_node(py, node)?, py_set)?;
        }
    
        Ok(map.into())
    })
}

#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, compare, is_label_cached = false))]
pub fn is_simulation_isomorphic_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, compare: Py<PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;
    
    graph1.register_compare_fn(compare);

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_inter(&graph2)))
}

#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, node_compare, edge_compare, is_label_cached = false))]
pub fn is_simulation_isomorphic_of_node_edge_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, node_compare: Py<PyAny>, edge_compare: Py<PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;
    
    graph1.register_compare_fn(node_compare);
    graph1.register_edge_compare_fn(edge_compare);

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_of_node_edge(&graph2)))
}

#[pyfunction]
#[pyo3(signature = (nx_graph1, nx_graph2, node_edge_compare, is_label_cached = false))]
pub fn is_simulation_isomorphic_of_edge_fn(nx_graph1: &Bound<'_, PyAny>, nx_graph2: &Bound<'_, PyAny>, node_edge_compare: Py<PyAny>, is_label_cached: bool) -> PyResult<bool> {
    let mut graph1 = NetworkXGraph::from_networkx(nx_graph1)?;
    let graph2 = NetworkXGraph::from_networkx(nx_graph2)?;
    
    graph1.register_node_edge_compare_fn(node_edge_compare);

    if is_label_cached {
        graph1.build_compare_cache(&graph2);
    }

    Ok(NetworkXGraph::has_simulation(graph1.get_simulation_of_edge(&graph2)))
}

impl Directed for NetworkXGraph {}

impl Adjacency<'_> for NetworkXGraph {}

impl AdjacencyInv<'_> for NetworkXGraph {}

// 模块定义
// #[pymodule]
// pub fn networkx_graph(_py: Python, m: &PyModule) -> PyResult<()> {
//     m.add_class::<NetworkXGraph>()?;
//     Ok(())
// }
