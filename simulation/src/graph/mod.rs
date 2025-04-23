
pub mod networkx_graph;

use pyo3::prelude::*;

#[pymodule]
pub fn register_graph_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let child_module = PyModule::new(parent_module.py(), "graph")?;
    child_module.add_class::<networkx_graph::NetworkXGraph>()?;
    child_module.add_function(wrap_pyfunction!(networkx_graph::get_simulation_inter, &child_module)?)?;
    child_module.add_function(wrap_pyfunction!(networkx_graph::is_simulation_isomorphic, &child_module)?)?;
    parent_module.add_submodule(&child_module)
}
