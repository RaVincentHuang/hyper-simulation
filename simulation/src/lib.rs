pub mod graph;
pub mod utils;

use pyo3::prelude::*;


#[pymodule]
fn simulation(m: &Bound<'_, PyModule>) -> PyResult<()> {
    graph::register_graph_module(m)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::get_simulation_inter, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::get_simulation_inter_fn, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic_fn, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic_of_edge_fn, m)?)?;
    m.add_function(wrap_pyfunction!(graph::networkx_graph::is_simulation_isomorphic_of_node_edge_fn, m)?)?;
    Ok(())
}
