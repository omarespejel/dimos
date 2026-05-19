// Local definitions of DimOS-internal LCM messages that aren't in the
// dimos-lcm rust-codegen branch yet (Graph3D, GraphDelta3D).
//
// Wire format mirrors the Python encoders in dimos/msgs/nav_msgs/*.py exactly:
// big-endian, no LCM hash header (these are dimos-custom types routed by
// channel-name suffix, not by LCM packed-fingerprint).
//
// Only encode is implemented — pgo_rust publishes, never subscribes.

#[derive(Debug, Clone)]
pub struct PoseStamped {
    pub ts: f64,
    pub frame_id: String,
    pub position: [f64; 3],
    pub orientation: [f64; 4],  // xyzw
}

#[derive(Debug, Clone)]
pub struct Node3D {
    pub pose: PoseStamped,
    pub id: u64,
    pub metadata_id: u64,
}

#[derive(Debug, Clone)]
pub struct Edge {
    pub start_id: u64,
    pub end_id: u64,
    pub timestamp: f64,
    pub metadata_id: u64,
}

#[derive(Debug, Clone, Default)]
pub struct Graph3D {
    pub ts: f64,
    pub nodes: Vec<Node3D>,
    pub edges: Vec<Edge>,
}

impl Graph3D {
    pub fn encode(graph: &Self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(graph.estimated_size());
        buf.extend_from_slice(&(graph.edges.len() as u64).to_be_bytes());
        buf.extend_from_slice(&(graph.nodes.len() as u64).to_be_bytes());
        buf.extend_from_slice(&graph.ts.to_be_bytes());
        for node in &graph.nodes {
            encode_node(node, &mut buf);
        }
        for edge in &graph.edges {
            buf.extend_from_slice(&edge.start_id.to_be_bytes());
            buf.extend_from_slice(&edge.end_id.to_be_bytes());
            buf.extend_from_slice(&edge.timestamp.to_be_bytes());
            buf.extend_from_slice(&edge.metadata_id.to_be_bytes());
        }
        buf
    }

    fn estimated_size(&self) -> usize {
        // 24B header + per-node (8+4+frame_id+56+16) + per-edge 32B
        let nodes = self.nodes.iter().map(|n| 84 + n.pose.frame_id.len()).sum::<usize>();
        24 + nodes + self.edges.len() * 32
    }
}

#[derive(Debug, Clone)]
pub struct Transform {
    pub translation: [f64; 3],
    pub rotation: [f64; 4],  // xyzw
}

#[derive(Debug, Clone, Default)]
pub struct GraphDelta3D {
    pub ts: f64,
    pub nodes: Vec<Node3D>,
    pub transforms: Vec<Transform>,
}

impl GraphDelta3D {
    pub fn encode(delta: &Self) -> Vec<u8> {
        assert_eq!(
            delta.nodes.len(),
            delta.transforms.len(),
            "GraphDelta3D nodes and transforms must have equal length",
        );
        let mut buf = Vec::with_capacity(delta.estimated_size());
        buf.extend_from_slice(&(delta.nodes.len() as u64).to_be_bytes());
        buf.extend_from_slice(&delta.ts.to_be_bytes());
        for node in &delta.nodes {
            encode_node(node, &mut buf);
        }
        for transform in &delta.transforms {
            for value in transform.translation.iter().chain(transform.rotation.iter()) {
                buf.extend_from_slice(&value.to_be_bytes());
            }
        }
        buf
    }

    fn estimated_size(&self) -> usize {
        // 16B header + per-node + per-transform 56B
        let nodes = self.nodes.iter().map(|n| 84 + n.pose.frame_id.len()).sum::<usize>();
        16 + nodes + self.transforms.len() * 56
    }
}

fn encode_node(node: &Node3D, buf: &mut Vec<u8>) {
    let frame_id_bytes = node.pose.frame_id.as_bytes();
    buf.extend_from_slice(&node.pose.ts.to_be_bytes());
    buf.extend_from_slice(&(frame_id_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(frame_id_bytes);
    for value in node.pose.position.iter().chain(node.pose.orientation.iter()) {
        buf.extend_from_slice(&value.to_be_bytes());
    }
    buf.extend_from_slice(&node.id.to_be_bytes());
    buf.extend_from_slice(&node.metadata_id.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph3d_empty_header() {
        let graph = Graph3D { ts: 1.5, nodes: vec![], edges: vec![] };
        let bytes = Graph3D::encode(&graph);
        assert_eq!(bytes.len(), 24);
        // edge_count=0, node_count=0, ts=1.5
        assert_eq!(&bytes[0..8], &0u64.to_be_bytes());
        assert_eq!(&bytes[8..16], &0u64.to_be_bytes());
        assert_eq!(&bytes[16..24], &1.5f64.to_be_bytes());
    }

    #[test]
    fn graph3d_one_node_one_edge() {
        let node = Node3D {
            pose: PoseStamped {
                ts: 2.0,
                frame_id: "map".to_string(),
                position: [1.0, 2.0, 3.0],
                orientation: [0.0, 0.0, 0.0, 1.0],
            },
            id: 42,
            metadata_id: 1,
        };
        let edge = Edge { start_id: 0, end_id: 42, timestamp: 3.0, metadata_id: 0 };
        let graph = Graph3D { ts: 1.0, nodes: vec![node], edges: vec![edge] };
        let bytes = Graph3D::encode(&graph);
        // 24 header + (8 pose_ts + 4 fr_len + 3 "map" + 56 pose_xyz + 16 ids) + 32 edge
        assert_eq!(bytes.len(), 24 + (8 + 4 + 3 + 56 + 16) + 32);
        // edge_count=1, node_count=1
        assert_eq!(&bytes[0..8], &1u64.to_be_bytes());
        assert_eq!(&bytes[8..16], &1u64.to_be_bytes());
    }

    #[test]
    fn graph_delta_aligned() {
        let node = Node3D {
            pose: PoseStamped {
                ts: 0.0, frame_id: String::new(),
                position: [0.0; 3], orientation: [0.0, 0.0, 0.0, 1.0],
            },
            id: 0, metadata_id: 0,
        };
        let transform = Transform { translation: [0.1, 0.2, 0.3], rotation: [0.0, 0.0, 0.0, 1.0] };
        let delta = GraphDelta3D {
            ts: 0.5,
            nodes: vec![node],
            transforms: vec![transform],
        };
        let bytes = GraphDelta3D::encode(&delta);
        // 16 header + (8 + 4 + 0 + 56 + 16) node + 56 transform
        assert_eq!(bytes.len(), 16 + (8 + 4 + 0 + 56 + 16) + 56);
    }

    #[test]
    #[should_panic(expected = "must have equal length")]
    fn graph_delta_mismatched_panics() {
        let delta = GraphDelta3D {
            ts: 0.0,
            nodes: vec![Node3D {
                pose: PoseStamped {
                    ts: 0.0, frame_id: String::new(),
                    position: [0.0; 3], orientation: [0.0, 0.0, 0.0, 1.0],
                },
                id: 0, metadata_id: 0,
            }],
            transforms: vec![],
        };
        let _ = GraphDelta3D::encode(&delta);
    }
}
