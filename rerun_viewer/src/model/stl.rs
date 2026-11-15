//! The binary STL files that a robot's description holds its collision meshes in: an
//! 80-byte header, the number of triangles (u32), then for each triangle its normal and
//! its three vertices (f32 triples) and a 2-byte attribute, little-endian.

use crate::model::scene::ShapeKind;

const HEADER_LEN: usize = 80;
const COUNT_LEN: usize = 4;
const TRIANGLE_LEN: usize = 50;
/// The normal comes first in a triangle's record; the viewer does not read it.
const NORMAL_LEN: usize = 12;
const VERTEX_LEN: usize = 12;

/// Why bytes are not a binary STL that the viewer draws.
#[derive(Debug, thiserror::Error, PartialEq)]
pub enum StlError {
    #[error("{0} bytes, fewer than the 84 of a binary STL's header and triangle count")]
    Short(usize),
    #[error("an ASCII STL, which the viewer does not read: give a binary STL")]
    Ascii,
    #[error("{triangles} triangles make {expected} bytes, the file has {got}")]
    Size {
        triangles: u64,
        expected: u64,
        got: usize,
    },
    #[error("{0} triangles have more vertices than a mesh's u32 indices count")]
    TooManyTriangles(u64),
    #[error("triangle {0} has a coordinate that is not a finite number")]
    NotFinite(usize),
}

/// The mesh of the binary STL `bytes`, each vertex scaled by `scale` along x, y and z:
/// three vertices for each triangle, in the file's order. A scale that mirrors the mesh
/// (an odd number of negative factors) turns the order of each triangle's vertices
/// around, so that its front still faces out of the solid.
pub fn parse_stl(bytes: &[u8], scale: [f64; 3]) -> Result<ShapeKind, StlError> {
    let Some(count) = bytes.get(HEADER_LEN..HEADER_LEN + COUNT_LEN) else {
        return Err(StlError::Short(bytes.len()));
    };
    let triangle_count = u64::from(u32::from_le_bytes(
        count.try_into().expect("a slice of 4 bytes"),
    ));
    let expected = (HEADER_LEN + COUNT_LEN) as u64 + triangle_count * TRIANGLE_LEN as u64;
    if bytes.len() as u64 != expected {
        if bytes.starts_with(b"solid") {
            return Err(StlError::Ascii);
        }
        return Err(StlError::Size {
            triangles: triangle_count,
            expected,
            got: bytes.len(),
        });
    }
    if u32::try_from(3 * triangle_count).is_err() {
        return Err(StlError::TooManyTriangles(triangle_count));
    }
    let mirrored = scale.iter().filter(|factor| **factor < 0.0).count() % 2 == 1;
    let (records, _) = bytes[HEADER_LEN + COUNT_LEN..].as_chunks::<TRIANGLE_LEN>();
    let mut vertices = Vec::with_capacity(3 * records.len());
    let mut triangles = Vec::with_capacity(records.len());
    for (index, record) in records.iter().enumerate() {
        let corners: [[f64; 3]; 3] = std::array::from_fn(|corner| {
            let start = NORMAL_LEN + corner * VERTEX_LEN;
            std::array::from_fn(|axis| {
                let at = start + 4 * axis;
                let value = f32::from_le_bytes(record[at..at + 4].try_into().expect("4 bytes"));
                f64::from(value) * scale[axis]
            })
        });
        if corners.iter().flatten().any(|c| !c.is_finite()) {
            return Err(StlError::NotFinite(index));
        }
        // Below 3 * triangle_count, which fits a u32.
        let first = vertices.len() as u32;
        vertices.extend(corners);
        triangles.push(if mirrored {
            [first, first + 2, first + 1]
        } else {
            [first, first + 1, first + 2]
        });
    }
    Ok(ShapeKind::Mesh {
        vertices,
        triangles,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::stl_bytes;

    const TWO_TRIANGLES: [[[f32; 3]; 3]; 2] = [
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
    ];

    #[test]
    fn each_triangle_gives_its_three_vertices_scaled() {
        let mesh = parse_stl(&stl_bytes(&TWO_TRIANGLES), [0.5, 2.0, 1.0]).expect("an STL");
        assert_eq!(
            mesh,
            ShapeKind::Mesh {
                vertices: vec![
                    [0.0, 0.0, 0.0],
                    [0.5, 0.0, 0.0],
                    [0.0, 2.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.5, 0.0, 1.0],
                    [0.0, 2.0, 1.0],
                ],
                triangles: vec![[0, 1, 2], [3, 4, 5]],
            }
        );
    }

    #[test]
    fn a_mirroring_scale_turns_each_triangle_around() {
        let ShapeKind::Mesh {
            vertices,
            triangles,
        } = parse_stl(&stl_bytes(&TWO_TRIANGLES[..1]), [1.0, -1.0, 1.0]).unwrap()
        else {
            panic!("a mesh")
        };
        assert_eq!(
            vertices,
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]
        );
        assert_eq!(triangles, [[0, 2, 1]]);
        // Two negative factors mirror twice, which is a turn: the order stays.
        let twice = parse_stl(&stl_bytes(&TWO_TRIANGLES[..1]), [-1.0, -1.0, 1.0]).unwrap();
        assert!(matches!(twice, ShapeKind::Mesh { triangles, .. } if triangles == [[0, 1, 2]]));
    }

    #[test]
    fn an_empty_mesh_has_no_triangles() {
        assert_eq!(
            parse_stl(&stl_bytes(&[]), [1.0; 3]),
            Ok(ShapeKind::Mesh {
                vertices: vec![],
                triangles: vec![],
            })
        );
    }

    #[test]
    fn bytes_that_are_no_binary_stl_are_refused() {
        assert_eq!(parse_stl(&[0; 10], [1.0; 3]), Err(StlError::Short(10)));
        let mut cut = stl_bytes(&TWO_TRIANGLES);
        cut.truncate(cut.len() - 1);
        assert_eq!(
            parse_stl(&cut, [1.0; 3]),
            Err(StlError::Size {
                triangles: 2,
                expected: 184,
                got: 183,
            })
        );
        let mut ascii = b"solid hand\nfacet normal 0 0 1\n".to_vec();
        ascii.resize(120, b' ');
        assert_eq!(parse_stl(&ascii, [1.0; 3]), Err(StlError::Ascii));
    }

    #[test]
    fn a_coordinate_that_is_not_finite_is_refused() {
        let nan = [[f32::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        assert_eq!(
            parse_stl(&stl_bytes(&[TWO_TRIANGLES[0], nan]), [1.0; 3]),
            Err(StlError::NotFinite(1))
        );
        assert_eq!(
            parse_stl(&stl_bytes(&TWO_TRIANGLES), [f64::INFINITY, 1.0, 1.0]),
            Err(StlError::NotFinite(0))
        );
    }
}
