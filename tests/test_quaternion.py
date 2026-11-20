import numpy as np

def quaternion_to_threejs(q):
    """Convert [w, x, y, z] to Three.js (x, y, z, w) ordering."""
    if len(q) != 4:
        raise ValueError('Quaternion must have 4 components')
    w, x, y, z = q
    return [x, y, z, w]


def normalize(q):
    norm = np.linalg.norm(q)
    if norm == 0:
        raise ValueError('Zero-length quaternion cannot be normalized')
    return q / norm

def rotate_vector(q, v):
    """Rotate vector v by quaternion q (w,x,y,z)."""
    q = normalize(q)
    w, x, y, z = q
    # quaternion multiplication: q * [0, v] * q_conj
    vq = np.array([0.0, *v])
    q_conj = np.array([w, -x, -y, -z])
    # helper
    def mul(a, b):
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return np.array([
            aw*bw - ax*bx - ay*by - az*bz,
            aw*bx + ax*bw + ay*bz - az*by,
            aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw,
        ])
    return mul(mul(q, vq), q_conj)[1:]

def test_identity_quaternion():
    q = [1.0, 0.0, 0.0, 0.0]
    v = [1, 2, 3]
    assert np.allclose(rotate_vector(q, v), v)

def test_90deg_z_rotation():
    # 90 deg around Z axis => quaternion [cos45, 0,0,sin45]
    angle = np.pi/2
    q = [np.cos(angle/2), 0, 0, np.sin(angle/2)]
    v = [1, 0, 0]
    rot = rotate_vector(q, v)
    expected = [0, 1, 0]
    assert np.allclose(rot, expected, atol=1e-7)

def test_known_vector_rotation():
    # rotate (1,0,0) by 180 deg around Y => should be (-1,0,0)
    angle = np.pi
    q = [np.cos(angle/2), 0, np.sin(angle/2), 0]
    v = [1, 0, 0]
    rot = rotate_vector(q, v)
    expected = [-1, 0, 0]
    assert np.allclose(rot, expected, atol=1e-7)

def test_normalization():
    q = [2.0, 0.0, 0.0, 0.0]
    normed = normalize(q)
    assert np.isclose(np.linalg.norm(normed), 1.0)
    # conversion should keep order
    assert quaternion_to_threejs(normed) == [0.0, 0.0, 0.0, 1.0]

def test_invalid_quaternion():
    q = [0.0, 0.0, 0.0, 0.0]
    try:
        normalize(q)
    except ValueError:
        assert True
    else:
        assert False
