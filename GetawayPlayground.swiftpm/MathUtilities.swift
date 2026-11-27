import simd

extension simd_float4x4 {
    /// Right-handed perspective projection with reversed depth (near = 1, far = 0).
    static func perspectiveReversedZ(fovY: Float, aspect: Float, near: Float, far: Float) -> simd_float4x4 {
        let ys = 1 / tan(fovY * 0.5)
        let xs = ys / aspect
        let a = near / (far - near)
        let b = near * far / (far - near)
        return simd_float4x4(columns: (
            SIMD4(xs, 0, 0, 0),
            SIMD4(0, ys, 0, 0),
            SIMD4(0, 0, a, -1),
            SIMD4(0, 0, b, 0)
        ))
    }

    static func lookAt(eye: SIMD3<Float>, forward: SIMD3<Float>, up: SIMD3<Float>) -> simd_float4x4 {
        let f = normalize(forward)
        let s = normalize(cross(f, up))
        let u = cross(s, f)
        return simd_float4x4(columns: (
            SIMD4(s.x, u.x, -f.x, 0),
            SIMD4(s.y, u.y, -f.y, 0),
            SIMD4(s.z, u.z, -f.z, 0),
            SIMD4(-dot(s, eye), -dot(u, eye), dot(f, eye), 1)
        ))
    }

    private func row(_ i: Int) -> SIMD4<Float> {
        SIMD4(columns.0[i], columns.1[i], columns.2[i], columns.3[i])
    }

    /// Left, right, bottom and top clip planes with inward-facing normalized normals.
    var frustumSidePlanes: [SIMD4<Float>] {
        let r0 = row(0), r1 = row(1), r3 = row(3)
        return [r3 + r0, r3 - r0, r3 + r1, r3 - r1].map { plane in
            plane / length(SIMD3(plane.x, plane.y, plane.z))
        }
    }
}
