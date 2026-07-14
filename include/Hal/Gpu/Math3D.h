#pragma once

#include <cstdint>

namespace Hal
{
    namespace Gpu
    {
        // Minimal 3D math for the hardware Scene3D renderer — a
        // hand-rolled alternative to pulling in GLM, keeping small math
        // helpers in-tree. Only what the render path actually needs: a
        // float 3-vector and a column-major 4x4 float matrix with the
        // perspective/view constructors the model-view-projection uniform
        // is built from.
        //
        // Mat4 is column-major with column-vector convention
        // (out = M * v), matching GLSL's default layout — the matrix bytes
        // can be memcpy'd straight into a push-constant block.
        struct Vec3
        {
            float x;
            float y;
            float z;
        };

        struct Mat4
        {
            // m[column * 4 + row], column-major as GLSL expects.
            float m[16];
        };

        Vec3 Vec3Sub(const Vec3 &a,
                     const Vec3 &b);

        Vec3 Vec3Cross(const Vec3 &a,
                       const Vec3 &b);

        float Vec3Dot(const Vec3 &a,
                      const Vec3 &b);

        // Returns (0, 0, 0) for a zero-length input rather than dividing
        // by zero.
        Vec3 Vec3Normalize(const Vec3 &v);

        Mat4 Mat4Identity();

        // out = a * b (apply b first, then a, under the column-vector
        // convention).
        Mat4 Mat4Multiply(const Mat4 &a,
                          const Mat4 &b);

        // Right-handed look-at view matrix (eye toward target, up
        // approximately worldUp), the standard gluLookAt construction.
        Mat4 Mat4LookAt(const Vec3 &eye,
                        const Vec3 &target,
                        const Vec3 &worldUp);

        // Right-handed perspective projection mapping to Vulkan clip
        // space: depth in [0, 1] and y pointing DOWN in NDC (the Vulkan
        // convention, opposite OpenGL's) — callers do not need to flip y
        // themselves. fovY is the full vertical field of view in radians.
        Mat4 Mat4Perspective(float fovY,
                             float aspect,
                             float nearPlane,
                             float farPlane);

        // out = M * (v, 1), divided by the resulting w (a full
        // perspective transform of a point). Used by tests to check
        // matrices against known projections; the render path itself does
        // this on the GPU.
        Vec3 Mat4TransformPoint(const Mat4 &m,
                                const Vec3 &v);
    }
}
