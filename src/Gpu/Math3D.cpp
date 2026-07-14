#include "Hal/Gpu/Math3D.h"

#include <cmath>

namespace Hal
{
    namespace Gpu
    {
        Vec3 Vec3Sub(const Vec3 &a,
                     const Vec3 &b)
        {
            Vec3 out = {a.x - b.x, a.y - b.y, a.z - b.z};
            return out;
        }

        Vec3 Vec3Cross(const Vec3 &a,
                       const Vec3 &b)
        {
            Vec3 out = {a.y * b.z - a.z * b.y,
                        a.z * b.x - a.x * b.z,
                        a.x * b.y - a.y * b.x};
            return out;
        }

        float Vec3Dot(const Vec3 &a,
                      const Vec3 &b)
        {
            return (a.x * b.x) + (a.y * b.y) + (a.z * b.z);
        }

        Vec3 Vec3Normalize(const Vec3 &v)
        {
            float lengthSquared = Vec3Dot(v, v);
            if (lengthSquared <= 0.0f)
            {
                Vec3 zero = {0.0f, 0.0f, 0.0f};
                return zero;
            }
            float invLength = 1.0f / std::sqrt(lengthSquared);
            Vec3  out = {v.x * invLength, v.y * invLength, v.z * invLength};
            return out;
        }

        Mat4 Mat4Identity()
        {
            Mat4 out = {};
            out.m[0] = 1.0f;
            out.m[5] = 1.0f;
            out.m[10] = 1.0f;
            out.m[15] = 1.0f;
            return out;
        }

        Mat4 Mat4Multiply(const Mat4 &a,
                          const Mat4 &b)
        {
            Mat4 out = {};
            for (int32_t col = 0; col < 4; col++)
            {
                for (int32_t row = 0; row < 4; row++)
                {
                    float sum = 0.0f;
                    for (int32_t k = 0; k < 4; k++)
                    {
                        sum += a.m[k * 4 + row] * b.m[col * 4 + k];
                    }
                    out.m[col * 4 + row] = sum;
                }
            }
            return out;
        }

        Mat4 Mat4LookAt(const Vec3 &eye,
                        const Vec3 &target,
                        const Vec3 &worldUp)
        {
            Vec3 forward = Vec3Normalize(Vec3Sub(target, eye));
            Vec3 right = Vec3Normalize(Vec3Cross(forward, worldUp));
            Vec3 up = Vec3Cross(right, forward);

            // Rows are the camera basis vectors; the translation column
            // moves the eye to the origin (standard gluLookAt layout,
            // right-handed: camera looks down -z in view space).
            Mat4 out = {};
            out.m[0] = right.x;
            out.m[4] = right.y;
            out.m[8] = right.z;
            out.m[1] = up.x;
            out.m[5] = up.y;
            out.m[9] = up.z;
            out.m[2] = -forward.x;
            out.m[6] = -forward.y;
            out.m[10] = -forward.z;
            out.m[12] = -Vec3Dot(right, eye);
            out.m[13] = -Vec3Dot(up, eye);
            out.m[14] = Vec3Dot(forward, eye);
            out.m[15] = 1.0f;
            return out;
        }

        Mat4 Mat4Perspective(float fovY,
                             float aspect,
                             float nearPlane,
                             float farPlane)
        {
            float tanHalfFovY = std::tan(fovY / 2.0f);

            // OpenGL-style right-handed perspective, adjusted for Vulkan
            // clip space: y is negated (Vulkan NDC y points down) and z
            // maps to [0, 1] rather than [-1, 1].
            Mat4 out = {};
            out.m[0] = 1.0f / (aspect * tanHalfFovY);
            out.m[5] = -1.0f / tanHalfFovY;
            out.m[10] = farPlane / (nearPlane - farPlane);
            out.m[11] = -1.0f;
            out.m[14] = (farPlane * nearPlane) / (nearPlane - farPlane);
            return out;
        }

        Vec3 Mat4TransformPoint(const Mat4 &m,
                                const Vec3 &v)
        {
            float x = (m.m[0] * v.x) + (m.m[4] * v.y) + (m.m[8] * v.z) + m.m[12];
            float y = (m.m[1] * v.x) + (m.m[5] * v.y) + (m.m[9] * v.z) + m.m[13];
            float z = (m.m[2] * v.x) + (m.m[6] * v.y) + (m.m[10] * v.z) + m.m[14];
            float w = (m.m[3] * v.x) + (m.m[7] * v.y) + (m.m[11] * v.z) + m.m[15];
            if (w != 0.0f)
            {
                x /= w;
                y /= w;
                z /= w;
            }
            Vec3 out = {x, y, z};
            return out;
        }
    }
}
