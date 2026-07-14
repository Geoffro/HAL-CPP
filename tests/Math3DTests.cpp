#include <doctest/doctest.h>

#include "Hal/Gpu/Math3D.h"

#include <cmath>

// Ground-truth checks for the hand-rolled 3D math the hardware Scene3D
// renderer builds its model-view-projection push constant from. These
// run headlessly in every configuration (no Vulkan device or window
// needed).

TEST_CASE("Hal::Gpu vector helpers: cross/dot/normalize behave to definition")
{
    Hal::Gpu::Vec3 xAxis = {1.0f, 0.0f, 0.0f};
    Hal::Gpu::Vec3 yAxis = {0.0f, 1.0f, 0.0f};

    Hal::Gpu::Vec3 cross = Hal::Gpu::Vec3Cross(xAxis, yAxis);
    CHECK(cross.x == doctest::Approx(0.0f));
    CHECK(cross.y == doctest::Approx(0.0f));
    CHECK(cross.z == doctest::Approx(1.0f));

    CHECK(Hal::Gpu::Vec3Dot(xAxis, yAxis) == doctest::Approx(0.0f));
    CHECK(Hal::Gpu::Vec3Dot(xAxis, xAxis) == doctest::Approx(1.0f));

    Hal::Gpu::Vec3 v = {3.0f, 0.0f, 4.0f};
    Hal::Gpu::Vec3 unit = Hal::Gpu::Vec3Normalize(v);
    CHECK(std::sqrt(Hal::Gpu::Vec3Dot(unit, unit)) == doctest::Approx(1.0f));
    CHECK(unit.x == doctest::Approx(0.6f));
    CHECK(unit.z == doctest::Approx(0.8f));

    // Zero-length input returns zero rather than dividing by zero.
    Hal::Gpu::Vec3 zero = Hal::Gpu::Vec3Normalize(Hal::Gpu::Vec3{0.0f, 0.0f, 0.0f});
    CHECK(zero.x == 0.0f);
    CHECK(zero.y == 0.0f);
    CHECK(zero.z == 0.0f);
}

TEST_CASE("Hal::Gpu::Mat4: identity and multiplication")
{
    Hal::Gpu::Mat4 identity = Hal::Gpu::Mat4Identity();
    Hal::Gpu::Vec3 point = {1.5f, -2.0f, 3.25f};

    Hal::Gpu::Vec3 same = Hal::Gpu::Mat4TransformPoint(identity, point);
    CHECK(same.x == doctest::Approx(point.x));
    CHECK(same.y == doctest::Approx(point.y));
    CHECK(same.z == doctest::Approx(point.z));

    // M * I == I * M == M, checked through the transform of a point.
    Hal::Gpu::Mat4 lookAt = Hal::Gpu::Mat4LookAt(Hal::Gpu::Vec3{4.0f, 3.0f, 7.0f},
                                                 Hal::Gpu::Vec3{0.0f, 0.0f, 0.0f},
                                                 Hal::Gpu::Vec3{0.0f, 1.0f, 0.0f});
    Hal::Gpu::Mat4 leftProduct = Hal::Gpu::Mat4Multiply(identity, lookAt);
    Hal::Gpu::Mat4 rightProduct = Hal::Gpu::Mat4Multiply(lookAt, identity);
    Hal::Gpu::Vec3 viaLookAt = Hal::Gpu::Mat4TransformPoint(lookAt, point);
    Hal::Gpu::Vec3 viaLeft = Hal::Gpu::Mat4TransformPoint(leftProduct, point);
    Hal::Gpu::Vec3 viaRight = Hal::Gpu::Mat4TransformPoint(rightProduct, point);
    CHECK(viaLeft.x == doctest::Approx(viaLookAt.x));
    CHECK(viaLeft.y == doctest::Approx(viaLookAt.y));
    CHECK(viaLeft.z == doctest::Approx(viaLookAt.z));
    CHECK(viaRight.x == doctest::Approx(viaLookAt.x));
    CHECK(viaRight.y == doctest::Approx(viaLookAt.y));
    CHECK(viaRight.z == doctest::Approx(viaLookAt.z));
}

TEST_CASE("Hal::Gpu::Mat4LookAt: eye maps to the view-space origin, the target to the -z axis")
{
    Hal::Gpu::Vec3 eye = {10.0f, 5.0f, -3.0f};
    Hal::Gpu::Vec3 target = {2.0f, 1.0f, 4.0f};
    Hal::Gpu::Mat4 view = Hal::Gpu::Mat4LookAt(eye, target, Hal::Gpu::Vec3{0.0f, 1.0f, 0.0f});

    Hal::Gpu::Vec3 eyeInView = Hal::Gpu::Mat4TransformPoint(view, eye);
    CHECK(eyeInView.x == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(eyeInView.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(eyeInView.z == doctest::Approx(0.0f).epsilon(1e-4));

    // The target lies straight ahead: on the -z view axis (right-handed
    // camera), at exactly the eye-target distance.
    Hal::Gpu::Vec3 offset = Hal::Gpu::Vec3Sub(target, eye);
    float          distance = std::sqrt(Hal::Gpu::Vec3Dot(offset, offset));
    Hal::Gpu::Vec3 targetInView = Hal::Gpu::Mat4TransformPoint(view, target);
    CHECK(targetInView.x == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(targetInView.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(targetInView.z == doctest::Approx(-distance).epsilon(1e-4));
}

TEST_CASE("Hal::Gpu::Mat4Perspective: Vulkan clip-space conventions (z in [0,1], y down)")
{
    const float fovY = 1.0f;
    const float aspect = 4.0f / 3.0f;
    const float nearPlane = 0.5f;
    const float farPlane = 100.0f;
    Hal::Gpu::Mat4 projection = Hal::Gpu::Mat4Perspective(fovY, aspect, nearPlane, farPlane);

    // A point on the optical axis stays centered, and the near/far
    // planes map to NDC depth 0 and 1 (not OpenGL's -1..1).
    Hal::Gpu::Vec3 onNear = Hal::Gpu::Mat4TransformPoint(projection, Hal::Gpu::Vec3{0.0f, 0.0f, -nearPlane});
    CHECK(onNear.x == doctest::Approx(0.0f));
    CHECK(onNear.y == doctest::Approx(0.0f));
    CHECK(onNear.z == doctest::Approx(0.0f).epsilon(1e-4));

    Hal::Gpu::Vec3 onFar = Hal::Gpu::Mat4TransformPoint(projection, Hal::Gpu::Vec3{0.0f, 0.0f, -farPlane});
    CHECK(onFar.z == doctest::Approx(1.0f).epsilon(1e-4));

    // Depth increases monotonically between the planes.
    Hal::Gpu::Vec3 midway = Hal::Gpu::Mat4TransformPoint(projection, Hal::Gpu::Vec3{0.0f, 0.0f, -10.0f});
    CHECK(midway.z > onNear.z);
    CHECK(midway.z < onFar.z);

    // Vulkan NDC y points down, so a point ABOVE the axis (+y in view
    // space) lands at negative NDC y — the matrix bakes the flip in.
    Hal::Gpu::Vec3 above = Hal::Gpu::Mat4TransformPoint(projection, Hal::Gpu::Vec3{0.0f, 1.0f, -10.0f});
    CHECK(above.y < 0.0f);

    // And a point to the right (+x) stays at positive NDC x.
    Hal::Gpu::Vec3 right = Hal::Gpu::Mat4TransformPoint(projection, Hal::Gpu::Vec3{1.0f, 0.0f, -10.0f});
    CHECK(right.x > 0.0f);
}

TEST_CASE("Hal::Gpu::Mat4: a full orbit-camera view-projection frames the look-at point at the NDC center")
{
    // The same construction Scene3DRenderContext uses per frame: orbit
    // camera at distance d from a domain center, perspective projection,
    // combined as projection * view.
    Hal::Gpu::Vec3 center = {10.0f, 5.0f, 5.0f};
    Hal::Gpu::Vec3 eye = {center.x - 20.0f, center.y + 12.0f, center.z - 15.0f};
    Hal::Gpu::Mat4 view = Hal::Gpu::Mat4LookAt(eye, center, Hal::Gpu::Vec3{0.0f, 1.0f, 0.0f});
    Hal::Gpu::Mat4 projection = Hal::Gpu::Mat4Perspective(1.0f, 1.5f, 0.1f, 200.0f);
    Hal::Gpu::Mat4 viewProjection = Hal::Gpu::Mat4Multiply(projection, view);

    Hal::Gpu::Vec3 centerNdc = Hal::Gpu::Mat4TransformPoint(viewProjection, center);
    CHECK(centerNdc.x == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(centerNdc.y == doctest::Approx(0.0f).epsilon(1e-4));
    CHECK(centerNdc.z > 0.0f);
    CHECK(centerNdc.z < 1.0f);

    // A point higher in the world (+y) than the look-at center appears
    // above it on screen (smaller NDC y, since NDC y points down).
    Hal::Gpu::Vec3 aboveCenter = {center.x, center.y + 2.0f, center.z};
    Hal::Gpu::Vec3 aboveNdc = Hal::Gpu::Mat4TransformPoint(viewProjection, aboveCenter);
    CHECK(aboveNdc.y < centerNdc.y);
}
