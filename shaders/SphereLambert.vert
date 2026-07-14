#version 450

// Instanced unit-sphere vertex shader for the hardware Scene3D renderer
// (Hal::Gpu::Scene3DRenderContext). Binding 0 carries the shared unit
// sphere mesh (one position per vertex — for a unit sphere centered at
// the origin the normal IS the position, so no separate normal
// attribute); binding 1 advances per instance with each queued sphere's
// center/radius and color.
//
// The push-constant block is shared with Line3D.vert (same pipeline
// layout): mvp at offset 0, lightDir at offset 64 (used by
// SphereLambert.frag).

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inCenterRadius;
layout(location = 2) in vec4 inColor;

layout(push_constant) uniform SceneConstants
{
    mat4 mvp;
    vec4 lightDir;
} pc;

layout(location = 0) out vec3 outNormal;
layout(location = 1) out vec4 outColor;

void main()
{
    vec3 worldPos = inCenterRadius.xyz + inPos * inCenterRadius.w;
    gl_Position = pc.mvp * vec4(worldPos, 1.0);
    outNormal = inPos;
    outColor = inColor;
}
