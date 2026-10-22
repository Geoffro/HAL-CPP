#version 450

// Instanced unit sphere for Scene3DRenderContext. Binding 0 is the shared
// mesh (position doubles as normal); binding 1 advances per instance with
// center/radius and color. Push constants are shared with Line3D.vert:
// mvp at offset 0, lightDir at 64.

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
