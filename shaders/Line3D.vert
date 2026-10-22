#version 450

// Unlit line-list geometry for Scene3DRenderContext; shares
// SphereLambert.vert's push constants but reads only mvp.

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec4 inColor;

layout(push_constant) uniform SceneConstants
{
    mat4 mvp;
    vec4 lightDir;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    gl_Position = pc.mvp * vec4(inPos, 1.0);
    outColor = inColor;
}
