#version 450

// Line-geometry vertex shader for the hardware Scene3D renderer: one
// world-space position + color per vertex, drawn as a line list (the
// domain wireframe box and any model line segments). Shares
// SphereLambert.vert's push-constant layout so both pipelines use one
// pipeline layout; lines are unlit, so only mvp is read here.

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
