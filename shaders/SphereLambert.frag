#version 450

// Basic Lambert shading for the instanced spheres: a single directional
// light (pc.lightDir, pointing from the surface toward the light — the
// renderer passes a "headlight" direction derived from the camera each
// frame) plus a constant ambient floor so back faces never go fully
// black.

layout(location = 0) in vec3 inNormal;
layout(location = 1) in vec4 inColor;

layout(push_constant) uniform SceneConstants
{
    mat4 mvp;
    vec4 lightDir;
} pc;

layout(location = 0) out vec4 outColor;

void main()
{
    float diffuse = max(dot(normalize(inNormal), normalize(pc.lightDir.xyz)), 0.0);
    float lit = 0.25 + 0.75 * diffuse;
    outColor = vec4(inColor.rgb * lit, inColor.a);
}
