#version 450

// Flat, unlit color for line geometry (wireframe box / model segments).

layout(location = 0) in vec4 inColor;

layout(location = 0) out vec4 outColor;

void main()
{
    outColor = inColor;
}
