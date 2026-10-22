#version 450

// One triangle covering the viewport, from gl_VertexIndex alone (no vertex
// buffer). outUV runs 0..2; clipping leaves the visible 0..1 rectangle.

layout(location = 0) out vec2 outUV;

void main()
{
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
