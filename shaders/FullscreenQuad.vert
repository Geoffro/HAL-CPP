#version 450

// Generates a single triangle covering the whole viewport from
// gl_VertexIndex alone (0, 1, 2) — the standard "fullscreen triangle"
// trick, avoiding a vertex/index buffer entirely for what's just a
// textured rectangle. outUV runs 0..2 on each axis; the swapchain's
// scissor/rasterizer clips everything outside the visible 0..1 portion,
// which is exactly the on-screen rectangle.

layout(location = 0) out vec2 outUV;

void main()
{
    outUV = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    gl_Position = vec4(outUV * 2.0 - 1.0, 0.0, 1.0);
}
