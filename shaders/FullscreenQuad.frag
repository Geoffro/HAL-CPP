#version 450

// Samples the pixel-grid texture (updated by
// WindowRenderContext::UpdateTexture) across the fullscreen triangle from
// FullscreenQuad.vert.

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D gridTexture;

void main()
{
    outColor = texture(gridTexture, inUV);
}
