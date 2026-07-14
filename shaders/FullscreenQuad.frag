#version 450

// Samples the live pixel-grid texture (updated per frame by
// WindowRenderContext::UpdateTexture, from e.g. Hal::RenderGrid2DToRgba)
// and writes it straight to the fullscreen triangle from
// FullscreenQuad.vert — this is Phase 6's live `UIGrid`-equivalent
// render surface (see docs/PHASED_PORT.md).

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(binding = 0) uniform sampler2D gridTexture;

void main()
{
    outColor = texture(gridTexture, inUV);
}
