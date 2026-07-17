#version 450

// Soft additive glow for the translucent sphere pipeline
// (Scene3DRenderContext::SphereGlow): instead of Lambert shading, the
// fragment fades from full brightness where the surface faces the camera
// to nothing at the silhouette, so an instance reads as a soft round puff
// of light with no hard edge. Drawn additively with depth writes off —
// overlapping puffs accumulate like fog or fire glow, in any draw order.
// pc.lightDir is the renderer's headlight direction (domain center toward
// the camera), which serves as the view direction here; the instance
// color's alpha carries the glow intensity.

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
    float facing = max(dot(normalize(inNormal), normalize(pc.lightDir.xyz)), 0.0);
    float soft = facing * facing;
    outColor = vec4(inColor.rgb * soft * inColor.a, 1.0);
}
