#version 450

// Soft additive glow (Scene3DRenderContext::SphereGlow): brightest where
// the surface faces the camera, fading to nothing at the silhouette, drawn
// additively without depth writes so overlapping glows accumulate in any
// order. pc.lightDir (the headlight) serves as the view direction; the
// instance color's alpha is the intensity.

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
