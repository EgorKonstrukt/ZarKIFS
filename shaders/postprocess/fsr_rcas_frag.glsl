#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform vec2  u_resolution;
uniform float u_gamma;
uniform float u_exposure;
uniform float u_saturation;
uniform int   u_flip_y;
uniform float u_rcas_sharpness;

void main() {
    vec2 uv  = (u_flip_y != 0) ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    vec2 rcp = 1.0 / u_resolution;

    vec3 e  = texture(u_scene, uv).rgb;
    vec3 n  = texture(u_scene, uv + vec2( 0.0,  1.0) * rcp).rgb;
    vec3 s  = texture(u_scene, uv + vec2( 0.0, -1.0) * rcp).rgb;
    vec3 ww = texture(u_scene, uv + vec2(-1.0,  0.0) * rcp).rgb;
    vec3 ee = texture(u_scene, uv + vec2( 1.0,  0.0) * rcp).rgb;

    vec3 lumW = vec3(0.299, 0.587, 0.114);
    float le = dot(e,  lumW);
    float ln = dot(n,  lumW);
    float ls = dot(s,  lumW);
    float lw = dot(ww, lumW);
    float la = dot(ee, lumW);

    float mn4 = min(le, min(min(ln, ls), min(lw, la)));
    float mx4 = max(le, max(max(ln, ls), max(lw, la)));
    float contrast = mx4 - mn4;

    float sharpAmt = -u_rcas_sharpness * 0.125;
    if (contrast < 1.0 / 255.0) sharpAmt = 0.0;

    float denom = 1.0 + 4.0 * sharpAmt;
    denom = max(denom, 1e-5);
    vec3 col = clamp((e + (n + s + ww + ee) * sharpAmt) / denom, vec3(0.0), vec3(1.0));

    col *= u_exposure;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1))), 1.0);
}