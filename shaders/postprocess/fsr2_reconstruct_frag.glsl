#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_accumulated;
uniform vec2  u_output_size;
uniform float u_rcas_sharpness;
uniform float u_exposure;
uniform float u_saturation;
uniform float u_gamma;
uniform int   u_flip_y;

float luminance(vec3 c) {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

vec3 rcas(vec2 uv, vec2 rcp) {
    vec3 e  = texture(u_accumulated, uv).rgb;
    vec3 n  = texture(u_accumulated, uv + vec2( 0.0,  1.0) * rcp).rgb;
    vec3 s  = texture(u_accumulated, uv + vec2( 0.0, -1.0) * rcp).rgb;
    vec3 w  = texture(u_accumulated, uv + vec2(-1.0,  0.0) * rcp).rgb;
    vec3 ee = texture(u_accumulated, uv + vec2( 1.0,  0.0) * rcp).rgb;

    float le = luminance(e);
    float ln = luminance(n);
    float ls = luminance(s);
    float lw = luminance(w);
    float la = luminance(ee);

    float mn4 = min(le, min(min(ln, ls), min(lw, la)));
    float mx4 = max(le, max(max(ln, ls), max(lw, la)));
    float contrast = mx4 - mn4;

    float rcpL = 1.0 / (4.0 * le + 1e-5);
    float ampN = clamp(min(mn4, 2.0 - mx4) * rcpL, 0.0, 1.0);
    ampN = sqrt(ampN);

    float w_rcas = -ampN * clamp(u_rcas_sharpness, 0.0, 1.0) * 0.25;
    if (contrast < 1.0 / 255.0) w_rcas = 0.0;

    float denom = 1.0 + 4.0 * w_rcas;
    denom = max(abs(denom), 1e-5) * sign(denom + 1e-9);

    vec3 col = (e + (n + s + w + ee) * w_rcas) / denom;
    col = max(col, vec3(0.0));

    col *= u_exposure;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation);
    return pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1)));
}

void main() {
    vec2 uv  = (u_flip_y != 0) ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    vec2 rcp = 1.0 / u_output_size;
    fragColor = vec4(rcas(uv, rcp), 1.0);
}