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

const vec3 LUM_W = vec3(0.2126, 0.7152, 0.0722);
float luminance(vec3 c) { return dot(c, LUM_W); }
vec3 tonemapW(vec3 c)    { float l = luminance(c); return c / (1.0 + l); }
vec3 tonemapWInv(vec3 c) { float l = luminance(c); return c / max(1.0 - l, 1e-5); }

vec3 rcas(vec2 uv, vec2 rcp) {
    vec3 eRaw  = texture(u_accumulated, uv).rgb;
    vec3 nRaw  = texture(u_accumulated, uv + vec2( 0.0,  1.0) * rcp).rgb;
    vec3 sRaw  = texture(u_accumulated, uv + vec2( 0.0, -1.0) * rcp).rgb;
    vec3 wRaw  = texture(u_accumulated, uv + vec2(-1.0,  0.0) * rcp).rgb;
    vec3 eeRaw = texture(u_accumulated, uv + vec2( 1.0,  0.0) * rcp).rgb;

    vec3 e  = tonemapW(eRaw);
    vec3 n  = tonemapW(nRaw);
    vec3 s  = tonemapW(sRaw);
    vec3 w  = tonemapW(wRaw);
    vec3 ee = tonemapW(eeRaw);

    float le = luminance(e);
    float ln = luminance(n);
    float ls = luminance(s);
    float lw = luminance(w);
    float la = luminance(ee);

    float mn4 = min(le, min(min(ln, ls), min(lw, la)));
    float mx4 = max(le, max(max(ln, ls), max(lw, la)));
    float contrast = mx4 - mn4;

    if (contrast < 1.5 / 255.0) {
        vec3 col = eRaw * u_exposure;
        float lum = luminance(col);
        col = mix(vec3(lum), col, u_saturation);
        return pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1)));
    }

    float rcpSum = 1.0 / (mn4 + mx4 + 1e-5);
    float ampN   = clamp(mn4 * 2.0 * rcpSum, 0.0, 1.0);
    ampN = sqrt(ampN);

    float sharpness = clamp(u_rcas_sharpness, 0.0, 1.0);
    float edgeScale = 1.0 - smoothstep(0.3, 0.7, contrast);
    float w_rcas    = -ampN * sharpness * edgeScale * 0.2;

    float rcpDenom = 1.0 / max(1.0 + 4.0 * w_rcas, 1e-5);
    vec3 col = (e + (n + s + w + ee) * w_rcas) * rcpDenom;
    col = max(col, vec3(0.0));

    float maxNeigh = max(le, max(max(ln, ls), max(lw, la)));
    float lumSharp = luminance(col);
    if (lumSharp > maxNeigh * 1.05) {
        col *= (maxNeigh * 1.05) / max(lumSharp, 1e-5);
    }

    col = tonemapWInv(col);
    col *= u_exposure;
    float lum = luminance(col);
    col = mix(vec3(lum), col, u_saturation);
    return pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1)));
}

void main() {
    vec2 uv  = (u_flip_y != 0) ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    vec2 rcp = 1.0 / u_output_size;
    fragColor = vec4(rcas(uv, rcp), 1.0);
}
