#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_current;
uniform sampler2D u_history;
uniform sampler2D u_depth;
uniform sampler2D u_prev_depth;
uniform vec2  u_render_size;
uniform vec2  u_jitter;
uniform float u_blend_alpha;
uniform int   u_first_frame;
uniform float u_depth_threshold;
uniform float u_reactive_scale;

vec3 ycocgFromRgb(vec3 c) {
    float co = c.r - c.b;
    float t  = c.b + co * 0.5;
    float cg = c.g - t;
    return vec3(t + cg * 0.5, co, cg);
}

vec3 rgbFromYcocg(vec3 c) {
    float t = c.x - c.z * 0.5;
    float g = c.z + t;
    float b = t - c.y * 0.5;
    return vec3(b + c.y, g, b);
}

float lumaYcocg(vec3 c) {
    return c.x;
}

vec3 tonemapW(vec3 c) {
    return c / (1.0 + c.x);
}

vec3 tonemapWInv(vec3 c) {
    return c / max(1.0 - c.x, 1e-5);
}

vec3 clipAabb(vec3 lo, vec3 hi, vec3 h) {
    vec3 center  = 0.5 * (hi + lo);
    vec3 extents = 0.5 * (hi - lo) + 1e-4;
    vec3 d       = h - center;
    vec3 ts      = abs(extents / (d + 1e-8));
    float t      = min(min(ts.x, ts.y), ts.z);
    return center + d * min(t, 1.0);
}

void main() {
    vec2 rcpRender = 1.0 / u_render_size;

    vec2 uvCur = clamp(v_uv - u_jitter, vec2(0.0), vec2(1.0));

    vec3 curRgb = texture(u_current, uvCur).rgb;

    if (u_first_frame != 0) {
        fragColor = vec4(curRgb, 1.0);
        return;
    }

    vec3 c = tonemapW(ycocgFromRgb(curRgb));
    vec3 n = tonemapW(ycocgFromRgb(texture(u_current, uvCur + vec2( 0.0,  1.0) * rcpRender).rgb));
    vec3 s = tonemapW(ycocgFromRgb(texture(u_current, uvCur + vec2( 0.0, -1.0) * rcpRender).rgb));
    vec3 e = tonemapW(ycocgFromRgb(texture(u_current, uvCur + vec2( 1.0,  0.0) * rcpRender).rgb));
    vec3 w = tonemapW(ycocgFromRgb(texture(u_current, uvCur + vec2(-1.0,  0.0) * rcpRender).rgb));

    vec3 aabbMin = min(c, min(min(n, s), min(e, w)));
    vec3 aabbMax = max(c, max(max(n, s), max(e, w)));

    vec3 mean = (c + n + s + e + w) * 0.2;
    vec3 dev  = vec3(0.0);
    dev += (c - mean) * (c - mean);
    dev += (n - mean) * (n - mean);
    dev += (s - mean) * (s - mean);
    dev += (e - mean) * (e - mean);
    dev += (w - mean) * (w - mean);
    dev = sqrt(dev * 0.2);

    aabbMin = max(aabbMin, mean - 1.25 * dev);
    aabbMax = min(aabbMax, mean + 1.25 * dev);

    float curDepth  = texture(u_depth,      uvCur).r;
    float prevDepth = texture(u_prev_depth, v_uv).r;
    float disocclusion = step(u_depth_threshold, abs(curDepth - prevDepth));

    vec3 histRgb = texture(u_history, v_uv).rgb;
    vec3 hist    = tonemapW(ycocgFromRgb(histRgb));
    hist         = clipAabb(aabbMin, aabbMax, hist);

    float lumDiff  = abs(lumaYcocg(c) - lumaYcocg(hist)) / (max(lumaYcocg(c), lumaYcocg(hist)) + 0.01);
    float reactive = clamp(lumDiff * u_reactive_scale, 0.0, 0.9);

    float alpha = clamp(u_blend_alpha + disocclusion * (1.0 - u_blend_alpha) + reactive * 0.3,
                        u_blend_alpha, 1.0);

    vec3 blended   = mix(hist, c, alpha);
    vec3 resultRgb = max(rgbFromYcocg(tonemapWInv(blended)), vec3(0.0));

    fragColor = vec4(resultRgb, 1.0);
}