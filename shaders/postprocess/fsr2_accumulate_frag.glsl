#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_current;
uniform sampler2D u_history;
uniform sampler2D u_depth;
uniform sampler2D u_prev_depth;
uniform sampler2D u_motion;
uniform vec2  u_render_size;
uniform vec2  u_jitter;
uniform float u_blend_alpha;
uniform int   u_first_frame;
uniform float u_depth_threshold;

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
float lumaYcocg(vec3 c) { return c.x; }
vec3 tonemapW(vec3 c)    { return c / (1.0 + c.x); }
vec3 tonemapWInv(vec3 c) { return c / max(1.0 - c.x, 1e-5); }

void buildNeighbourhood(vec2 uv, vec2 rcp,
                        out vec3 boxMin, out vec3 boxMax,
                        out vec3 mean,   out vec3 stdDev) {
    vec3 s[9];
    s[0] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2( 0.0,  0.0) * rcp).rgb));
    s[1] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2( 1.0,  0.0) * rcp).rgb));
    s[2] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2(-1.0,  0.0) * rcp).rgb));
    s[3] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2( 0.0,  1.0) * rcp).rgb));
    s[4] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2( 0.0, -1.0) * rcp).rgb));
    s[5] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2( 1.0,  1.0) * rcp).rgb));
    s[6] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2(-1.0,  1.0) * rcp).rgb));
    s[7] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2( 1.0, -1.0) * rcp).rgb));
    s[8] = tonemapW(ycocgFromRgb(texture(u_current, uv + vec2(-1.0, -1.0) * rcp).rgb));
    boxMin = s[0]; boxMax = s[0]; mean = s[0];
    for (int i = 1; i < 9; i++) { boxMin = min(boxMin, s[i]); boxMax = max(boxMax, s[i]); mean += s[i]; }
    mean *= (1.0 / 9.0);
    vec3 var = vec3(0.0);
    for (int i = 0; i < 9; i++) { vec3 d = s[i] - mean; var += d * d; }
    stdDev = sqrt(var * (1.0 / 9.0));
}

vec2 bestMotionVector(vec2 uv, vec2 rcp) {
    vec2  best  = texture(u_motion, uv).rg;
    float bestD = texture(u_depth,  uv).r;
    const vec2 offs[8] = vec2[8](
        vec2( 1.0, 0.0), vec2(-1.0, 0.0), vec2(0.0,  1.0), vec2(0.0, -1.0),
        vec2( 1.0, 1.0), vec2(-1.0, 1.0), vec2(1.0, -1.0), vec2(-1.0,-1.0));
    for (int i = 0; i < 8; i++) {
        vec2  s = uv + offs[i] * rcp;
        float d = texture(u_depth, s).r;
        if (d > 0.0 && d < bestD) { bestD = d; best = texture(u_motion, s).rg; }
    }
    return best;
}

void main() {
    vec2 rcp    = 1.0 / u_render_size;
    vec2 uvCur  = clamp(v_uv - u_jitter, vec2(0.0), vec2(1.0));
    vec3 curRgb = texture(u_current, uvCur).rgb;

    if (u_first_frame != 0) { fragColor = vec4(curRgb, 1.0); return; }

    vec3 c = tonemapW(ycocgFromRgb(curRgb));

    vec3 boxMin, boxMax, mean, stdDev;
    buildNeighbourhood(uvCur, rcp, boxMin, boxMax, mean, stdDev);

    float lumC      = lumaYcocg(c);
    float lumMean   = lumaYcocg(mean);
    float lumRange  = lumaYcocg(boxMax) - lumaYcocg(boxMin) + 1e-5;
    float subpixel  = clamp((lumC - lumMean) / lumRange, 0.0, 1.0);
    subpixel        = smoothstep(0.2, 0.8, subpixel);

    float sigma = mix(1.0, 3.5, subpixel);
    vec3 clipMin = mean - sigma * stdDev;
    vec3 clipMax = mean + sigma * stdDev;

    vec2 mv     = bestMotionVector(uvCur, rcp);
    vec2 prevUV = clamp(v_uv - mv, vec2(0.0), vec2(1.0));

    float outOfView    = float(any(lessThan(v_uv - mv, vec2(0.0))) ||
                               any(greaterThan(v_uv - mv, vec2(1.0))));
    float curDepth     = texture(u_depth,      uvCur).r;
    float prevDepth    = texture(u_prev_depth, prevUV).r;
    float depthDiff    = abs(curDepth - prevDepth) / max(curDepth + prevDepth, 1e-4);
    float disocclusion = smoothstep(u_depth_threshold, u_depth_threshold * 4.0, depthDiff);
    disocclusion = max(disocclusion, outOfView);

    vec3 histRaw = tonemapW(ycocgFromRgb(texture(u_history, prevUV).rgb));
    vec3 histClipped = clamp(histRaw, clipMin, clipMax);

    float histLum    = lumaYcocg(histClipped);
    float clipAmount = abs(lumaYcocg(histRaw) - histLum) / (lumRange + 1e-5);
    float blendFromClip = smoothstep(0.0, 1.0, clipAmount);

    float baseAlpha = mix(u_blend_alpha, u_blend_alpha * 0.4, subpixel);
    float alpha     = baseAlpha;
    alpha = mix(alpha, 1.0, disocclusion);
    alpha = mix(alpha, min(alpha + 0.3, 1.0), blendFromClip * (1.0 - subpixel));

    vec3 blended   = mix(histClipped, c, clamp(alpha, 0.0, 1.0));
    fragColor      = vec4(max(rgbFromYcocg(tonemapWInv(blended)), vec3(0.0)), 1.0);
}