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
    return max(vec3(b + c.y, g, b), vec3(0.0));
}
float lumaYcocg(vec3 c) { return c.x; }
vec3 tonemapW(vec3 c)    { return c / (1.0 + c.x); }
vec3 tonemapWInv(vec3 c) { return c / max(1.0 - c.x, 1e-5); }

float fireflyWeight(float luma) {
    return 1.0 / (1.0 + luma * luma * 4.0);
}

void buildNeighbourhood(vec2 uv, vec2 rcp,
                        out vec3 mean, out vec3 stdDev,
                        out vec3 boxMin, out vec3 boxMax,
                        out float maxNeighbourLuma) {
    const vec2 kernel[9] = vec2[9](
        vec2( 0.0,  0.0), vec2( 1.0,  0.0), vec2(-1.0,  0.0),
        vec2( 0.0,  1.0), vec2( 0.0, -1.0), vec2( 1.0,  1.0),
        vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2(-1.0, -1.0)
    );
    const float w[9] = float[9](
        0.25, 0.125, 0.125, 0.125, 0.125, 0.0625, 0.0625, 0.0625, 0.0625
    );
    vec3 s[9];
    float wSum = 0.0;
    mean = vec3(0.0);
    vec3 meanSq = vec3(0.0);
    boxMin =  vec3(1e9);
    boxMax = vec3(-1e9);
    maxNeighbourLuma = 0.0;
    for (int i = 0; i < 9; i++) {
        vec3 raw  = texture(u_current, uv + kernel[i] * rcp).rgb;
        float fw  = fireflyWeight(dot(raw, vec3(0.2126, 0.7152, 0.0722)));
        float wi  = w[i] * fw;
        s[i]      = tonemapW(ycocgFromRgb(raw));
        mean      += s[i] * wi;
        meanSq    += s[i] * s[i] * wi;
        wSum      += wi;
        boxMin     = min(boxMin, s[i]);
        boxMax     = max(boxMax, s[i]);
        maxNeighbourLuma = max(maxNeighbourLuma, lumaYcocg(s[i]));
    }
    mean   /= max(wSum, 1e-5);
    meanSq /= max(wSum, 1e-5);
    stdDev  = sqrt(max(meanSq - mean * mean, vec3(0.0)));
}

vec3 clipToAabb(vec3 history, vec3 mn, vec3 mx) {
    vec3 center = 0.5 * (mx + mn);
    vec3 ext    = max(0.5 * (mx - mn), vec3(1e-5));
    vec3 delta  = history - center;
    vec3 scale  = abs(delta) / ext;
    float maxS  = max(scale.x, max(scale.y, scale.z));
    return (maxS > 1.0) ? center + delta / maxS : history;
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

    vec3 mean, stdDev, boxMin, boxMax;
    float maxNeighLuma;
    buildNeighbourhood(uvCur, rcp, mean, stdDev, boxMin, boxMax, maxNeighLuma);

    vec3 c = tonemapW(ycocgFromRgb(curRgb));

    float lumC    = lumaYcocg(c);
    float lumMean = lumaYcocg(mean);
    float lumStd  = max(lumaYcocg(stdDev), 1e-5);

    float subpixel = clamp(abs(lumC - lumMean) / (lumStd * 2.0 + 1e-5), 0.0, 1.0);
    subpixel = smoothstep(0.0, 1.0, subpixel);

    float sigma  = mix(1.5, 2.5, subpixel);
    vec3 clipMin = max(mean - sigma * stdDev, boxMin);
    vec3 clipMax = min(mean + sigma * stdDev, boxMax);

    vec2 mv     = bestMotionVector(uvCur, rcp);
    vec2 prevUV = v_uv - mv;
    float outOfView = float(any(lessThan(prevUV, vec2(0.0))) ||
                            any(greaterThan(prevUV, vec2(1.0))));
    prevUV = clamp(prevUV, vec2(0.0), vec2(1.0));

    float curDepth  = texture(u_depth,      uvCur).r;
    float prevDepth = texture(u_prev_depth, prevUV).r;
    float depthDiff = abs(curDepth - prevDepth) / max(curDepth + prevDepth, 1e-4);
    float disocclusion = smoothstep(u_depth_threshold, u_depth_threshold * 3.0, depthDiff);
    disocclusion = max(disocclusion, outOfView);

    vec3 histRaw     = tonemapW(ycocgFromRgb(texture(u_history, prevUV).rgb));
    float histLuma   = lumaYcocg(histRaw);

    float fireflySuppress = smoothstep(maxNeighLuma * 1.5, maxNeighLuma * 3.0, histLuma);
    histRaw = mix(histRaw, mean, fireflySuppress);

    vec3 histClipped = clipToAabb(histRaw, clipMin, clipMax);

    float clipDelta  = abs(lumaYcocg(histRaw) - lumaYcocg(histClipped)) / (lumStd + 1e-5);
    float blendBoost = smoothstep(0.0, 2.0, clipDelta);

    float baseAlpha = mix(u_blend_alpha, u_blend_alpha * 0.5, subpixel);
    float alpha     = baseAlpha;
    alpha = mix(alpha, 1.0, disocclusion);
    alpha = mix(alpha, min(alpha + 0.2, 1.0), blendBoost * (1.0 - subpixel));
    alpha = clamp(alpha, 0.0, 1.0);

    float wCur  = fireflyWeight(lumC);
    float wHist = fireflyWeight(lumaYcocg(histClipped)) * (1.0 - alpha);
    vec3 blended = (c * wCur + histClipped * wHist) / max(wCur + wHist, 1e-5);

    fragColor = vec4(max(rgbFromYcocg(tonemapWInv(blended)), vec3(0.0)), 1.0);
}
