#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform vec2  u_src_resolution;
uniform int   u_flip_y;
uniform float u_easu_sharpness;

float lanczos2Weight(float x) {
    if (abs(x) < 1e-5) return 1.0;
    if (abs(x) >= 2.0)  return 0.0;
    float px = 3.14159265 * x;
    float px2 = 3.14159265 * x * 0.5;
    return (sin(px) / px) * (sin(px2) / px2);
}

vec3 sampleSrc(vec2 uv) {
    return texture(u_scene, clamp(uv, vec2(0.0), vec2(1.0))).rgb;
}

vec3 fsrEasu(vec2 uv) {
    vec2 rcp = 1.0 / u_src_resolution;
    vec2 pp  = uv * u_src_resolution - 0.5;
    vec2 tc  = floor(pp) + 0.5;
    vec2 fp  = pp - floor(pp);

    vec3 lumW = vec3(0.299, 0.587, 0.114);

    vec3 c00 = sampleSrc((tc + vec2(-1.0, -1.0)) * rcp);
    vec3 c10 = sampleSrc((tc + vec2( 0.0, -1.0)) * rcp);
    vec3 c20 = sampleSrc((tc + vec2( 1.0, -1.0)) * rcp);
    vec3 c30 = sampleSrc((tc + vec2( 2.0, -1.0)) * rcp);
    vec3 c01 = sampleSrc((tc + vec2(-1.0,  0.0)) * rcp);
    vec3 c11 = sampleSrc((tc + vec2( 0.0,  0.0)) * rcp);
    vec3 c21 = sampleSrc((tc + vec2( 1.0,  0.0)) * rcp);
    vec3 c31 = sampleSrc((tc + vec2( 2.0,  0.0)) * rcp);
    vec3 c02 = sampleSrc((tc + vec2(-1.0,  1.0)) * rcp);
    vec3 c12 = sampleSrc((tc + vec2( 0.0,  1.0)) * rcp);
    vec3 c22 = sampleSrc((tc + vec2( 1.0,  1.0)) * rcp);
    vec3 c32 = sampleSrc((tc + vec2( 2.0,  1.0)) * rcp);
    vec3 c03 = sampleSrc((tc + vec2(-1.0,  2.0)) * rcp);
    vec3 c13 = sampleSrc((tc + vec2( 0.0,  2.0)) * rcp);
    vec3 c23 = sampleSrc((tc + vec2( 1.0,  2.0)) * rcp);
    vec3 c33 = sampleSrc((tc + vec2( 2.0,  2.0)) * rcp);

    float l11 = dot(c11, lumW), l21 = dot(c21, lumW);
    float l12 = dot(c12, lumW), l22 = dot(c22, lumW);

    float gx = (l21 - l11) + (l22 - l12);
    float gy = (l12 - l11) + (l22 - l21);
    float gLen = sqrt(gx*gx + gy*gy) + 1e-5;
    vec2 dir = vec2(gx, gy) / gLen;

    float edgeness = clamp(gLen * 4.0, 0.0, 1.0);
    float aniso = mix(1.0, 1.0 + u_easu_sharpness * 1.5, edgeness);

    vec3 col = vec3(0.0);
    float wsum = 0.0;

    vec2 offsets[16];
    vec3 samples[16];
    offsets[0]  = vec2(-1.0,-1.0); samples[0]  = c00;
    offsets[1]  = vec2( 0.0,-1.0); samples[1]  = c10;
    offsets[2]  = vec2( 1.0,-1.0); samples[2]  = c20;
    offsets[3]  = vec2( 2.0,-1.0); samples[3]  = c30;
    offsets[4]  = vec2(-1.0, 0.0); samples[4]  = c01;
    offsets[5]  = vec2( 0.0, 0.0); samples[5]  = c11;
    offsets[6]  = vec2( 1.0, 0.0); samples[6]  = c21;
    offsets[7]  = vec2( 2.0, 0.0); samples[7]  = c31;
    offsets[8]  = vec2(-1.0, 1.0); samples[8]  = c02;
    offsets[9]  = vec2( 0.0, 1.0); samples[9]  = c12;
    offsets[10] = vec2( 1.0, 1.0); samples[10] = c22;
    offsets[11] = vec2( 2.0, 1.0); samples[11] = c32;
    offsets[12] = vec2(-1.0, 2.0); samples[12] = c03;
    offsets[13] = vec2( 0.0, 2.0); samples[13] = c13;
    offsets[14] = vec2( 1.0, 2.0); samples[14] = c23;
    offsets[15] = vec2( 2.0, 2.0); samples[15] = c33;

    vec2 perp = vec2(-dir.y, dir.x);

    for (int k = 0; k < 16; k++) {
        vec2 d = offsets[k] - fp;
        float along = dot(d, dir);
        float across = dot(d, perp);
        float wx = lanczos2Weight(along * aniso);
        float wy = lanczos2Weight(across / aniso);
        float w  = wx * wy;
        col  += w * samples[k];
        wsum += w;
    }

    if (wsum < 1e-4) {
        return mix(c11, c22, fp.x * 0.5 + fp.y * 0.5);
    }
    return clamp(col / wsum, vec3(0.0), vec3(1.0));
}

void main() {
    vec2 uv = (u_flip_y != 0) ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    fragColor = vec4(fsrEasu(uv), 1.0);
}