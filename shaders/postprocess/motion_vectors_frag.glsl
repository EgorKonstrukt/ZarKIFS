#version 330 core
in vec2 v_uv;
out vec2 fragMotion;

uniform sampler2D u_depth;
uniform vec2  u_render_size;
uniform vec2  u_jitter;
uniform vec3  u_cam_pos;
uniform vec3  u_cam_fwd;
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform float u_fov;
uniform vec3  u_prev_cam_pos;
uniform vec3  u_prev_cam_fwd;
uniform vec3  u_prev_cam_right;
uniform vec3  u_prev_cam_up;
uniform float u_prev_fov;

const float CLIP_NEAR = 0.001;

vec3 uvToWorld(vec2 uv, float rayDist) {
    float aspect = u_render_size.x / u_render_size.y;
    vec2  ndc    = uv * 2.0 - 1.0;
    vec3  rd     = normalize(ndc.x * aspect * u_cam_right +
                             ndc.y * u_cam_up +
                             u_fov * u_cam_fwd);
    return u_cam_pos + rd * rayDist;
}

vec2 worldToPrevUV(vec3 wp) {
    vec3  d      = wp - u_prev_cam_pos;
    float aspect = u_render_size.x / u_render_size.y;
    float projZ  = dot(d, u_prev_cam_fwd);
    if (projZ < CLIP_NEAR) return vec2(-2.0);
    vec2 screen;
    screen.x = dot(d, u_prev_cam_right) / (u_prev_fov * aspect * projZ);
    screen.y = dot(d, u_prev_cam_up)    / (u_prev_fov         * projZ);
    return screen * 0.5 + 0.5;
}

vec2 motionAt(vec2 uv, vec2 rcpSize) {
    float depth = texture(u_depth, uv).r;
    if (depth <= 0.0) return vec2(0.0);
    vec3 wp     = uvToWorld(uv, depth);
    vec2 prevUV = worldToPrevUV(wp);
    if (prevUV.x < -1.0) return vec2(0.0);
    return uv - prevUV;
}

void main() {
    vec2 rcpSize = 1.0 / u_render_size;
    vec2 center  = clamp(v_uv - u_jitter, vec2(0.0), vec2(1.0));
    vec2 offsets[8];
    offsets[0] = vec2( 1.0,  0.0);
    offsets[1] = vec2(-1.0,  0.0);
    offsets[2] = vec2( 0.0,  1.0);
    offsets[3] = vec2( 0.0, -1.0);
    offsets[4] = vec2( 1.0,  1.0);
    offsets[5] = vec2(-1.0,  1.0);
    offsets[6] = vec2( 1.0, -1.0);
    offsets[7] = vec2(-1.0, -1.0);
    float bestDepth = texture(u_depth, center).r;
    vec2  bestMV    = motionAt(center, rcpSize);
    for (int i = 0; i < 8; i++) {
        vec2  s = clamp(center + offsets[i] * rcpSize, vec2(0.0), vec2(1.0));
        float d = texture(u_depth, s).r;
        if (d > 0.0 && d < bestDepth) {
            bestDepth = d;
            bestMV    = motionAt(s, rcpSize);
        }
    }
    fragMotion = bestMV;
}