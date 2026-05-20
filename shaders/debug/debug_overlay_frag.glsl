#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform vec2  u_res;
uniform int   u_on_ground;
uniform int   u_enabled;
uniform float u_sdf_val;
uniform float u_radius;
uniform float u_gnd_thresh;
uniform float u_speed;

uniform vec2  u_norm_dir_ss;
uniform vec2  u_grav_dir_ss;
uniform float u_col_ring_px;
uniform float u_gnd_ring_px;

uniform int   u_probe_count;
uniform vec2  u_probe_ss[32];
uniform float u_probe_sdf[32];
uniform float u_probe_radius;

float sdSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa,ba)/dot(ba,ba), 0.0, 1.0));
}

float sdRing(vec2 p, vec2 c, float r) {
    return abs(length(p - c) - r);
}

vec4 blend(vec4 dst, vec3 rgb, float a) {
    return vec4(mix(dst.rgb, rgb, a), 1.0);
}

void main() {
    if (u_enabled == 0) { discard; return; }

    vec2 px   = v_uv * u_res;
    vec2 ctr  = u_res * 0.5;
    vec4 col  = vec4(0.0);

    float crossR = 7.0;
    float crossW = 1.8;
    vec2  dc = abs(px - ctr);
    if ((dc.x < crossR && dc.y < crossW) || (dc.y < crossR && dc.x < crossW)) {
        vec3 cc = u_on_ground == 1 ? vec3(0.15, 1.0, 0.25) : vec3(1.0, 0.55, 0.05);
        col = blend(col, cc, 0.95);
    }

    if (u_col_ring_px > 2.0) {
        float dr = sdRing(px, ctr, u_col_ring_px);
        vec3 rc = u_on_ground == 1 ? vec3(0.2, 1.0, 0.3) : vec3(1.0, 0.45, 0.05);
        float a = clamp(1.0 - dr / 2.5, 0.0, 1.0);
        if (dr < 3.5) col = blend(col, rc, a * 0.9);
    }

    if (u_gnd_ring_px > 2.0) {
        float dr = sdRing(px, ctr, u_gnd_ring_px);
        float a = clamp(1.0 - dr / 2.0, 0.0, 1.0) * 0.55;
        if (dr < 3.0) col = blend(col, vec3(0.25, 0.6, 1.0), a);
    }

    vec2 normEnd = ctr + u_norm_dir_ss;
    float dn = sdSeg(px, ctr, normEnd);
    float an = clamp(1.0 - dn / 2.5, 0.0, 1.0);
    if (dn < 3.5) col = blend(col, vec3(0.2, 0.85, 1.0), an * 0.92);
    float dnd = length(px - normEnd);
    if (dnd < 5.0) col = blend(col, vec3(0.2, 0.85, 1.0), 0.95);

    vec2 gravEnd = ctr + u_grav_dir_ss;
    float dg = sdSeg(px, ctr, gravEnd);
    float ag = clamp(1.0 - dg / 2.5, 0.0, 1.0);
    if (dg < 3.5) col = blend(col, vec3(1.0, 0.25, 0.25), ag * 0.92);
    float dgd = length(px - gravEnd);
    if (dgd < 5.0) col = blend(col, vec3(1.0, 0.25, 0.25), 0.95);

    for (int i = 0; i < u_probe_count; i++) {
        float d = length(px - u_probe_ss[i]);
        if (d < 4.5) {
            vec3 pc = u_probe_sdf[i] < u_probe_radius
                ? vec3(1.0, 0.15, 0.15)
                : vec3(0.15, 0.95, 0.35);
            col = blend(col, pc, clamp(1.0 - d / 4.5, 0.0, 1.0) * 0.88);
        }
    }

    fragColor = col;
}
