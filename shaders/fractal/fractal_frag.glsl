#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform float u_time;
uniform vec2  u_resolution;
uniform int   u_iterations;
uniform float u_scale;
uniform float u_fold_x;
uniform float u_fold_y;
uniform float u_fold_z;
uniform float u_rot_x;
uniform float u_rot_y;
uniform float u_rot_z;
uniform mat3  u_rot_mat;
uniform float u_offset_x;
uniform float u_offset_y;
uniform float u_offset_z;
uniform float u_julia_x;
uniform float u_julia_y;
uniform float u_julia_z;
uniform int   u_fractal_type;
uniform float u_bailout;
uniform float u_min_dist;
uniform float u_fog_density;
uniform vec3  u_fog_color;
uniform vec3  u_color1;
uniform vec3  u_color2;
uniform vec3  u_color3;
uniform int   u_color_mode;
uniform float u_ao_strength;
uniform float u_ao_radius;
uniform int   u_ao_samples;
uniform float u_shadow_soft;
uniform int   u_shadows;
uniform float u_glow;
uniform float u_de_multiplier;
uniform vec3  u_cam_pos;
uniform vec3  u_cam_fwd;
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform float u_fov;
uniform int   u_animate;
uniform float u_anim_speed;

// --- Orbit trap ---
uniform int   u_orbit_trap_type;  // 0=sphere, 1=plane-y, 2=cube, 3=torus

// --- Light primary ---
uniform vec3  u_light_dir;
uniform float u_specular_power;
uniform float u_specular_strength;
uniform float u_ambient;
uniform float u_subsurface;
uniform float u_fresnel_power;

// --- Second light ---
uniform vec3  u_light2_dir;
uniform vec3  u_light2_color;
uniform float u_light2_strength;

// --- Color animation ---
uniform float u_color_anim_speed;
uniform float u_color_offset;

// --- Raymarching ---
uniform float u_step_scale;
uniform float u_normal_eps;
uniform float u_reflection;
uniform int   u_max_steps;
uniform float u_max_dist;
uniform float u_hit_eps;
uniform int   u_shadow_steps;
uniform float u_shadow_mint;
uniform float u_shadow_maxt;
uniform float u_ao_step_scale;
uniform int   u_rm_overrelax;
uniform float u_overrelax_factor;

// --- Glow ---
uniform float u_glow_intensity;
uniform float u_glow_falloff;
uniform float u_glow_radius;
uniform float u_rim_strength;
uniform float u_emission;

// --- Background ---
uniform vec3  u_bg_color1;
uniform vec3  u_bg_color2;
uniform int   u_bg_mode;

// --- Stars ---
uniform float u_star_density;
uniform float u_star_brightness;
uniform float u_star_twinkle;
uniform float u_star_size;
uniform int   u_milkyway;

// --- Anti-aliasing ---
uniform int   u_aa_samples;
uniform vec2  u_taa_jitter;    // субпиксельный NDC-сдвиг для TAA

// --- Performance feature flags ---
uniform int u_feat_ao;           // 0=off 1=on
uniform int u_feat_shadows;      // 0=off 1=on  (overrides u_shadows)
uniform int u_feat_normals_full; // 0=3-tap cheap 1=6-tap full
uniform int u_feat_second_light; // 0=off 1=on
uniform int u_feat_fog;          // 0=off 1=on
uniform int u_feat_glow;         // 0=off 1=on
uniform int u_feat_reflection;   // 0=off 1=on
uniform int u_feat_subsurface;   // 0=off 1=on
uniform int u_feat_orbit_trap;   // 0=off (step count only) 1=on

// --- Mandelbox fine-tune ---
uniform float u_mb_fold_limit;
uniform float u_mb_sphere_inner;
uniform float u_mb_sphere_outer;
uniform float u_mb_fixed_radius;
uniform float u_mb_color_scale;
uniform float u_mb_rot_per_iter;
uniform int   u_mb_fold_mode;     // 0=clamp, 1=abs, 2=sin

// --- Menger Sponge fine-tune ---
uniform float u_ms_cross_width;
uniform float u_ms_scale;
uniform float u_ms_offset;
uniform float u_ms_twist;
uniform float u_ms_sharpness;

// --- Sierpinski fine-tune ---
uniform float u_si_vertex_spread;
uniform float u_si_fold_bias;
uniform float u_si_twist;
uniform float u_si_squash;
uniform float u_si_vertex_jitter;

// --- Octahedron IFS fine-tune ---
uniform float u_oc_ifs_scale;
uniform float u_oc_twist;
uniform float u_oc_sharpness;
uniform float u_oc_offset_uni;
uniform float u_oc_fold_amount;
uniform float u_oc_offset_x;
uniform float u_oc_offset_y;
uniform float u_oc_offset_z;
uniform float u_oc_rot_x;
uniform float u_oc_rot_z;

// --- Mandelbulb fine-tune ---
uniform float u_mb2_power;
uniform float u_mb2_bailout;
uniform float u_mb2_julia_x;
uniform float u_mb2_julia_y;
uniform float u_mb2_julia_z;
uniform int   u_mb2_julia_mode;
uniform float u_mb2_fold_strength;
uniform int   u_mb2_fold_type;

// --- Pseudo-Kleinian fine-tune ---
uniform float u_kl_scale;
uniform float u_kl_cx;
uniform float u_kl_cy;
uniform float u_kl_cz;
uniform float u_kl_fold_limit;
uniform float u_kl_sph_radius;
uniform float u_kl_rot_per_iter;
uniform float u_kl_mix_factor;

// --- Mandelbox per-axis fold ---
uniform float u_mb_fold_x;
uniform float u_mb_fold_y;
uniform float u_mb_fold_z;
uniform int   u_mb_julia_mode;

// --- Menger per-axis / rotation ---
uniform float u_ms_rot_x;
uniform float u_ms_rot_z;
uniform float u_ms_scale_y;
uniform float u_ms_scale_z;

// --- Sierpinski per-axis rotation ---
uniform float u_si_rot_x;
uniform float u_si_rot_z;

// --- Mandelbox extended ---
uniform float u_mb_scale_x;
uniform float u_mb_scale_y;
uniform float u_mb_scale_z;
uniform float u_mb_offset_x;
uniform float u_mb_offset_y;
uniform float u_mb_offset_z;
uniform float u_mb_inversion_radius;

// --- Mandelbulb extended ---
uniform float u_mb2_polar_mix;
uniform float u_mb2_rot_per_iter;
uniform int   u_mb2_abs_x;
uniform int   u_mb2_abs_y;
uniform int   u_mb2_abs_z;

// --- Menger extended ---
uniform float u_ms_offset_x;
uniform float u_ms_offset_y;
uniform float u_ms_offset_z;
uniform int   u_ms_fold_type;
uniform float u_ms_fold_abs_amount;

// --- Sierpinski extended ---
uniform float u_si_scale_x;
uniform float u_si_scale_y;
uniform float u_si_scale_z;
uniform float u_si_offset_x;
uniform float u_si_offset_y;
uniform float u_si_offset_z;
uniform float u_si_rot_y;

// --- Octahedron extended ---
uniform float u_oc_scale_y;
uniform float u_oc_scale_z;
uniform int   u_oc_julia_mode;
uniform float u_oc_julia_x;
uniform float u_oc_julia_y;
uniform float u_oc_julia_z;

// --- Kleinian extended ---
uniform float u_kl_fold_limit_x;
uniform float u_kl_fold_limit_y;
uniform float u_kl_fold_limit_z;
uniform int   u_kl_julia_mode;
uniform float u_kl_offset_x;
uniform float u_kl_offset_y;
uniform float u_kl_offset_z;

// --- Quaternion Julia 4D ---
uniform float u_qj_cx;
uniform float u_qj_cy;
uniform float u_qj_cz;
uniform float u_qj_cw;
uniform float u_qj_w_slice;
uniform float u_qj_bailout;
uniform float u_qj_slice_rot_xw;
uniform float u_qj_slice_rot_yw;
uniform float u_qj_slice_rot_zw;

// --- Global space operators ---
uniform int   u_warp_enabled;
uniform float u_warp_strength;
uniform float u_warp_freq;
uniform int   u_warp_type;
uniform int   u_twist_axis;
uniform float u_twist_amount;
uniform int   u_fold_mirror_x;
uniform int   u_fold_mirror_y;
uniform int   u_fold_mirror_z;
uniform int   u_rep_enabled;
uniform float u_rep_cell_x;
uniform float u_rep_cell_y;
uniform float u_rep_cell_z;

// --- Spherical inversion ---
uniform int   u_sph_inv_enabled;
uniform float u_sph_inv_radius;
uniform float u_sph_inv_cx;
uniform float u_sph_inv_cy;
uniform float u_sph_inv_cz;

// --- Lattice fold ---
uniform int   u_lattice_fold_enabled;
uniform float u_lattice_fold_x;
uniform float u_lattice_fold_y;
uniform float u_lattice_fold_z;

#define PI 3.14159265358979323846
#define MAX_STEPS 512
#define MAX_DIST  200.0

mat3 rotX(float a) {
    float c = cos(a), s = sin(a);
    return mat3(1,0,0, 0,c,-s, 0,s,c);
}
mat3 rotY(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,0,s, 0,1,0, -s,0,c);
}
mat3 rotZ(float a) {
    float c = cos(a), s = sin(a);
    return mat3(c,-s,0, s,c,0, 0,0,1);
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float sdTetra(vec3 p, float r) {
    float md = max(max(-p.x-p.y-p.z, p.x+p.y-p.z), max(-p.x+p.y+p.z, p.x-p.y+p.z));
    return (md - r) / sqrt(3.0);
}

float orbitTrap(vec3 p, float s) {
    if (u_feat_orbit_trap == 0) return dot(p, p) * (1.0 / (s * s));
    float sinvs = 1.0 / (abs(s) + 0.0001);
    if (u_orbit_trap_type == 0) {
        return dot(p, p) * (1.0 / (s * s));
    } else if (u_orbit_trap_type == 1) {
        return abs(p.y) * sinvs;
    } else if (u_orbit_trap_type == 2) {
        return max(abs(p.x), max(abs(p.y), abs(p.z))) * sinvs;
    } else {
        float r = length(p.xz);
        return length(vec2(r - abs(s) * 0.4, p.y)) / (abs(s) * 0.3 + 0.0001);
    }
}

vec2 mandelbox(vec3 pos) {
    vec3 p = pos;
    float trap = 1e10;
    float dr = 1.0;
    float foldX = (u_mb_fold_x > 0.0) ? u_mb_fold_x : u_mb_fold_limit;
    float foldY = (u_mb_fold_y > 0.0) ? u_mb_fold_y : u_mb_fold_limit;
    float foldZ = (u_mb_fold_z > 0.0) ? u_mb_fold_z : u_mb_fold_limit;
    float sphIn  = u_mb_sphere_inner;
    float sphOut = u_mb_sphere_outer;
    float sc     = u_scale;
    float absSc  = abs(sc);
    vec3 posOff = (u_mb_offset_x != 0.0 || u_mb_offset_y != 0.0 || u_mb_offset_z != 0.0)
        ? vec3(u_mb_offset_x, u_mb_offset_y, u_mb_offset_z)
        : pos;
    vec3 joff = vec3(u_julia_x, u_julia_y, u_julia_z);
    if (u_mb_inversion_radius > 0.001) {
        float r2 = dot(p, p);
        float k  = (u_mb_inversion_radius * u_mb_inversion_radius) / max(r2, 1e-6);
        p  *= k;
        dr *= k;
    }
    for (int i = 0; i < u_iterations; i++) {
        if (u_mb_rot_per_iter > 0.0001) p = rotY(u_mb_rot_per_iter * float(i)) * p;
        if (u_mb_fold_mode == 0) {
            p.x = clamp(p.x, -foldX, foldX) * 2.0 - p.x;
            p.y = clamp(p.y, -foldY, foldY) * 2.0 - p.y;
            p.z = clamp(p.z, -foldZ, foldZ) * 2.0 - p.z;
        } else {
            p.x = abs(p.x + foldX) - abs(p.x - foldX) - p.x;
            p.y = abs(p.y + foldY) - abs(p.y - foldY) - p.y;
            p.z = abs(p.z + foldZ) - abs(p.z - foldZ) - p.z;
        }
        float r2 = dot(p, p);
        float r  = sqrt(r2);
        float sphK;
        if (r < sphIn) {
            sphK = (sphOut * sphOut) / (sphIn * sphIn);
        } else if (r < sphOut) {
            sphK = (sphOut * sphOut) / r2;
        } else {
            sphK = 1.0;
        }
        p  *= sphK;
        dr *= sphK;
        dr  = dr * absSc + 1.0;
        if (u_mb_julia_mode == 1) {
            p = p * sc + joff;
        } else {
            p = p * sc + posOff;
        }
        trap = min(trap, orbitTrap(p, u_mb_color_scale));
        if (dot(p, p) > u_bailout * u_bailout) break;
    }
    return vec2(length(p) / max(abs(dr), 1e-6) * u_de_multiplier, trap);
}

vec2 mengerSponge(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    float ms = u_ms_scale;
    float sy = (u_ms_scale_y > 0.0) ? u_ms_scale_y : ms;
    float sz = (u_ms_scale_z > 0.0) ? u_ms_scale_z : ms;
    float ox = (u_ms_offset_x != 0.0) ? u_ms_offset_x : u_ms_offset;
    float oy = (u_ms_offset_y != 0.0) ? u_ms_offset_y : u_ms_offset;
    float oz = (u_ms_offset_z != 0.0) ? u_ms_offset_z : u_ms_offset;
    for (int i = 0; i < u_iterations; i++) {
        if (u_ms_twist > 0.001)  p = rotY(u_ms_twist) * p;
        if (u_ms_rot_x > 0.001) p = rotX(u_ms_rot_x) * p;
        if (u_ms_rot_z > 0.001) p = rotZ(u_ms_rot_z) * p;
        if (u_ms_fold_type == 0) {
            p = abs(p);
        } else if (u_ms_fold_type == 1) {
            p = abs(p) - vec3(u_ms_fold_abs_amount);
        } else {
            p = abs(p + vec3(u_ms_fold_abs_amount)) - abs(p - vec3(u_ms_fold_abs_amount)) - p;
        }
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;
        p.x = p.x * ms - ox;
        p.y = p.y * sy - oy;
        p.z = p.z * sz - oz;
        if (abs(oz) > 0.0001)
            p.z += oz * clamp(p.z / oz * 0.5 + 0.5, 0.0, 1.0) * u_ms_cross_width;
        s *= ms;
        trap = min(trap, orbitTrap(p, s));
    }
    float boxDist;
    vec3 q = abs(p) - vec3(1.0);
    if (u_ms_sharpness >= 0.99) {
        boxDist = sdBox(p, vec3(1.0));
    } else {
        float r = 1.0 - u_ms_sharpness;
        vec3 qr = max(q + r, 0.0);
        boxDist = length(qr) - r + min(max(q.x, max(q.y, q.z)), 0.0);
    }
    return vec2(boxDist / s * u_de_multiplier, trap);
}

vec2 sierpinski(vec3 pos) {
    float vs = u_si_vertex_spread;
    vec3 A = vec3( vs,  vs,  vs);
    vec3 B = vec3(-vs, -vs,  vs);
    vec3 C = vec3(-vs,  vs, -vs);
    vec3 D = vec3( vs, -vs, -vs);
    if (u_si_vertex_jitter > 0.0001) {
        float j = u_si_vertex_jitter;
        A += vec3( j, -j,  j) * 0.5;
        B += vec3(-j,  j,  j) * 0.5;
        C += vec3( j,  j, -j) * 0.5;
        D += vec3(-j, -j, -j) * 0.5;
    }
    vec3 p = pos;
    p.y *= u_si_squash;
    float scX = (u_si_scale_x > 0.0) ? u_si_scale_x : u_si_fold_bias;
    float scY = (u_si_scale_y > 0.0) ? u_si_scale_y : u_si_fold_bias;
    float scZ = (u_si_scale_z > 0.0) ? u_si_scale_z : u_si_fold_bias;
    vec3 siOff = vec3(u_si_offset_x, u_si_offset_y, u_si_offset_z);
    float scale = 1.0;
    float trap = 1e10;
    for (int i = 0; i < u_iterations; i++) {
        if (u_si_twist > 0.001)  p = rotY(u_si_twist) * p;
        if (u_si_rot_x > 0.001) p = rotX(u_si_rot_x) * p;
        if (u_si_rot_z > 0.001) p = rotZ(u_si_rot_z) * p;
        if (u_si_rot_y > 0.001) p = rotY(u_si_rot_y) * p;
        vec3 closest = A;
        float d = dot(p - A, p - A);
        float db = dot(p - B, p - B);
        float dc = dot(p - C, p - C);
        float dd = dot(p - D, p - D);
        if (db < d) { closest = B; d = db; }
        if (dc < d) { closest = C; d = dc; }
        if (dd < d) { closest = D; }
        p = vec3(scX * p.x - closest.x * (scX - 1.0),
                 scY * p.y - closest.y * (scY - 1.0),
                 scZ * p.z - closest.z * (scZ - 1.0)) + siOff;
        scale *= u_si_fold_bias;
        trap = min(trap, orbitTrap(p, scale));
    }
    return vec2(sdTetra(p, 1.0) / scale * u_de_multiplier, trap);
}

vec2 octahedronIFS(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    float IFS_SCALE = u_oc_ifs_scale;
    float scY = (u_oc_scale_y > 0.0) ? u_oc_scale_y : IFS_SCALE;
    float scZ = (u_oc_scale_z > 0.0) ? u_oc_scale_z : IFS_SCALE;
    vec3 off = vec3(
        (u_oc_offset_x > 0.0) ? u_oc_offset_x : u_oc_offset_uni,
        (u_oc_offset_y > 0.0) ? u_oc_offset_y : u_oc_offset_uni,
        (u_oc_offset_z > 0.0) ? u_oc_offset_z : u_oc_offset_uni
    );
    vec3 juliaC = vec3(u_oc_julia_x, u_oc_julia_y, u_oc_julia_z);
    for (int i = 0; i < u_iterations; i++) {
        if (u_oc_twist  > 0.001) p = rotY(u_oc_twist)  * p;
        if (u_oc_rot_x  > 0.001) p = rotX(u_oc_rot_x)  * p;
        if (u_oc_rot_z  > 0.001) p = rotZ(u_oc_rot_z)  * p;
        if (u_oc_fold_amount > 0.001) {
            vec3 pabs = abs(p);
            p = mix(p, pabs, u_oc_fold_amount);
        } else {
            p = abs(p);
        }
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;
        if (u_oc_julia_mode == 1) {
            p = vec3(IFS_SCALE * p.x, scY * p.y, scZ * p.z) - off * (IFS_SCALE - 1.0) + juliaC;
        } else {
            p = vec3(IFS_SCALE * p.x, scY * p.y, scZ * p.z) - off * (IFS_SCALE - 1.0);
        }
        s *= IFS_SCALE;
        trap = min(trap, orbitTrap(p, s));
    }
    float sh = max(u_oc_sharpness, 0.5);
    float r;
    if (sh < 1.05) {
        r = abs(p.x) + abs(p.y) + abs(p.z) - 1.0;
    } else {
        r = pow(pow(abs(p.x), sh) + pow(abs(p.y), sh) + pow(abs(p.z), sh), 1.0/sh) - 1.0;
    }
    return vec2(r / s * u_de_multiplier, trap);
}

vec2 mandelbulb(vec3 pos) {
    vec3 p = pos;
    float trap = 1e10;
    float dr = 1.0;
    float r = 0.0;
    float pw = max(u_mb2_power, 1.0);
    float bail = u_mb2_bailout;
    for (int i = 0; i < u_iterations; i++) {
        r = length(p);
        if (r > bail) break;
        if (u_mb2_rot_per_iter > 0.0001) p = rotY(u_mb2_rot_per_iter * float(i)) * p;
        if (u_mb2_abs_x == 1) p.x = abs(p.x);
        if (u_mb2_abs_y == 1) p.y = abs(p.y);
        if (u_mb2_abs_z == 1) p.z = abs(p.z);
        float theta = acos(clamp(p.z / max(r, 1e-9), -1.0, 1.0));
        float phi   = atan(p.y, p.x);
        dr = pow(r, pw - 1.0) * pw * dr + 1.0;
        float zr = pow(r, pw);
        float thetaSph = theta * pw;
        float thetaCyl = atan(length(p.xy), p.z) * pw;
        float thetaFin = mix(thetaSph, thetaCyl, u_mb2_polar_mix);
        phi   *= pw;
        vec3 np = zr * vec3(sin(thetaFin)*cos(phi), sin(thetaFin)*sin(phi), cos(thetaFin));
        if (u_mb2_fold_type == 1 && u_mb2_fold_strength > 0.0) {
            float fs = u_mb2_fold_strength;
            np = clamp(np, -fs, fs) * 2.0 - np;
            dr *= 2.0;
        } else if (u_mb2_fold_type == 2 && u_mb2_fold_strength > 0.0) {
            np = abs(np + u_mb2_fold_strength) - abs(np - u_mb2_fold_strength) - np;
            dr *= 2.0;
        }
        if (u_mb2_julia_mode == 1) {
            p = np + vec3(u_mb2_julia_x, u_mb2_julia_y, u_mb2_julia_z);
        } else {
            p = np + pos;
        }
        trap = min(trap, orbitTrap(p, r));
    }
    return vec2(0.5 * log(max(r, 1e-6)) * r / max(dr, 1e-6) * u_de_multiplier, trap);
}

vec2 pseudoKleinian(vec3 pos) {
    vec3 p = pos;
    float trap = 1e10;
    float dr = 1.0;
    float kscale = u_kl_scale;
    vec3  c = vec3(u_kl_cx, u_kl_cy, u_kl_cz);
    float fl = u_kl_fold_limit;
    float flX = (u_kl_fold_limit_x > 0.0) ? u_kl_fold_limit_x : fl;
    float flY = (u_kl_fold_limit_y > 0.0) ? u_kl_fold_limit_y : fl;
    float flZ = (u_kl_fold_limit_z > 0.0) ? u_kl_fold_limit_z : fl;
    float sr = u_kl_sph_radius;
    vec3 klOff = vec3(u_kl_offset_x, u_kl_offset_y, u_kl_offset_z);
    for (int i = 0; i < u_iterations; i++) {
        if (u_kl_rot_per_iter > 0.0001) p = rotY(u_kl_rot_per_iter * float(i)) * p;
        p.x = clamp(p.x, -flX, flX) * 2.0 - p.x;
        p.y = clamp(p.y, -flY, flY) * 2.0 - p.y;
        p.z = clamp(p.z, -flZ, flZ) * 2.0 - p.z;
        float r2 = dot(p, p);
        float k  = max(sr * sr / r2, 1.0);
        p  *= k;
        dr *= k;
        if (u_kl_julia_mode == 1) {
            p   = p * kscale + vec3(u_julia_x, u_julia_y, u_julia_z) + klOff;
        } else {
            p   = p * kscale + c + klOff;
        }
        dr  = dr * abs(kscale) + 1.0;
        trap = min(trap, orbitTrap(p, abs(kscale)));
        if (r2 > u_bailout * u_bailout) break;
    }
    float d = (length(p) - abs(kscale - 1.0)) / abs(dr);
    float d2 = length(p) / abs(dr);
    return vec2(mix(d, d2, clamp(u_kl_mix_factor, 0.0, 1.0)) * u_de_multiplier, trap);
}

vec3 applySpaceOps(vec3 p) {
    if (u_sph_inv_enabled == 1) {
        vec3 center = vec3(u_sph_inv_cx, u_sph_inv_cy, u_sph_inv_cz);
        vec3 q  = p - center;
        float r2 = dot(q, q);
        float ir = u_sph_inv_radius;
        p = center + q * (ir * ir / max(r2, 1e-6));
    }
    if (u_lattice_fold_enabled == 1) {
        p.x = p.x - u_lattice_fold_x * round(p.x / u_lattice_fold_x);
        p.y = p.y - u_lattice_fold_y * round(p.y / u_lattice_fold_y);
        p.z = p.z - u_lattice_fold_z * round(p.z / u_lattice_fold_z);
        p = abs(p) - vec3(u_lattice_fold_x, u_lattice_fold_y, u_lattice_fold_z) * 0.5;
    }
    if (u_rep_enabled == 1) {
        if (u_rep_cell_x > 0.001) p.x = p.x - u_rep_cell_x * round(p.x / u_rep_cell_x);
        if (u_rep_cell_y > 0.001) p.y = p.y - u_rep_cell_y * round(p.y / u_rep_cell_y);
        if (u_rep_cell_z > 0.001) p.z = p.z - u_rep_cell_z * round(p.z / u_rep_cell_z);
    }
    if (u_fold_mirror_x == 1) p.x = abs(p.x);
    if (u_fold_mirror_y == 1) p.y = abs(p.y);
    if (u_fold_mirror_z == 1) p.z = abs(p.z);
    if (u_twist_amount > 0.0001) {
        float angle;
        if (u_twist_axis == 0) {
            angle = p.y * u_twist_amount;
            float c = cos(angle), s = sin(angle);
            p.xz = vec2(c*p.x - s*p.z, s*p.x + c*p.z);
        } else if (u_twist_axis == 1) {
            angle = p.x * u_twist_amount;
            float c = cos(angle), s = sin(angle);
            p.yz = vec2(c*p.y - s*p.z, s*p.y + c*p.z);
        } else {
            angle = p.z * u_twist_amount;
            float c = cos(angle), s = sin(angle);
            p.xy = vec2(c*p.x - s*p.y, s*p.x + c*p.y);
        }
    }
    if (u_warp_enabled == 1) {
        float f = u_warp_freq;
        float str = u_warp_strength;
        if (u_warp_type == 0) {
            p += str * vec3(sin(f*p.y), sin(f*p.z), sin(f*p.x));
        } else if (u_warp_type == 1) {
            vec3 q = vec3(
                sin(f*p.x + sin(f*p.z)), sin(f*p.y + sin(f*p.x)), sin(f*p.z + sin(f*p.y))
            );
            p += str * q;
        } else {
            float n  = sin(f*p.x)*cos(f*p.y) + sin(f*p.y)*cos(f*p.z) + sin(f*p.z)*cos(f*p.x);
            p += str * vec3(n);
        }
    }
    return p;
}

vec4 qmul(vec4 a, vec4 b) {
    return vec4(
        a.x*b.x - a.y*b.y - a.z*b.z - a.w*b.w,
        a.x*b.y + a.y*b.x + a.z*b.w - a.w*b.z,
        a.x*b.z - a.y*b.w + a.z*b.x + a.w*b.y,
        a.x*b.w + a.y*b.z - a.z*b.y + a.w*b.x
    );
}

vec4 qsq(vec4 q) {
    return vec4(
        q.x*q.x - q.y*q.y - q.z*q.z - q.w*q.w,
        2.0*q.x*q.y,
        2.0*q.x*q.z,
        2.0*q.x*q.w
    );
}

vec2 quaternionJulia(vec3 pos) {
    float cxw = cos(u_qj_slice_rot_xw);
    float sxw = sin(u_qj_slice_rot_xw);
    float cyw = cos(u_qj_slice_rot_yw);
    float syw = sin(u_qj_slice_rot_yw);
    float czw = cos(u_qj_slice_rot_zw);
    float szw = sin(u_qj_slice_rot_zw);
    float wx = u_qj_w_slice;
    float wy = pos.x * sxw + wx * cxw;
    float wz = pos.y * syw + wy * cyw;
    float w  = pos.z * szw + wz * czw;
    vec4 q  = vec4(pos.x * cxw - wx * sxw,
                   pos.y * cyw - wy * syw,
                   pos.z * czw - wz * szw,
                   w);
    vec4 c  = vec4(u_qj_cx, u_qj_cy, u_qj_cz, u_qj_cw);
    vec4 dq = vec4(1.0, 0.0, 0.0, 0.0);
    float bail2 = u_qj_bailout * u_qj_bailout;
    float trap  = 1e10;
    float r2    = 0.0;
    for (int i = 0; i < u_iterations; i++) {
        dq = 2.0 * qmul(q, dq);
        q  = qsq(q) + c;
        r2 = dot(q, q);
        trap = min(trap, orbitTrap(q.xyz, u_qj_bailout * 0.5));
        if (r2 > bail2) break;
    }
    float r  = sqrt(r2);
    float dr = length(dq);
    return vec2(0.5 * r * log(max(r, 1e-9)) / max(dr, 1e-9) * u_de_multiplier, trap);
}

vec2 sceneDist(vec3 p) {
    p = u_rot_mat * p;
    p = applySpaceOps(p);
    if (u_fractal_type == 0) return mandelbox(p);
    if (u_fractal_type == 1) return mengerSponge(p);
    if (u_fractal_type == 2) return sierpinski(p);
    if (u_fractal_type == 3) return octahedronIFS(p);
    if (u_fractal_type == 4) return mandelbulb(p);
    if (u_fractal_type == 6) return quaternionJulia(p);
    return pseudoKleinian(p);
}

vec3 calcNormalEps(vec3 p, float e) {
    if (u_feat_normals_full == 1) {
        vec2 k = vec2(1.0, -1.0);
        return normalize(
            k.xyy * sceneDist(p + k.xyy*e).x +
            k.yyx * sceneDist(p + k.yyx*e).x +
            k.yxy * sceneDist(p + k.yxy*e).x +
            k.xxx * sceneDist(p + k.xxx*e).x
        );
    } else {
        float base = sceneDist(p).x;
        return normalize(vec3(
            sceneDist(p+vec3(e,0,0)).x - base,
            sceneDist(p+vec3(0,e,0)).x - base,
            sceneDist(p+vec3(0,0,e)).x - base
        ));
    }
}

vec3 calcNormal(vec3 p) {
    return calcNormalEps(p, u_normal_eps);
}

float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res  = 1.0;
    float t    = mint;
    float ph   = 1e10;
    int   ns   = clamp(u_shadow_steps, 4, 64);
    for (int i = 0; i < 64; i++) {
        if (i >= ns) break;
        float h = sceneDist(ro + rd * t).x;
        float shadowHitEps = max(0.0001, t * 0.0001);
        if (h < shadowHitEps) return 0.0;
        float y = h * h / (2.0 * ph);
        float d = sqrt(h * h - y * y);
        res = min(res, k * d / max(t - y, 0.0001));
        ph  = h;
        t  += clamp(h, 0.005, 0.3);
        if (t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n, float dist) {
    if (u_feat_ao == 0) return 1.0;
    float occ     = 0.0;
    float scale   = 1.0;
    int   ns      = clamp(u_ao_samples, 1, 16);
    float minStep = max(u_ao_radius * u_ao_step_scale * 0.05, dist * 0.002);
    float stepInc = u_ao_radius * u_ao_step_scale / float(ns);
    for (int i = 0; i < 16; i++) {
        if (i >= ns) break;
        float h = minStep + stepInc * float(i);
        float d = sceneDist(p + n * h).x;
        occ += (h - d) * scale;
        scale *= 0.9;
    }
    return clamp(1.0 - 3.0 * occ * u_ao_strength, 0.0, 1.0);
}

vec3 palette(float t) {
    vec3 a = u_color1, b = u_color2, c = u_color3;
    return a + b * cos(2.0*PI*(c*t + vec3(0.0, 0.33, 0.67)));
}

// ---- Background -------------------------------------------------------
float hash(vec2 p) {
    return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453);
}
vec2 hash2(vec2 p) {
    return fract(sin(vec2(dot(p, vec2(127.1, 311.7)),
                          dot(p, vec2(269.5, 183.3)))) * 43758.5453);
}
float smoothNoise2(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),
               mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * smoothNoise2(p);
        p  = p * 2.1 + vec2(1.7, 9.2);
        a *= 0.5;
    }
    return v;
}

vec2 cubeFaceUV(vec3 rd) {
    vec3 a = abs(rd);
    vec2 uv;
    if (a.x >= a.y && a.x >= a.z) {
        uv = (rd.x > 0.0) ? vec2(-rd.z, rd.y) / a.x : vec2(rd.z, rd.y) / a.x;
        uv += vec2(0.0, 10.0);
    } else if (a.y >= a.x && a.y >= a.z) {
        uv = (rd.y > 0.0) ? vec2(rd.x, -rd.z) / a.y : vec2(rd.x, rd.z) / a.y;
        uv += vec2(20.0, 0.0);
    } else {
        uv = (rd.z > 0.0) ? vec2(rd.x, rd.y) / a.z : vec2(-rd.x, rd.y) / a.z;
        uv += vec2(40.0, 0.0);
    }
    return uv;
}

vec3 starTemperature(float h) {
    if (h < 0.15) return vec3(0.6, 0.7, 1.0);
    if (h < 0.40) return vec3(0.9, 0.95, 1.0);
    if (h < 0.65) return vec3(1.0, 0.97, 0.88);
    if (h < 0.85) return vec3(1.0, 0.85, 0.5);
    return vec3(1.0, 0.55, 0.25);
}

vec2 sphericalUV(vec3 rd) {
    float phi   = atan(rd.z, rd.x);
    float theta = asin(clamp(rd.y, -1.0, 1.0));
    return vec2(phi / (2.0 * PI) + 0.5, theta / PI + 0.5);
}

vec3 starField(vec3 rd, float t) {
    vec2 suv = sphericalUV(rd);
    vec3 col = vec3(0.0);

    float scales[4];
    float thresh[4];
    float seeds[4];
    scales[0] = 120.0; thresh[0] = 0.965; seeds[0] = 0.0;
    scales[1] = 240.0; thresh[1] = 0.960; seeds[1] = 5.3;
    scales[2] = 480.0; thresh[2] = 0.955; seeds[2] = 11.7;
    scales[3] = 800.0; thresh[3] = 0.950; seeds[3] = 23.1;

    for (int i = 0; i < 4; i++) {
        vec2 cell = floor(suv * scales[i] + seeds[i]);
        vec2 loc  = fract(suv * scales[i] + seeds[i]) - 0.5;
        vec2 ji   = hash2(cell + seeds[i]) - 0.5;
        vec2 off  = loc - ji * 0.6;
        float br  = hash(cell + seeds[i] + 3.7);
        if (br >= thresh[i]) {
            float r     = length(off);
            float sharp = mix(6000.0, 2000.0, br);
            float core  = exp(-r * r * sharp);
            float halo  = exp(-r * r * sharp * 0.06) * 0.08;
            float lum   = (core + halo) * (0.5 + br * 1.5);
            float tw    = 1.0 + 0.06 * sin(t * (2.0 + br * 3.0) + br * 47.3);
            vec3  tint  = starTemperature(hash(cell + seeds[i] + 99.1));
            col += tint * lum * tw;
        }
    }
    return col;
}

vec3 milkyWay(vec3 rd) {
    vec2 suv    = sphericalUV(rd);
    float band  = fbm(suv * 0.8 + vec2(3.1, 7.4));
    float band2 = fbm(suv * 1.6 + vec2(11.2, 2.9));
    float mw    = smoothstep(0.35, 0.65, band) * smoothstep(0.30, 0.60, band2);
    float dust  = 1.0 - smoothstep(0.45, 0.55, fbm(suv * 2.5 + vec2(5.5, 1.1)));
    mw *= dust;
    vec3 mwCol  = mix(vec3(0.05, 0.07, 0.15), vec3(0.20, 0.22, 0.35), mw);
    return mwCol * mw * 0.6;
}

vec3 background(vec3 rd, float t) {
    if (u_bg_mode == 0) return u_bg_color1;
    float h = rd.y * 0.5 + 0.5;
    if (u_bg_mode == 1) return mix(u_bg_color2, u_bg_color1, h);
    vec2 cuv = cubeFaceUV(rd);
    if (u_bg_mode == 2) {
        float n1 = smoothNoise2(cuv * 1.5 + vec2(t * 0.05, 0.0));
        float n2 = smoothNoise2(cuv * 3.0 - vec2(0.0, t * 0.03));
        float n  = n1 * 0.6 + n2 * 0.4;
        return mix(u_bg_color2, u_bg_color1, n) * (0.4 + 0.6 * h);
    }
    vec3 base  = mix(u_bg_color2, u_bg_color1, h * h);
    vec3 stars = starField(rd, t);
    vec3 mw    = milkyWay(rd);
    return base + stars + mw;
}

// ---- Single ray -------------------------------------------------------
vec3 castRay(vec2 uv, float t) {
    vec3 ro = u_cam_pos;
    float focalLen = u_fov;
    vec3 rd = normalize(uv.x * u_cam_right + uv.y * u_cam_up + focalLen * u_cam_fwd);

    float totalDist = 0.0;
    float minDist   = 1e10;
    float trap      = 0.0;
    int   steps     = 0;
    bool  hit       = false;
    float absHitEps = u_hit_eps * 0.001;
    int   ms        = clamp(u_max_steps, 4, MAX_STEPS);
    float md        = max(u_max_dist, 1.0);
    float prevD     = 1e10;
    float prevT     = 0.0;
    vec3  p         = ro;
    float d         = 1e10;

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= ms) break;
        p   = ro + rd * totalDist;
        vec2 res = sceneDist(p);
        d   = res.x;
        trap = res.y;
        if (d < minDist) minDist = d;

        if (d < absHitEps) { hit = true; steps = i; break; }
        if (totalDist > md) break;

        float stepD;
        if (u_rm_overrelax == 1) {
            float candD = d * u_overrelax_factor;
            if (candD + d < prevD + 1e-5) {
                stepD = candD;
            } else {
                stepD = d * u_step_scale;
            }
        } else {
            stepD = d * u_step_scale;
        }
        stepD = max(stepD, absHitEps * 0.1);
        prevT = totalDist;
        prevD = d;
        totalDist += stepD;
        steps = i;
    }

    if (hit && prevD > absHitEps) {
        float tLo = prevT;
        float tHi = totalDist;
        for (int b = 0; b < 12; b++) {
            float tMid = (tLo + tHi) * 0.5;
            float dMid = sceneDist(ro + rd * tMid).x;
            if (dMid < absHitEps) {
                tHi = tMid;
            } else {
                tLo = tMid;
            }
        }
        totalDist = tHi;
    }

    vec3 bg       = background(rd, t);
    vec3 col      = bg;
    vec3 lightDir = u_light_dir;

    if (hit) {
        p = ro + rd * totalDist;
        float adaptEps = max(u_normal_eps, absHitEps * 2.0);
        vec3 n   = calcNormalEps(p, adaptEps);

        float diff   = max(dot(n, lightDir), 0.0);
        float spec   = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), u_specular_power);
        float ao     = ambientOcclusion(p, n, totalDist);
        float adaptMint = max(u_shadow_mint, absHitEps * 8.0);
        float shadow = (u_feat_shadows == 1 && u_shadows == 1)
                       ? softShadow(p, lightDir, adaptMint, u_shadow_maxt, u_shadow_soft) : 1.0;

        float colorParam;
        if (u_color_mode == 0)      colorParam = float(steps) / float(u_iterations * 2);
        else if (u_color_mode == 1) colorParam = clamp(sqrt(max(trap, 0.0)) * 0.5, 0.0, 1.0);
        else if (u_color_mode == 2) colorParam = n.x * 0.5 + 0.5;
        else                        colorParam = clamp(totalDist / MAX_DIST, 0.0, 1.0);

        vec3 baseCol = palette(colorParam + u_color_offset + t * u_color_anim_speed);
        float NdotV  = max(dot(n, -rd), 0.0);
        float rim    = pow(1.0 - NdotV, 3.0) * u_rim_strength;
        float emitStr = u_emission * (1.0 + colorParam);

        col  = baseCol * (diff * shadow * (1.0 - u_ambient) + u_ambient) * ao;
        col += spec * u_specular_strength * shadow;
        col += baseCol * emitStr;
        col += palette(colorParam) * rim;
        col += baseCol * u_glow_intensity * 0.04;

        if (u_feat_second_light == 1) {
            vec3  ld2  = u_light2_dir;
            float d2   = max(dot(n, ld2), 0.0);
            float sp2  = pow(max(dot(reflect(-ld2, n), -rd), 0.0), u_specular_power);
            col += (baseCol * d2 + sp2 * u_specular_strength) * u_light2_color * u_light2_strength;
        }

        if (u_feat_subsurface == 1) {
            float sss = u_subsurface * max(-dot(n, lightDir), 0.0) * (0.5 + 0.5 * colorParam);
            col += baseCol * sss;
        }

        if (u_feat_reflection == 1) {
            float fresnelFactor = pow(1.0 - NdotV, max(u_fresnel_power, 0.5));
            vec3 reflDir = reflect(rd, n);
            vec3 envRefl = background(reflDir, t);
            float reflStr = u_reflection * fresnelFactor;
            col = mix(col, envRefl, clamp(reflStr, 0.0, 1.0));
        }

        if (u_feat_fog == 1) {
            float fog = exp(-totalDist * u_fog_density * 0.1);
            vec3 fogC = mix(bg, u_fog_color, clamp(u_fog_density * 0.3, 0.0, 1.0));
            col = mix(fogC, col, fog);
        }
    } else {
        if (u_feat_glow == 1) {
            float falloff  = max(u_glow_falloff, 0.1);
            float gd       = max(minDist, 0.0001);
            float glow     = exp(-gd * falloff * u_glow_radius);
            float colorT   = clamp(-log(gd * falloff + 0.001) * 0.15, 0.0, 1.0) + t * 0.04;
            vec3  glowCol  = palette(colorT);
            col += glowCol * glow * u_glow_intensity * 0.8;
        }
    }
    return col;
}

void main() {
    float aspect  = u_resolution.x / u_resolution.y;
    float t       = u_animate == 1 ? u_time * u_anim_speed : 0.0;
    vec2  pixSize = 1.0 / u_resolution;

    vec3 col = vec3(0.0);
    if (u_aa_samples <= 1) {
        // TAA: субпиксельный джиттер (u_taa_jitter == 0 когда TAA выключен)
        vec2 jUV = v_uv + u_taa_jitter;
        col = castRay(vec2(jUV.x * aspect, jUV.y), t);
    } else if (u_aa_samples == 2) {
        const vec2 o[4] = vec2[4](vec2(-0.25,-0.25), vec2( 0.25,-0.25),
                                   vec2(-0.25, 0.25), vec2( 0.25, 0.25));
        for (int i = 0; i < 4; i++) {
            vec2 uv = vec2((v_uv.x + o[i].x * pixSize.x * 2.0) * aspect,
                            v_uv.y + o[i].y * pixSize.y * 2.0);
            col += castRay(uv, t);
        }
        col *= 0.25;
    } else {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                vec2 off = vec2(float(x), float(y)) * 0.33;
                vec2 uv  = vec2((v_uv.x + off.x * pixSize.x) * aspect,
                                 v_uv.y + off.y * pixSize.y);
                col += castRay(uv, t);
            }
        }
        col /= 9.0;
    }

    fragColor = vec4(col, 1.0);
}
