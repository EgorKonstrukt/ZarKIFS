import json
import math
import os
import sys
import threading
import time
from pathlib import Path

import moderngl
import moderngl_window as mglw
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QColor, QPalette, QIcon
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QScrollArea,
    QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QSlider, QRadioButton, QCheckBox, QPushButton,
    QButtonGroup, QGroupBox, QColorDialog, QTabWidget,
    QFileDialog, QInputDialog, QMessageBox,
)
from moderngl_window.conf import settings as mglw_settings

APP_VERSION = "1.4.1"

VERT_SHADER = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

FRAG_SHADER = """
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
    if (u_feat_orbit_trap == 0) return dot(p, p) / (s * s);
    if (u_orbit_trap_type == 0) {
        return dot(p, p) / (s * s);
    } else if (u_orbit_trap_type == 1) {
        return abs(p.y) / (abs(s) + 0.0001);
    } else if (u_orbit_trap_type == 2) {
        return max(abs(p.x), max(abs(p.y), abs(p.z))) / (abs(s) + 0.0001);
    } else {
        float r = length(p.xz);
        return length(vec2(r - abs(s) * 0.4, p.y)) / (abs(s) * 0.3 + 0.0001);
    }
}

vec2 mandelbox(vec3 pos) {
    vec3 p = pos;
    float trap = 1e10;
    float dr = 1.0;
    float foldX = (u_mb_fold_x > 0.001) ? u_mb_fold_x : u_mb_fold_limit;
    float foldY = (u_mb_fold_y > 0.001) ? u_mb_fold_y : u_mb_fold_limit;
    float foldZ = (u_mb_fold_z > 0.001) ? u_mb_fold_z : u_mb_fold_limit;
    float sphIn = u_mb_sphere_inner;
    float sphOut = u_mb_sphere_outer * u_mb_fixed_radius;
    for (int i = 0; i < u_iterations; i++) {
        if (u_mb_rot_per_iter > 0.0001) p = rotY(u_mb_rot_per_iter * float(i)) * p;
        if (u_mb_fold_mode == 0) {
            p.x = clamp(p.x, -foldX, foldX) * 2.0 - p.x;
            p.y = clamp(p.y, -foldY, foldY) * 2.0 - p.y;
            p.z = clamp(p.z, -foldZ, foldZ) * 2.0 - p.z;
        } else if (u_mb_fold_mode == 1) {
            p.x = abs(p.x + foldX) - abs(p.x - foldX) - p.x;
            p.y = abs(p.y + foldY) - abs(p.y - foldY) - p.y;
            p.z = abs(p.z + foldZ) - abs(p.z - foldZ) - p.z;
        } else {
            p.x = sin(p.x * PI / (2.0 * foldX)) * foldX;
            p.y = sin(p.y * PI / (2.0 * foldY)) * foldY;
            p.z = sin(p.z * PI / (2.0 * foldZ)) * foldZ;
        }
        float r2 = dot(p, p);
        if (r2 < sphIn) {
            float k = sphOut / sphIn;
            p *= k; dr *= k;
        } else if (r2 < sphOut) {
            float k = sphOut / r2;
            p *= k; dr *= k;
        }
        if (u_mb_julia_mode == 1) {
            p = p * u_scale + vec3(u_julia_x, u_julia_y, u_julia_z);
        } else {
            p = p * u_scale + pos * (1.0 - u_scale) * 0.1;
        }
        dr = dr * abs(u_scale) + 1.0;
        trap = min(trap, orbitTrap(p, u_mb_color_scale));
        if (dot(p,p) > u_bailout * u_bailout) break;
    }
    return vec2(length(p) / abs(dr) * u_de_multiplier, trap);
}

vec2 mengerSponge(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    float ms = u_ms_scale;
    float mo = u_ms_offset;
    float sy = (u_ms_scale_y > 0.001) ? u_ms_scale_y : ms;
    float sz = (u_ms_scale_z > 0.001) ? u_ms_scale_z : ms;
    for (int i = 0; i < u_iterations; i++) {
        if (u_ms_twist > 0.001)  p = rotY(u_ms_twist) * p;
        if (u_ms_rot_x > 0.001) p = rotX(u_ms_rot_x) * p;
        if (u_ms_rot_z > 0.001) p = rotZ(u_ms_rot_z) * p;
        p = abs(p);
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;
        p.x = p.x * ms - mo;
        p.y = p.y * sy - mo;
        p.z = p.z * sz - mo;
        p.z += mo * clamp(p.z / mo * 0.5 + 0.5, 0.0, 1.0) * u_ms_cross_width;
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
    float scale = 1.0;
    float trap = 1e10;
    for (int i = 0; i < u_iterations; i++) {
        if (u_si_twist > 0.001)  p = rotY(u_si_twist) * p;
        if (u_si_rot_x > 0.001) p = rotX(u_si_rot_x) * p;
        if (u_si_rot_z > 0.001) p = rotZ(u_si_rot_z) * p;
        vec3 closest = A;
        float d = dot(p - A, p - A);
        float db = dot(p - B, p - B);
        float dc = dot(p - C, p - C);
        float dd = dot(p - D, p - D);
        if (db < d) { closest = B; d = db; }
        if (dc < d) { closest = C; d = dc; }
        if (dd < d) { closest = D; }
        p = u_si_fold_bias * p - closest * (u_si_fold_bias - 1.0);
        scale *= u_si_fold_bias;
        trap = min(trap, orbitTrap(p, scale));
    }
    return vec2(sdTetra(p, scale) / scale * u_de_multiplier, trap);
}

vec2 octahedronIFS(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    float IFS_SCALE = u_oc_ifs_scale;
    vec3 off = vec3(
        (u_oc_offset_x > 0.0001) ? u_oc_offset_x : u_offset_x * u_oc_offset_uni,
        (u_oc_offset_y > 0.0001) ? u_oc_offset_y : u_offset_y * u_oc_offset_uni,
        (u_oc_offset_z > 0.0001) ? u_oc_offset_z : u_offset_z * u_oc_offset_uni
    );
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
        p = IFS_SCALE * p - off * (IFS_SCALE - 1.0);
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
        float theta = acos(clamp(p.z / r, -1.0, 1.0));
        float phi   = atan(p.y, p.x);
        dr = pow(r, pw - 1.0) * pw * dr + 1.0;
        float zr = pow(r, pw);
        theta *= pw;
        phi   *= pw;
        vec3 np = zr * vec3(sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta));
        if (u_mb2_fold_type == 1) {
            float fs = u_mb2_fold_strength;
            np = clamp(np, -fs, fs) * 2.0 - np;
        } else if (u_mb2_fold_type == 2) {
            np = abs(np + u_mb2_fold_strength) - abs(np - u_mb2_fold_strength) - np;
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
    float sr = u_kl_sph_radius;
    for (int i = 0; i < u_iterations; i++) {
        if (u_kl_rot_per_iter > 0.0001) p = rotY(u_kl_rot_per_iter * float(i)) * p;
        p = clamp(p, -fl, fl) * 2.0 - p;
        float r2 = dot(p, p);
        float k  = max(sr * sr / r2, 1.0);
        p  *= k;
        dr *= k;
        p   = p * kscale + c;
        dr  = dr * abs(kscale) + 1.0;
        trap = min(trap, orbitTrap(p, abs(kscale)));
        if (r2 > u_bailout * u_bailout) break;
    }
    float d = (length(p) - abs(kscale - 1.0)) / abs(dr);
    return vec2(mix(d, d * 0.5, u_kl_mix_factor) * u_de_multiplier, trap);
}

vec3 applySpaceOps(vec3 p) {
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

vec2 sceneDist(vec3 p) {
    mat3 rx = rotX(u_rot_x), ry = rotY(u_rot_y), rz = rotZ(u_rot_z);
    p = rz * ry * rx * p;
    p = applySpaceOps(p);
    if (u_fractal_type == 0) return mandelbox(p);
    if (u_fractal_type == 1) return mengerSponge(p);
    if (u_fractal_type == 2) return sierpinski(p);
    if (u_fractal_type == 3) return octahedronIFS(p);
    if (u_fractal_type == 4) return mandelbulb(p);
    return pseudoKleinian(p);
}

vec3 calcNormal(vec3 p) {
    float e = u_normal_eps;
    if (u_feat_normals_full == 1) {
        return normalize(vec3(
            sceneDist(p+vec3(e,0,0)).x - sceneDist(p-vec3(e,0,0)).x,
            sceneDist(p+vec3(0,e,0)).x - sceneDist(p-vec3(0,e,0)).x,
            sceneDist(p+vec3(0,0,e)).x - sceneDist(p-vec3(0,0,e)).x
        ));
    } else {
        float base = sceneDist(p).x;
        return normalize(vec3(
            sceneDist(p+vec3(e,0,0)).x - base,
            sceneDist(p+vec3(0,e,0)).x - base,
            sceneDist(p+vec3(0,0,e)).x - base
        ));
    }
}

float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    int ns = clamp(u_shadow_steps, 4, 64);
    for (int i = 0; i < 64; i++) {
        if (i >= ns) break;
        float h = sceneDist(ro + rd*t).x;
        if (h < 0.0001) return 0.0;
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.2);
        if (t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n) {
    if (u_feat_ao == 0) return 1.0;
    float occ = 0.0, scale = 1.0;
    int ns = clamp(u_ao_samples, 1, 16);
    float stepBase = u_ao_radius * u_ao_step_scale;
    for (int i = 0; i < 16; i++) {
        if (i >= ns) break;
        float h = stepBase * 0.1 + stepBase * float(i) / float(ns);
        float d = sceneDist(p + n*h).x;
        occ += (h - d) * scale;
        scale *= 0.95;
    }
    return clamp(1.0 - 3.0*occ * u_ao_strength, 0.0, 1.0);
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
float noise2(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f*f*(3.0-2.0*f);
    return mix(mix(hash(i),hash(i+vec2(1,0)),f.x),
               mix(hash(i+vec2(0,1)),hash(i+vec2(1,1)),f.x),f.y);
}
float fbm(vec2 p) {
    float v = 0.0, a = 0.5;
    for (int i = 0; i < 5; i++) {
        v += a * noise2(p);
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
        float n1 = noise2(cuv * 1.5 + vec2(t * 0.05, 0.0));
        float n2 = noise2(cuv * 3.0 - vec2(0.0, t * 0.03));
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
    float hitEps    = u_hit_eps * 0.001;
    int   ms        = clamp(u_max_steps, 4, MAX_STEPS);
    float md        = max(u_max_dist, 1.0);
    float prevD     = 1e10;

    for (int i = 0; i < MAX_STEPS; i++) {
        if (i >= ms) break;
        vec3 p   = ro + rd * totalDist;
        vec2 res = sceneDist(p);
        float d  = res.x;
        trap     = res.y;
        minDist  = min(minDist, d);
        if (d < hitEps) { hit = true; steps = i; break; }
        if (totalDist > md) break;
        float stepD = d * u_step_scale;
        if (u_rm_overrelax == 1) {
            float candD = d * u_overrelax_factor;
            if (candD < prevD * 2.0) stepD = candD;
        }
        totalDist += stepD;
        prevD = d;
        steps = i;
    }

    vec3 bg = background(rd, t);
    vec3 col;
    if (hit) {
        vec3 p   = ro + rd * totalDist;
        vec3 n   = calcNormal(p);
        vec3 lightDir = normalize(u_light_dir);

        float diff   = max(dot(n, lightDir), 0.0);
        float spec   = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), u_specular_power);
        float ao     = ambientOcclusion(p, n);
        float shadow = (u_feat_shadows == 1 && u_shadows == 1)
                       ? softShadow(p, lightDir, u_shadow_mint, u_shadow_maxt, u_shadow_soft) : 1.0;

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
            vec3 ld2   = normalize(u_light2_dir);
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
        col = bg;
        if (u_feat_glow == 1) {
            float falloff = max(u_glow_falloff, 0.1);

            // Single smooth exponential envelope — no discontinuous pieces
            float d       = max(minDist, 0.0001);
            float glow    = exp(-d * falloff * u_glow_radius);

            // Use minDist continuously for color to avoid step-count banding
            // log(d) maps [near..far] to a smooth continuous range
            float colorT  = clamp(-log(d * falloff + 0.001) * 0.15, 0.0, 1.0)
                            + t * 0.04;
            vec3  glowCol = palette(colorT);

            col += glowCol * glow * u_glow_intensity * 0.8;
        }
    }
    return col;
}

void main() {
    float aspect  = u_resolution.x / u_resolution.y;
    float t       = u_animate == 1 ? u_time * u_anim_speed : 0.0;
    vec2  pixSize = vec2(1.0 / u_resolution.x, 1.0 / u_resolution.y);

    vec3 col = vec3(0.0);
    if (u_aa_samples <= 1) {
        col = castRay(vec2(v_uv.x * aspect, v_uv.y), t);
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
"""

POST_VERT = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

POST_FRAG = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_scene;
uniform float u_gamma;
uniform float u_exposure;
uniform float u_saturation;

void main() {
    vec3 col = texture(u_scene, v_uv).rgb;

    col *= u_exposure;

    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation);

    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1)));
    fragColor = vec4(col, 1.0);
}
"""

DEBUG_OVERLAY_VERT = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

DEBUG_OVERLAY_FRAG = """
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
"""

_debug_overlay_enabled = False

class FractalParams:
    def __init__(self):
        self.iterations   = 8
        self.scale        = 3.0
        self.fold_x       = True
        self.fold_y       = True
        self.fold_z       = True
        self.rot_x        = 0.0
        self.rot_y        = 0.0
        self.rot_z        = 0.0
        self.offset_x     = 2.0
        self.offset_y     = 2.0
        self.offset_z     = 2.0
        self.julia_x      = -0.5
        self.julia_y      = -0.5
        self.julia_z      = -0.5
        self.fractal_type = 0
        self.bailout      = 20.0
        self.min_dist     = 1.0
        self.fog_density  = 0.5
        self.color1       = (0.5, 0.5, 0.5)
        self.color2       = (0.5, 0.5, 0.5)
        self.color3       = (1.0, 1.0, 1.0)
        self.color_mode   = 0
        self.ao_strength  = 1.0
        self.shadow_soft  = 8.0
        self.shadows      = True
        self.glow         = 5.0
        self.cam_pos      = [0.0, 0.0, 5.0]
        self.cam_yaw      = 0.0
        self.cam_pitch    = 0.0
        self.cam_roll     = 0.0
        self.animate      = True
        self.anim_speed   = 2.5
        # --- Mandelbox fine-tune ---
        self.mb_fold_limit    = 1.0
        self.mb_sphere_inner  = 0.25
        self.mb_sphere_outer  = 1.0
        self.mb_fixed_radius  = 1.0
        self.mb_color_scale   = 0.5
        self.mb_rot_per_iter  = 0.0
        self.mb_fold_mode     = 0   # 0=clamp 1=abs 2=sin
        # --- Menger Sponge fine-tune ---
        self.ms_cross_width   = 1.0
        self.ms_scale         = 3.0
        self.ms_offset        = 2.0
        self.ms_twist         = 0.0
        self.ms_sharpness     = 1.0
        # --- Sierpinski fine-tune ---
        self.si_vertex_spread = 1.0
        self.si_fold_bias     = 2.0
        self.si_twist         = 0.0
        self.si_squash        = 1.0
        self.si_vertex_jitter = 0.0
        # --- Octahedron IFS fine-tune ---
        self.oc_ifs_scale     = 2.0
        self.oc_twist         = 0.0
        self.oc_sharpness     = 1.0
        self.oc_offset_uni    = 1.0
        self.oc_fold_amount   = 0.0
        self.oc_offset_x      = 0.0
        self.oc_offset_y      = 0.0
        self.oc_offset_z      = 0.0
        self.oc_rot_x         = 0.0
        self.oc_rot_z         = 0.0
        # --- Mandelbulb fine-tune ---
        self.mb2_power        = 8.0
        self.mb2_bailout      = 4.0
        self.mb2_julia_x      = 0.0
        self.mb2_julia_y      = 0.0
        self.mb2_julia_z      = 0.0
        self.mb2_julia_mode   = 0
        self.mb2_fold_strength= 0.0
        self.mb2_fold_type    = 0
        # --- Pseudo-Kleinian fine-tune ---
        self.kl_scale         = 1.5
        self.kl_cx            = 0.0
        self.kl_cy            = 0.0
        self.kl_cz            = 0.0
        self.kl_fold_limit    = 1.0
        self.kl_sph_radius    = 0.5
        self.kl_rot_per_iter  = 0.0
        self.kl_mix_factor    = 0.0
        # --- Mandelbox per-axis fold ---
        self.mb_fold_x        = 0.0
        self.mb_fold_y        = 0.0
        self.mb_fold_z        = 0.0
        self.mb_julia_mode    = 1
        # --- Menger per-axis / rotation ---
        self.ms_rot_x         = 0.0
        self.ms_rot_z         = 0.0
        self.ms_scale_y       = 0.0
        self.ms_scale_z       = 0.0
        # --- Sierpinski per-axis rotation ---
        self.si_rot_x         = 0.0
        self.si_rot_z         = 0.0
        # --- Global space operators ---
        self.warp_enabled     = False
        self.warp_strength    = 0.3
        self.warp_freq        = 1.0
        self.warp_type        = 0
        self.twist_axis       = 0
        self.twist_amount     = 0.0
        self.fold_mirror_x    = False
        self.fold_mirror_y    = False
        self.fold_mirror_z    = False
        self.rep_enabled      = False
        self.rep_cell_x       = 4.0
        self.rep_cell_y       = 4.0
        self.rep_cell_z       = 4.0
        # --- Orbit trap ---
        self.orbit_trap_type  = 0   # 0=sphere 1=plane 2=cube 3=torus
        # --- DE multiplier ---
        self.de_multiplier    = 1.0
        # --- Light primary ---
        self.light_x          = 1.0
        self.light_y          = 2.0
        self.light_z          = 1.5
        self.specular_power   = 32.0
        self.specular_strength= 0.3
        self.ambient          = 0.2
        self.subsurface       = 0.0
        self.fresnel_power    = 5.0
        # --- Second light ---
        self.light2_x         = -1.0
        self.light2_y         = -0.5
        self.light2_z         = 1.0
        self.light2_r         = 0.2
        self.light2_g         = 0.3
        self.light2_b         = 0.5
        self.light2_strength  = 0.0
        # --- Color animation ---
        self.color_anim_speed = 0.05
        self.color_offset     = 0.0
        # --- Raymarching ---
        self.step_scale       = 0.5
        self.normal_eps       = 0.001
        self.reflection       = 0.0
        self.max_steps        = 200
        self.max_dist         = 100.0
        self.hit_eps          = 1.0
        self.shadow_steps     = 32
        self.shadow_mint      = 0.02
        self.shadow_maxt      = 10.0
        self.ao_step_scale    = 1.0
        self.rm_overrelax     = False
        self.overrelax_factor = 1.6
        # --- FOV ---
        self.fov              = 1.5
        # --- AO ---
        self.ao_radius        = 0.12
        self.ao_samples       = 5
        # --- Fog ---
        self.fog_color        = (0.02, 0.03, 0.08)
        # --- Post-process ---
        self.gamma            = 2.2
        self.exposure         = 1.0
        self.saturation       = 1.0
        # --- Performance feature flags ---
        self.feat_ao           = True
        self.feat_shadows      = True
        self.feat_normals_full = True
        self.feat_second_light = True
        self.feat_fog          = True
        self.feat_glow         = True
        self.feat_reflection   = True
        self.feat_subsurface   = True
        self.feat_orbit_trap   = True
        # --- Glow ---
        self.glow_intensity   = 5.0
        self.glow_falloff     = 8.0
        self.glow_radius      = 1.0
        self.rim_strength     = 0.4
        self.emission         = 0.2
        # --- Background ---
        self.bg_color1        = (0.02, 0.03, 0.08)
        self.bg_color2        = (0.0,  0.0,  0.0)
        self.bg_mode          = 1   # 0=flat 1=gradient 2=nebula 3=starfield
        # --- Anti-aliasing ---
        self.aa_samples       = 1   # 1=off 2=4xRGSS 3=9x
        # --- Screenshot ---
        self.screenshot_requested = False
        self.player_mode = False

_params = FractalParams()

_ZKF_VERSION = 1
_ZKS_VERSION = 1
_SAVES_DIR = Path(__file__).parent / "saves"

_FRACTAL_FIELDS = [
    'iterations', 'scale', 'fold_x', 'fold_y', 'fold_z',
    'rot_x', 'rot_y', 'rot_z', 'offset_x', 'offset_y', 'offset_z',
    'julia_x', 'julia_y', 'julia_z', 'fractal_type', 'bailout', 'min_dist',
    'fog_density', 'color1', 'color2', 'color3', 'color_mode',
    'ao_strength', 'shadow_soft', 'shadows', 'glow', 'animate', 'anim_speed',
    'mb_fold_limit', 'mb_sphere_inner', 'mb_sphere_outer', 'mb_fixed_radius',
    'mb_color_scale', 'mb_rot_per_iter', 'mb_fold_mode',
    'ms_cross_width', 'ms_scale', 'ms_offset', 'ms_twist', 'ms_sharpness',
    'si_vertex_spread', 'si_fold_bias', 'si_twist', 'si_squash', 'si_vertex_jitter',
    'oc_ifs_scale', 'oc_twist', 'oc_sharpness', 'oc_offset_uni', 'oc_fold_amount',
    'oc_offset_x', 'oc_offset_y', 'oc_offset_z', 'oc_rot_x', 'oc_rot_z',
    'mb2_power', 'mb2_bailout', 'mb2_julia_x', 'mb2_julia_y', 'mb2_julia_z',
    'mb2_julia_mode', 'mb2_fold_strength', 'mb2_fold_type',
    'kl_scale', 'kl_cx', 'kl_cy', 'kl_cz', 'kl_fold_limit', 'kl_sph_radius',
    'kl_rot_per_iter', 'kl_mix_factor',
    'mb_fold_x', 'mb_fold_y', 'mb_fold_z', 'mb_julia_mode',
    'ms_rot_x', 'ms_rot_z', 'ms_scale_y', 'ms_scale_z',
    'si_rot_x', 'si_rot_z',
    'warp_enabled', 'warp_strength', 'warp_freq', 'warp_type',
    'twist_axis', 'twist_amount',
    'fold_mirror_x', 'fold_mirror_y', 'fold_mirror_z',
    'rep_enabled', 'rep_cell_x', 'rep_cell_y', 'rep_cell_z',
    'orbit_trap_type', 'de_multiplier',
    'light_x', 'light_y', 'light_z', 'specular_power', 'specular_strength',
    'ambient', 'subsurface', 'fresnel_power',
    'light2_x', 'light2_y', 'light2_z', 'light2_r', 'light2_g', 'light2_b', 'light2_strength',
    'color_anim_speed', 'color_offset',
    'step_scale', 'normal_eps', 'reflection', 'max_steps', 'max_dist', 'hit_eps',
    'shadow_steps', 'shadow_mint', 'shadow_maxt', 'ao_step_scale',
    'rm_overrelax', 'overrelax_factor', 'fov', 'ao_radius', 'ao_samples',
    'fog_color', 'gamma', 'exposure', 'saturation',
    'feat_ao', 'feat_shadows', 'feat_normals_full', 'feat_second_light',
    'feat_fog', 'feat_glow', 'feat_reflection', 'feat_subsurface', 'feat_orbit_trap',
    'glow_intensity', 'glow_falloff', 'glow_radius', 'rim_strength', 'emission',
    'bg_color1', 'bg_color2', 'bg_mode', 'aa_samples',
]
_SESSION_EXTRA_FIELDS = ['cam_pos', 'cam_yaw', 'cam_pitch', 'cam_roll', 'player_mode']

def _params_to_dict(fields):
    out = {}
    for f in fields:
        v = getattr(_params, f)
        if isinstance(v, tuple):
            v = list(v)
        elif isinstance(v, list):
            v = list(v)
        out[f] = v
    return out

def _dict_to_params(d):
    for k, v in d.items():
        if not hasattr(_params, k):
            continue
        cur = getattr(_params, k)
        if isinstance(cur, tuple):
            v = tuple(v)
        setattr(_params, k, v)

def save_zkf(path, name):
    _SAVES_DIR.mkdir(exist_ok=True)
    data = {'version': _ZKF_VERSION, 'name': name, 'params': _params_to_dict(_FRACTAL_FIELDS)}
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

def load_zkf(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def save_zks(path):
    fields = _FRACTAL_FIELDS + _SESSION_EXTRA_FIELDS
    data = {
        'version': _ZKS_VERSION,
        'params': _params_to_dict(fields),
        'player': {
            'vel': list(_player_state.vel),
            'on_ground': _player_state.on_ground,
        },
    }
    Path(path).write_text(json.dumps(data, indent=2), encoding='utf-8')

def load_zks(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    _dict_to_params(data['params'])
    ps = data.get('player', {})
    if 'vel' in ps:
        _player_state.vel = ps['vel']
    if 'on_ground' in ps:
        _player_state.on_ground = ps['on_ground']

INTERP_FLOAT_ATTRS = [
    'scale', 'rot_x', 'rot_y', 'rot_z',
    'offset_x', 'offset_y', 'offset_z',
    'julia_x', 'julia_y', 'julia_z',
    'bailout', 'min_dist', 'fog_density',
    'ao_strength', 'ao_radius', 'shadow_soft', 'glow',
    'anim_speed', 'fov', 'de_multiplier',
    'glow_intensity', 'glow_falloff', 'glow_radius', 'rim_strength', 'emission',
    'mb_fold_limit', 'mb_sphere_inner', 'mb_sphere_outer', 'mb_fixed_radius', 'mb_color_scale',
    'mb_rot_per_iter',
    'ms_cross_width', 'ms_scale', 'ms_offset', 'ms_twist', 'ms_sharpness',
    'si_vertex_spread', 'si_fold_bias', 'si_twist', 'si_squash', 'si_vertex_jitter',
    'oc_ifs_scale', 'oc_twist', 'oc_sharpness', 'oc_offset_uni', 'oc_fold_amount',
    'oc_offset_x', 'oc_offset_y', 'oc_offset_z', 'oc_rot_x', 'oc_rot_z',
    'mb2_power', 'mb2_bailout', 'mb2_julia_x', 'mb2_julia_y', 'mb2_julia_z', 'mb2_fold_strength',
    'kl_scale', 'kl_cx', 'kl_cy', 'kl_cz', 'kl_fold_limit', 'kl_sph_radius', 'kl_rot_per_iter', 'kl_mix_factor',
    'mb_fold_x', 'mb_fold_y', 'mb_fold_z',
    'ms_rot_x', 'ms_rot_z', 'ms_scale_y', 'ms_scale_z',
    'si_rot_x', 'si_rot_z',
    'warp_strength', 'warp_freq', 'twist_amount',
    'rep_cell_x', 'rep_cell_y', 'rep_cell_z',
    'light_x', 'light_y', 'light_z',
    'specular_power', 'specular_strength', 'ambient', 'subsurface', 'fresnel_power',
    'light2_x', 'light2_y', 'light2_z', 'light2_r', 'light2_g', 'light2_b', 'light2_strength',
    'color_anim_speed', 'color_offset',
    'step_scale', 'normal_eps', 'reflection',
    'max_dist', 'hit_eps', 'shadow_mint', 'shadow_maxt', 'ao_step_scale', 'overrelax_factor',
    'gamma', 'exposure', 'saturation',
]
INTERP_COLOR_ATTRS = ['color1', 'color2', 'color3', 'bg_color1', 'bg_color2', 'fog_color']

class PresetInterpolator:
    DURATION_MS  = 1200
    TICK_MS      = 16
    EASE_EXP     = 3.0

    def __init__(self):
        self._timer   = QTimer()
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._src_f:   dict = {}
        self._dst_f:   dict = {}
        self._src_col: dict = {}
        self._dst_col: dict = {}
        self._elapsed: float = 0.0
        self._gui_sync_cb = None

    def set_gui_sync(self, cb):
        self._gui_sync_cb = cb

    def _ease(self, t: float) -> float:
        t = max(0.0, min(1.0, t))
        if t < 0.5:
            return 0.5 * (2.0 * t) ** self.EASE_EXP
        return 1.0 - 0.5 * (2.0 * (1.0 - t)) ** self.EASE_EXP

    def start(self, target: dict):
        self._timer.stop()
        self._elapsed = 0.0
        for a in INTERP_FLOAT_ATTRS:
            self._src_f[a] = float(getattr(_params, a, 0.0))
            self._dst_f[a] = float(target.get(a, self._src_f[a]))
        for a in INTERP_COLOR_ATTRS:
            src = getattr(_params, a, (0.0, 0.0, 0.0))
            self._src_col[a] = tuple(float(c) for c in src)
            self._dst_col[a] = tuple(float(c) for c in target.get(a, self._src_col[a]))
        self._timer.start()

    def _tick(self):
        self._elapsed += self.TICK_MS
        alpha = self._ease(self._elapsed / self.DURATION_MS)
        for a in INTERP_FLOAT_ATTRS:
            s, d = self._src_f[a], self._dst_f[a]
            setattr(_params, a, s + (d - s) * alpha)
        for a in INTERP_COLOR_ATTRS:
            s, d = self._src_col[a], self._dst_col[a]
            setattr(_params, a, tuple(s[i] + (d[i] - s[i]) * alpha for i in range(3)))
        if self._elapsed >= self.DURATION_MS:
            self._finalize()

    def _finalize(self):
        self._timer.stop()
        for a in INTERP_FLOAT_ATTRS:
            setattr(_params, a, self._dst_f[a])
        for a in INTERP_COLOR_ATTRS:
            setattr(_params, a, self._dst_col[a])
        if self._gui_sync_cb:
            self._gui_sync_cb()

_interpolator = PresetInterpolator()


class InfiniteEvolution:
    TICK_MS = 16

    CHANNELS = [
        ('scale',           -2.8,  2.8,   0.3),
        ('julia_x',        -15.0, 15.0,   0.5),
        ('julia_y',        -15.0, 15.0,   0.5),
        ('julia_z',        -15.0, 15.0,   0.5),
        ('mb_fold_limit',    0.2,  2.5,   0.4),
        ('mb_sphere_inner',  0.05, 0.9,   0.3),
        ('mb_sphere_outer',  0.2,  3.5,   0.25),
        ('mb_rot_per_iter',  0.0,  0.45,  0.2),
        ('rot_x',            0.0,  6.28,  0.15),
        ('rot_y',            0.0,  6.28,  0.15),
        ('rot_z',            0.0,  6.28,  0.1),
        ('ms_scale',         2.1,  4.8,   0.2),
        ('ms_offset',        1.1,  3.8,   0.25),
        ('ms_twist',         0.0,  0.28,  0.15),
        ('si_vertex_spread', 0.3,  2.8,   0.3),
        ('si_fold_bias',     1.3,  3.8,   0.2),
        ('si_squash',        0.3,  2.8,   0.2),
        ('si_twist',         0.0,  0.28,  0.1),
        ('oc_ifs_scale',     1.3,  3.8,   0.2),
        ('oc_twist',         0.0,  0.28,  0.15),
        ('oc_fold_amount',   0.0,  1.0,   0.25),
        ('oc_offset_uni',    0.15, 2.8,   0.2),
        ('glow_intensity',   0.5, 18.0,   0.15),
        ('emission',         0.0,  2.8,   0.1),
        ('rim_strength',     0.0,  2.5,   0.12),
        ('color_offset',     0.0,  1.0,   0.08),
        ('fog_density',      0.0,  2.0,   0.1),
        ('ambient',          0.05, 0.6,   0.1),
        ('specular_power',   4.0, 80.0,   0.1),
        ('de_multiplier',    0.3,  2.8,   0.15),
    ]

    COLOR_CHANNELS = [
        ('color1', 0.08),
        ('color2', 0.06),
        ('color3', 0.05),
        ('bg_color1', 0.04),
        ('bg_color2', 0.03),
        ('fog_color', 0.05),
    ]

    def __init__(self):
        self._active     = False
        self._speed      = 1.0
        self._mutation   = 0.5
        self._timer      = QTimer()
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)
        self._phases_f   = {}
        self._freqs_f    = {}
        self._amps_f     = {}
        self._centers_f  = {}
        self._phases_c   = {}
        self._freqs_c    = {}
        self._centers_c  = {}
        self._t          = 0.0
        self._status_cb  = None
        self._randomize_all()

    def _rand(self, lo=0.0, hi=1.0):
        import random
        return lo + (hi - lo) * random.random()

    def _randomize_all(self):
        import random
        for attr, lo, hi, base_freq in self.CHANNELS:
            cur = float(getattr(_params, attr, (lo + hi) * 0.5))
            cur = max(lo, min(hi, cur))
            self._centers_f[attr] = cur
            half = (hi - lo) * 0.5 * max(0.05, min(1.0, self._mutation))
            self._amps_f[attr]    = self._rand(half * 0.2, half)
            self._freqs_f[attr]   = base_freq * self._rand(0.4, 2.2)
            self._phases_f[attr]  = self._rand(0.0, 6.283)
        for attr, base_freq in self.COLOR_CHANNELS:
            cur = getattr(_params, attr, (0.5, 0.5, 0.5))
            self._centers_c[attr]  = tuple(float(c) for c in cur)
            self._freqs_c[attr]    = base_freq * self._rand(0.4, 2.0)
            self._phases_c[attr]   = tuple(self._rand(0.0, 6.283) for _ in range(3))

    def _tick(self):
        dt = self.TICK_MS * 0.001 * self._speed
        self._t += dt
        t = self._t
        for attr, lo, hi, _ in self.CHANNELS:
            center = self._centers_f[attr]
            amp    = self._amps_f[attr]
            freq   = self._freqs_f[attr]
            phase  = self._phases_f[attr]
            val    = center + amp * math.sin(freq * t + phase)
            val    = max(lo, min(hi, val))
            setattr(_params, attr, val)
        drift_rate = dt * 0.004
        for attr, lo, hi, _ in self.CHANNELS:
            self._centers_f[attr] += math.sin(
                self._freqs_f[attr] * 0.17 * t + self._phases_f[attr] * 1.3
            ) * drift_rate * (hi - lo)
            self._centers_f[attr] = max(
                lo + self._amps_f[attr],
                min(hi - self._amps_f[attr], self._centers_f[attr])
            )
        for attr, base_freq in self.COLOR_CHANNELS:
            center = self._centers_c[attr]
            freq   = self._freqs_c[attr]
            phases = self._phases_c[attr]
            amp    = 0.25 * max(0.05, min(1.0, self._mutation))
            r = max(0.0, min(1.0, center[0] + amp * math.sin(freq * t + phases[0])))
            g = max(0.0, min(1.0, center[1] + amp * math.sin(freq * t + phases[1])))
            b = max(0.0, min(1.0, center[2] + amp * math.sin(freq * t + phases[2])))
            setattr(_params, attr, (r, g, b))
        if self._status_cb:
            self._status_cb(self._t)

    def set_speed(self, v):
        self._speed = max(0.01, v)

    def set_mutation(self, v):
        self._mutation = max(0.01, min(1.0, v))
        for attr, lo, hi, base_freq in self.CHANNELS:
            half = (hi - lo) * 0.5 * self._mutation
            self._amps_f[attr] = self._rand(half * 0.2, half)

    def reseed(self):
        self._randomize_all()

    def start(self):
        if not self._active:
            self._active = True
            self._timer.start()

    def stop(self):
        if self._active:
            self._active = False
            self._timer.stop()

    @property
    def active(self):
        return self._active


_infinite_evo = InfiniteEvolution()


def _rot_x(px, py, pz, a):
    c, s = math.cos(a), math.sin(a)
    return px, c*py - s*pz, s*py + c*pz

def _rot_y(px, py, pz, a):
    c, s = math.cos(a), math.sin(a)
    return c*px + s*pz, py, -s*px + c*pz

def _rot_z(px, py, pz, a):
    c, s = math.cos(a), math.sin(a)
    return c*px - s*py, s*px + c*py, pz

def _apply_space_ops(px, py, pz):
    p = _params
    if p.rep_enabled:
        if p.rep_cell_x > 0.001:
            px = px - p.rep_cell_x * round(px / p.rep_cell_x)
        if p.rep_cell_y > 0.001:
            py = py - p.rep_cell_y * round(py / p.rep_cell_y)
        if p.rep_cell_z > 0.001:
            pz = pz - p.rep_cell_z * round(pz / p.rep_cell_z)
    if p.fold_mirror_x:
        px = abs(px)
    if p.fold_mirror_y:
        py = abs(py)
    if p.fold_mirror_z:
        pz = abs(pz)
    if p.twist_amount > 0.0001:
        ta = p.twist_axis
        if ta == 0:
            ang = py * p.twist_amount
            c, s = math.cos(ang), math.sin(ang)
            px, pz = c*px - s*pz, s*px + c*pz
        elif ta == 1:
            ang = px * p.twist_amount
            c, s = math.cos(ang), math.sin(ang)
            py, pz = c*py - s*pz, s*py + c*pz
        else:
            ang = pz * p.twist_amount
            c, s = math.cos(ang), math.sin(ang)
            px, py = c*px - s*py, s*px + c*py
    if p.warp_enabled:
        f, st = p.warp_freq, p.warp_strength
        wt = p.warp_type
        if wt == 0:
            px += st * math.sin(f * py)
            py += st * math.sin(f * pz)
            pz += st * math.sin(f * px)
        elif wt == 1:
            qx = math.sin(f*px + math.sin(f*pz))
            qy = math.sin(f*py + math.sin(f*px))
            qz = math.sin(f*pz + math.sin(f*py))
            px += st * qx; py += st * qy; pz += st * qz
        else:
            n = (math.sin(f*px)*math.cos(f*py) +
                 math.sin(f*py)*math.cos(f*pz) +
                 math.sin(f*pz)*math.cos(f*px))
            px += st * n; py += st * n; pz += st * n
    return px, py, pz

def _py_sdf(pos):
    p = _params
    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    px, py, pz = _rot_y(px, py, pz, p.rot_y)
    px, py, pz = _rot_z(px, py, pz, p.rot_z)
    px, py, pz = _rot_x(px, py, pz, p.rot_x)
    px, py, pz = _apply_space_ops(px, py, pz)
    ft = p.fractal_type
    if ft == 0:   return _py_sdf_mandelbox(px, py, pz)
    elif ft == 1: return _py_sdf_menger(px, py, pz)
    elif ft == 2: return _py_sdf_sierpinski(px, py, pz)
    elif ft == 3: return _py_sdf_octa(px, py, pz)
    elif ft == 4: return _py_sdf_mandelbulb(px, py, pz)
    else:         return _py_sdf_kleinian(px, py, pz)

def _py_sdf_mandelbox(ox, oy, oz):
    p = _params
    px, py, pz = ox, oy, oz
    dr = 1.0
    fldx = p.mb_fold_x if p.mb_fold_x > 0.001 else p.mb_fold_limit
    fldy = p.mb_fold_y if p.mb_fold_y > 0.001 else p.mb_fold_limit
    fldz = p.mb_fold_z if p.mb_fold_z > 0.001 else p.mb_fold_limit
    si = p.mb_sphere_inner
    so = p.mb_sphere_outer * p.mb_fixed_radius
    for i in range(p.iterations):
        if p.mb_rot_per_iter > 0.0001:
            px, py, pz = _rot_y(px, py, pz, p.mb_rot_per_iter * i)
        if p.mb_fold_mode == 0:
            px = max(-fldx, min(fldx, px)) * 2.0 - px
            py = max(-fldy, min(fldy, py)) * 2.0 - py
            pz = max(-fldz, min(fldz, pz)) * 2.0 - pz
        elif p.mb_fold_mode == 1:
            px = abs(px + fldx) - abs(px - fldx) - px
            py = abs(py + fldy) - abs(py - fldy) - py
            pz = abs(pz + fldz) - abs(pz - fldz) - pz
        else:
            PI = math.pi
            px = math.sin(px * PI / (2.0 * fldx)) * fldx
            py = math.sin(py * PI / (2.0 * fldy)) * fldy
            pz = math.sin(pz * PI / (2.0 * fldz)) * fldz
        r2 = px*px + py*py + pz*pz
        if r2 < si * si:
            k = so / (si * si)
            px *= k; py *= k; pz *= k; dr *= k
        elif r2 < so * so:
            k = so * so / r2
            px *= k; py *= k; pz *= k; dr *= k
        if p.mb_julia_mode == 1:
            px = px * p.scale + p.julia_x
            py = py * p.scale + p.julia_y
            pz = pz * p.scale + p.julia_z
        else:
            px = px * p.scale + ox * (1.0 - p.scale) * 0.1
            py = py * p.scale + oy * (1.0 - p.scale) * 0.1
            pz = pz * p.scale + oz * (1.0 - p.scale) * 0.1
        dr = dr * abs(p.scale) + 1.0
        if px*px + py*py + pz*pz > p.bailout * p.bailout:
            break
    ln = math.sqrt(px*px + py*py + pz*pz)
    return ln / max(abs(dr), 1e-9) * p.de_multiplier

def _py_sdf_menger(ox, oy, oz):
    p = _params
    px, py, pz = ox, oy, oz
    s = 1.0
    ms = p.ms_scale
    mo = p.ms_offset
    sy = p.ms_scale_y if p.ms_scale_y > 0.001 else ms
    sz = p.ms_scale_z if p.ms_scale_z > 0.001 else ms
    for _ in range(p.iterations):
        if p.ms_twist > 0.001:   px, py, pz = _rot_y(px, py, pz, p.ms_twist)
        if p.ms_rot_x > 0.001:   px, py, pz = _rot_x(px, py, pz, p.ms_rot_x)
        if p.ms_rot_z > 0.001:   px, py, pz = _rot_z(px, py, pz, p.ms_rot_z)
        px, py, pz = abs(px), abs(py), abs(pz)
        if px < py: px, py = py, px
        if px < pz: px, pz = pz, px
        if py < pz: py, pz = pz, py
        px = px * ms - mo
        py = py * sy - mo
        pz = pz * sz - mo
        pz += mo * max(0.0, min(1.0, pz / mo * 0.5 + 0.5)) * p.ms_cross_width
        s *= ms
    qx, qy, qz = abs(px) - 1.0, abs(py) - 1.0, abs(pz) - 1.0
    d = (math.sqrt(max(qx,0)**2 + max(qy,0)**2 + max(qz,0)**2)
         + min(max(qx, max(qy, qz)), 0.0))
    return d / s * p.de_multiplier

def _py_sdf_sierpinski(ox, oy, oz):
    p = _params
    vs = p.si_vertex_spread
    jt = p.si_vertex_jitter
    A = ( vs + jt*0.5,  vs - jt*0.5,  vs + jt*0.5)
    B = (-vs - jt*0.5, -vs + jt*0.5,  vs + jt*0.5)
    C = (-vs + jt*0.5,  vs + jt*0.5, -vs - jt*0.5)
    D = ( vs - jt*0.5, -vs - jt*0.5, -vs - jt*0.5)
    verts = (A, B, C, D)
    px, py, pz = ox, oy * p.si_squash, oz
    scale = 1.0
    fb = p.si_fold_bias
    for _ in range(p.iterations):
        if p.si_twist > 0.001: px, py, pz = _rot_y(px, py, pz, p.si_twist)
        if p.si_rot_x > 0.001: px, py, pz = _rot_x(px, py, pz, p.si_rot_x)
        if p.si_rot_z > 0.001: px, py, pz = _rot_z(px, py, pz, p.si_rot_z)
        best = verts[0]; best_d = 1e18
        for v in verts:
            dd = (px-v[0])**2 + (py-v[1])**2 + (pz-v[2])**2
            if dd < best_d:
                best_d = dd; best = v
        px = fb * px - best[0] * (fb - 1.0)
        py = fb * py - best[1] * (fb - 1.0)
        pz = fb * pz - best[2] * (fb - 1.0)
        scale *= fb
    md = max(max(-px-py-pz, px+py-pz), max(-px+py+pz, px-py+pz))
    r = scale * math.sqrt(3.0)
    return (md - r) / max(r, 1e-9) * p.de_multiplier

def _py_sdf_octa(ox, oy, oz):
    p = _params
    px, py, pz = ox, oy, oz
    s = 1.0
    ifs_s = p.oc_ifs_scale
    off = p.oc_offset_uni
    ox2 = p.oc_offset_x if p.oc_offset_x > 0.0001 else p.offset_x * off
    oy2 = p.oc_offset_y if p.oc_offset_y > 0.0001 else p.offset_y * off
    oz2 = p.oc_offset_z if p.oc_offset_z > 0.0001 else p.offset_z * off
    for _ in range(p.iterations):
        if p.oc_twist > 0.001: px, py, pz = _rot_y(px, py, pz, p.oc_twist)
        if p.oc_rot_x > 0.001: px, py, pz = _rot_x(px, py, pz, p.oc_rot_x)
        if p.oc_rot_z > 0.001: px, py, pz = _rot_z(px, py, pz, p.oc_rot_z)
        if p.oc_fold_amount > 0.001:
            fa = p.oc_fold_amount
            px = px + (abs(px) - px) * fa
            py = py + (abs(py) - py) * fa
            pz = pz + (abs(pz) - pz) * fa
        else:
            px, py, pz = abs(px), abs(py), abs(pz)
        if px < py: px, py = py, px
        if px < pz: px, pz = pz, px
        if py < pz: py, pz = pz, py
        px = ifs_s * px - ox2 * (ifs_s - 1.0)
        py = ifs_s * py - oy2 * (ifs_s - 1.0)
        pz = ifs_s * pz - oz2 * (ifs_s - 1.0)
        s *= ifs_s
    sh = max(p.oc_sharpness, 0.5)
    if sh < 1.05:
        r = abs(px) + abs(py) + abs(pz) - 1.0
    else:
        r = (abs(px)**sh + abs(py)**sh + abs(pz)**sh)**(1.0/sh) - 1.0
    return r / s * p.de_multiplier

def _py_sdf_mandelbulb(ox, oy, oz):
    p = _params
    px, py, pz = ox, oy, oz
    dr = 1.0
    r = 0.0
    pw = max(p.mb2_power, 1.0)
    bail = p.mb2_bailout
    for _ in range(p.iterations):
        r = math.sqrt(px*px + py*py + pz*pz)
        if r > bail:
            break
        theta = math.acos(max(-1.0, min(1.0, pz / max(r, 1e-9))))
        phi = math.atan2(py, px)
        dr = r**(pw - 1.0) * pw * dr + 1.0
        zr = r**pw
        np_x = zr * math.sin(theta*pw) * math.cos(phi*pw)
        np_y = zr * math.sin(theta*pw) * math.sin(phi*pw)
        np_z = zr * math.cos(theta*pw)
        fs = p.mb2_fold_strength
        if p.mb2_fold_type == 1 and fs > 0.0:
            np_x = max(-fs, min(fs, np_x)) * 2.0 - np_x
            np_y = max(-fs, min(fs, np_y)) * 2.0 - np_y
            np_z = max(-fs, min(fs, np_z)) * 2.0 - np_z
        elif p.mb2_fold_type == 2 and fs > 0.0:
            np_x = abs(np_x + fs) - abs(np_x - fs) - np_x
            np_y = abs(np_y + fs) - abs(np_y - fs) - np_y
            np_z = abs(np_z + fs) - abs(np_z - fs) - np_z
        if p.mb2_julia_mode == 1:
            px = np_x + p.mb2_julia_x
            py = np_y + p.mb2_julia_y
            pz = np_z + p.mb2_julia_z
        else:
            px = np_x + ox
            py = np_y + oy
            pz = np_z + oz
    return 0.5 * math.log(max(r, 1e-9)) * r / max(dr, 1e-9) * p.de_multiplier

def _py_sdf_kleinian(ox, oy, oz):
    p = _params
    px, py, pz = ox, oy, oz
    dr = 1.0
    ksc = p.kl_scale
    c = (p.kl_cx, p.kl_cy, p.kl_cz)
    fl = p.kl_fold_limit
    sr = p.kl_sph_radius
    for i in range(p.iterations):
        if p.kl_rot_per_iter > 0.0001:
            px, py, pz = _rot_y(px, py, pz, p.kl_rot_per_iter * i)
        px = max(-fl, min(fl, px)) * 2.0 - px
        py = max(-fl, min(fl, py)) * 2.0 - py
        pz = max(-fl, min(fl, pz)) * 2.0 - pz
        r2 = px*px + py*py + pz*pz
        k = max(sr * sr / max(r2, 1e-9), 1.0)
        px *= k; py *= k; pz *= k; dr *= k
        px = px * ksc + c[0]
        py = py * ksc + c[1]
        pz = pz * ksc + c[2]
        dr = dr * abs(ksc) + 1.0
        if r2 > p.bailout * p.bailout:
            break
    ln = math.sqrt(px*px + py*py + pz*pz)
    d = (ln - abs(ksc - 1.0)) / max(abs(dr), 1e-9)
    mix = p.kl_mix_factor
    return (d * (1.0 - mix) + d * 0.5 * mix) * p.de_multiplier

def _py_sdf_normal(pos, eps=None):
    if eps is None:
        eps = max(_params.normal_eps * 2.0, 0.003)
    d1x = _py_sdf((pos[0]+eps, pos[1], pos[2]))
    d2x = _py_sdf((pos[0]-eps, pos[1], pos[2]))
    d1y = _py_sdf((pos[0], pos[1]+eps, pos[2]))
    d2y = _py_sdf((pos[0], pos[1]-eps, pos[2]))
    d1z = _py_sdf((pos[0], pos[1], pos[2]+eps))
    d2z = _py_sdf((pos[0], pos[1], pos[2]-eps))
    nx, ny, nz = d1x - d2x, d1y - d2y, d1z - d2z
    ln = math.sqrt(nx*nx + ny*ny + nz*nz)
    if ln < 1e-9:
        return (0.0, 1.0, 0.0)
    return (nx/ln, ny/ln, nz/ln)

def _py_sdf_gradient(pos, eps=0.005):
    return _py_sdf_normal(pos, eps)

class PlayerState:
    GRAVITY_STRENGTH = 1.0
    MOVE_SPEED       = 0.5
    JUMP_SPEED       = 0.6
    FRICTION         = 10.0
    AIR_CONTROL      = 0.13
    SPEED_CAP        = 4.2
    COLLISION_BIAS   = 1.0
    GRAVITY_MODE     = 0
    PLAYER_HEIGHT    = 0.08
    GROUND_DIST      = 0.12
    _PROBE_STEPS     = 6
    _PUSH_ITERS      = 4
    _GROUND_PROBES   = 5
    CAM_SPRING_K     = 100.0
    CAM_DAMPING      = 10.0
    BOB_FREQ         = 0.0
    BOB_AMP_V        = 0.006
    BOB_AMP_H        = 0.003
    BOB_SPEED_THRESH = 0.05
    NORMAL_SMOOTH    = 6.0

    def __init__(self):
        self.vel              = [0.0, 0.0, 0.0]
        self.on_ground        = False
        self.jump_queued      = False
        self._gravity_dir     = (0.0, -1.0, 0.0)
        self._surface_normal  = (0.0, 1.0, 0.0)
        self._sdf_scale_cache = 1.0
        self._smooth_pos      = None
        self._smooth_vel      = [0.0, 0.0, 0.0]
        self._smooth_normal   = [0.0, 1.0, 0.0]
        self._bob_phase       = 0.0

    def _calibrate_sdf_scale(self, pos):
        eps = 0.2
        samples = [
            _py_sdf((pos[0]+eps, pos[1], pos[2])),
            _py_sdf((pos[0]-eps, pos[1], pos[2])),
            _py_sdf((pos[0], pos[1]+eps, pos[2])),
            _py_sdf((pos[0], pos[1]-eps, pos[2])),
            _py_sdf((pos[0], pos[1], pos[2]+eps)),
            _py_sdf((pos[0], pos[1], pos[2]-eps)),
        ]
        d0 = _py_sdf(pos)
        gradients = [abs(s - d0) / eps for s in samples]
        mean_grad = sum(gradients) / len(gradients)
        return max(mean_grad, 0.01)

    def _effective_radius(self):
        return max(self.PLAYER_HEIGHT, 0.005) * self.COLLISION_BIAS

    def _ground_threshold(self):
        return self._effective_radius() + max(self.GROUND_DIST, 0.005)

    def _compute_gravity_dir(self, pos, surface_normal):
        mode = self.GRAVITY_MODE
        if mode == 0:
            return (-surface_normal[0], -surface_normal[1], -surface_normal[2])
        elif mode == 1:
            cx, cy, cz = -pos[0], -pos[1], -pos[2]
            clen = math.sqrt(cx*cx + cy*cy + cz*cz)
            if clen < 1e-6:
                return (0.0, -1.0, 0.0)
            return (cx/clen, cy/clen, cz/clen)
        else:
            return (0.0, -1.0, 0.0)

    def _push_out(self, pos, radius):
        for _ in range(self._PUSH_ITERS):
            d = _py_sdf(pos)
            if d >= radius:
                break
            nx, ny, nz = _py_sdf_normal(pos)
            push = (radius - d) + 1e-4
            pos[0] += nx * push
            pos[1] += ny * push
            pos[2] += nz * push
        return pos

    def _is_on_ground(self, pos, gd, radius, gnd_thresh):
        d = _py_sdf(pos)
        if d < gnd_thresh:
            return True
        probe_dist = gnd_thresh * 0.8
        for i in range(self._GROUND_PROBES):
            t = (i + 1) / self._GROUND_PROBES
            px = pos[0] + gd[0] * probe_dist * t
            py = pos[1] + gd[1] * probe_dist * t
            pz = pos[2] + gd[2] * probe_dist * t
            if _py_sdf((px, py, pz)) < radius:
                return True
        return False

    def update(self, dt, spd_mul=1.0):
        if dt <= 0 or dt > 0.1:
            return
        pos = list(_params.cam_pos)
        radius    = self._effective_radius()
        gnd_thresh = self._ground_threshold()

        nx, ny, nz = _py_sdf_normal(pos)
        self._surface_normal = (nx, ny, nz)
        self._gravity_dir = self._compute_gravity_dir(pos, (nx, ny, nz))
        gd = self._gravity_dir

        self.on_ground = self._is_on_ground(pos, gd, radius, gnd_thresh)

        fwd, right, _ = _calc_basis_from_params()

        def _proj_on_plane(vx, vy, vz, nx_, ny_, nz_):
            dot = vx*nx_ + vy*ny_ + vz*nz_
            return vx - dot*nx_, vy - dot*ny_, vz - dot*nz_

        fpx, fpy, fpz = _proj_on_plane(fwd[0], fwd[1], fwd[2], gd[0], gd[1], gd[2])
        flen = math.sqrt(fpx*fpx + fpy*fpy + fpz*fpz)
        if flen > 1e-6:
            fpx /= flen; fpy /= flen; fpz /= flen
        else:
            fpx, fpy, fpz = fwd[0], fwd[1], fwd[2]

        rpx, rpy, rpz = _proj_on_plane(right[0], right[1], right[2], gd[0], gd[1], gd[2])
        rlen = math.sqrt(rpx*rpx + rpy*rpy + rpz*rpz)
        if rlen > 1e-6:
            rpx /= rlen; rpy /= rlen; rpz /= rlen
        else:
            rpx, rpy, rpz = right[0], right[1], right[2]

        ci = _cam_input
        mvx, mvy, mvz = 0.0, 0.0, 0.0
        if 'w' in ci.keys_pressed: mvx += fpx; mvy += fpy; mvz += fpz
        if 's' in ci.keys_pressed: mvx -= fpx; mvy -= fpy; mvz -= fpz
        if 'a' in ci.keys_pressed: mvx -= rpx; mvy -= rpy; mvz -= rpz
        if 'd' in ci.keys_pressed: mvx += rpx; mvy += rpy; mvz += rpz
        mv_len = math.sqrt(mvx*mvx + mvy*mvy + mvz*mvz)
        if mv_len > 1e-6:
            mvx /= mv_len; mvy /= mv_len; mvz /= mv_len

        eff_move_spd = self.MOVE_SPEED * spd_mul
        eff_speed_cap = self.SPEED_CAP * spd_mul

        if self.on_ground:
            vdot = self.vel[0]*gd[0] + self.vel[1]*gd[1] + self.vel[2]*gd[2]
            if vdot > 0.0:
                self.vel[0] -= gd[0] * vdot
                self.vel[1] -= gd[1] * vdot
                self.vel[2] -= gd[2] * vdot
            alpha = min(self.FRICTION * dt, 1.0)
            if mv_len > 1e-6:
                tvx = mvx * eff_move_spd
                tvy = mvy * eff_move_spd
                tvz = mvz * eff_move_spd
            else:
                tvx = tvy = tvz = 0.0
            self.vel[0] += (tvx - self.vel[0]) * alpha
            self.vel[1] += (tvy - self.vel[1]) * alpha
            self.vel[2] += (tvz - self.vel[2]) * alpha
            if self.jump_queued:
                sn = self._surface_normal
                self.vel[0] += sn[0] * self.JUMP_SPEED
                self.vel[1] += sn[1] * self.JUMP_SPEED
                self.vel[2] += sn[2] * self.JUMP_SPEED
        else:
            self.vel[0] += gd[0] * self.GRAVITY_STRENGTH * dt
            self.vel[1] += gd[1] * self.GRAVITY_STRENGTH * dt
            self.vel[2] += gd[2] * self.GRAVITY_STRENGTH * dt
            if mv_len > 1e-6:
                ac = eff_move_spd * self.AIR_CONTROL * dt
                self.vel[0] += mvx * ac
                self.vel[1] += mvy * ac
                self.vel[2] += mvz * ac

        self.jump_queued = False

        spd = math.sqrt(self.vel[0]**2 + self.vel[1]**2 + self.vel[2]**2)
        if spd > eff_speed_cap:
            inv = eff_speed_cap / spd
            self.vel[0] *= inv; self.vel[1] *= inv; self.vel[2] *= inv

        n_substeps = self._PROBE_STEPS
        sub_dt = dt / n_substeps
        for _ in range(n_substeps):
            dx = self.vel[0] * sub_dt
            dy = self.vel[1] * sub_dt
            dz = self.vel[2] * sub_dt
            step_len = math.sqrt(dx*dx + dy*dy + dz*dz)
            if step_len < 1e-9:
                break
            sdx, sdy, sdz = dx/step_len, dy/step_len, dz/step_len
            t = 0.0
            max_t = step_len
            for _ in range(64):
                d = _py_sdf((pos[0]+sdx*t, pos[1]+sdy*t, pos[2]+sdz*t))
                safe_step = d * 0.85
                if d < radius:
                    break
                t += max(safe_step, 1e-4)
                if t >= max_t:
                    t = max_t
                    break
            pos[0] += sdx * t
            pos[1] += sdy * t
            pos[2] += sdz * t
            d_final = _py_sdf(pos)
            if d_final < radius:
                cn, cy_n, cz_n = _py_sdf_normal(pos)
                push = (radius - d_final) + 1e-4
                pos[0] += cn * push
                pos[1] += cy_n * push
                pos[2] += cz_n * push
                vdot = self.vel[0]*cn + self.vel[1]*cy_n + self.vel[2]*cz_n
                if vdot < 0.0:
                    self.vel[0] -= cn * vdot
                    self.vel[1] -= cy_n * vdot
                    self.vel[2] -= cz_n * vdot

        _params.cam_pos = pos
        sn = self._surface_normal
        alpha_n = 1.0 - math.exp(-self.NORMAL_SMOOTH * dt)
        for i in range(3):
            self._smooth_normal[i] += (sn[i] - self._smooth_normal[i]) * alpha_n
        nl = math.sqrt(sum(x*x for x in self._smooth_normal)) or 1.0
        self._smooth_normal = [x / nl for x in self._smooth_normal]

    def get_render_pos(self, dt):
        phys_pos = _params.cam_pos
        if self._smooth_pos is None:
            self._smooth_pos = list(phys_pos)
        k, d = self.CAM_SPRING_K, self.CAM_DAMPING
        for i in range(3):
            diff = phys_pos[i] - self._smooth_pos[i]
            accel = k * diff - d * self._smooth_vel[i]
            self._smooth_vel[i] += accel * dt
            self._smooth_pos[i] += self._smooth_vel[i] * dt
        spd = math.sqrt(sum(v*v for v in self.vel))
        is_moving = self.on_ground and spd > self.BOB_SPEED_THRESH
        if is_moving:
            self._bob_phase += self.BOB_FREQ * 2.0 * math.pi * dt
        else:
            self._bob_phase *= max(0.0, 1.0 - dt * 8.0)
        spd_t = min(spd / max(self.MOVE_SPEED, 1e-6), 1.0) if is_moving else 0.0
        bob_v = math.sin(self._bob_phase) * self.BOB_AMP_V * spd_t
        bob_h = math.sin(self._bob_phase * 0.5) * self.BOB_AMP_H * spd_t
        sn = self._smooth_normal
        _, right, _ = _calc_basis_from_params()
        return [
            self._smooth_pos[0] + sn[0] * bob_v + right[0] * bob_h,
            self._smooth_pos[1] + sn[1] * bob_v + right[1] * bob_h,
            self._smooth_pos[2] + sn[2] * bob_v + right[2] * bob_h,
        ]

def _calc_basis_from_params():
    yaw, pitch = _params.cam_yaw, _params.cam_pitch
    fx =  math.cos(pitch) * math.sin(yaw)
    fy =  math.sin(pitch)
    fz = -math.cos(pitch) * math.cos(yaw)
    fwd = (fx, fy, fz)
    rx, ry, rz = -fz, 0.0, fx
    rlen = math.sqrt(rx*rx + rz*rz) or 1.0
    right = (rx/rlen, 0.0, rz/rlen)
    ux = right[1]*fwd[2] - right[2]*fwd[1]
    uy = right[2]*fwd[0] - right[0]*fwd[2]
    uz = right[0]*fwd[1] - right[1]*fwd[0]
    ulen = math.sqrt(ux*ux + uy*uy + uz*uz) or 1.0
    up = (ux/ulen, uy/ulen, uz/ulen)
    return fwd, right, up

_player_state = PlayerState()
_player_move_input = [0.0, 0.0]

class _CamInput:
    keys_pressed: set    = set()
    mouse_dragging: bool = False

_cam_input = _CamInput()
_cam_vel   = [0.0, 0.0, 0.0]

class FractalWindow(mglw.WindowConfig):
    title        = "Kaleidoscopic IFS Fractal " + APP_VERSION
    gl_version   = (3, 3)
    window_size  = (1280, 720)
    aspect_ratio = None
    resizable    = True

    # samples      = 8
    MOUSE_SENS_YAW   = 0.003
    MOUSE_SENS_PITCH = 0.003
    MOUSE_SCROLL_SPD = 0.5
    KEY_MOVE_SPD     = 2.0
    SHIFT_MUL        = 5.0
    ALT_MUL          = 0.2
    PITCH_LIMIT      = math.pi * 0.49
    SMOOTHING        = 12.0
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.prog = self.ctx.program(vertex_shader=VERT_SHADER,
                                     fragment_shader=FRAG_SHADER)
        verts = np.array([-1,-1, 1,-1, -1,1, 1,1], dtype='f4')
        vbo = self.ctx.buffer(verts)
        self.vao = self.ctx.simple_vertex_array(self.prog, vbo, 'in_position')

        self.post_prog = self.ctx.program(vertex_shader=POST_VERT,
                                          fragment_shader=POST_FRAG)
        self.post_vao  = self.ctx.simple_vertex_array(self.post_prog, vbo, 'in_position')

        w, h = self.wnd.size
        self._fbo_size = (w, h)
        self._scene_tex = self.ctx.texture((w, h), 3, dtype='f4')
        self._scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._scene_fbo = self.ctx.framebuffer(color_attachments=[self._scene_tex])

        self.start = time.time()
        self._pending_screenshot = False
        self._debug_vbo = vbo
        self._init_debug_overlay()

    def _init_debug_overlay(self):
        self._dbg_prog = self.ctx.program(
            vertex_shader=DEBUG_OVERLAY_VERT,
            fragment_shader=DEBUG_OVERLAY_FRAG,
        )
        self._dbg_vao = self.ctx.simple_vertex_array(
            self._dbg_prog, self._debug_vbo, 'in_position'
        )
        self._dbg_probe_cache = []

    def _dset(self, name, val):
        if name in self._dbg_prog:
            self._dbg_prog[name].value = val

    def _build_debug_probes(self, pos, ps):
        probes = []
        radius   = ps._effective_radius()
        gnd_thr  = ps._ground_threshold()
        gd       = ps._gravity_dir
        sn       = ps._surface_normal
        fwd, right, up = _calc_basis_from_params()
        dirs = [
            (sn[0], sn[1], sn[2]),
            (-sn[0], -sn[1], -sn[2]),
            (gd[0], gd[1], gd[2]),
            (fwd[0], fwd[1], fwd[2]),
            (-fwd[0], -fwd[1], -fwd[2]),
            (right[0], right[1], right[2]),
            (-right[0], -right[1], -right[2]),
            (up[0], up[1], up[2]),
            (-up[0], -up[1], -up[2]),
        ]
        diag_scale = radius * 0.85
        for i in range(len(dirs)):
            for j in range(i+1, len(dirs)):
                dx = (dirs[i][0]+dirs[j][0]) * 0.5
                dy = (dirs[i][1]+dirs[j][1]) * 0.5
                dz = (dirs[i][2]+dirs[j][2]) * 0.5
                ln = math.sqrt(dx*dx+dy*dy+dz*dz)
                if ln > 1e-6:
                    dirs.append((dx/ln, dy/ln, dz/ln))
        for d in dirs[:32]:
            px = pos[0] + d[0] * radius
            py = pos[1] + d[1] * radius
            pz = pos[2] + d[2] * radius
            sdf_v = _py_sdf((px, py, pz))
            probes.append(((px, py, pz), sdf_v))
        return probes[:32], radius

    def _render_debug_overlay(self):
        global _debug_overlay_enabled
        if not _debug_overlay_enabled or not _params.player_mode:
            return
        ps  = _player_state
        pos = list(_params.cam_pos)
        fwd, right, up = self._calc_basis()
        w, h = self.wnd.size
        cw, ch = w * 0.5, h * 0.5
        fov   = _params.fov
        import math as _m

        def _world_to_2d_offset(wx, wy, wz):
            dx, dy, dz = wx - pos[0], wy - pos[1], wz - pos[2]
            z = dx*fwd[0] + dy*fwd[1] + dz*fwd[2]
            if z < 0.001:
                return None
            half_tan = _m.tan(fov * 0.5)
            sx = (dx*right[0] + dy*right[1] + dz*right[2]) / (z * half_tan)
            sy = (dx*up[0]    + dy*up[1]    + dz*up[2])    / (z * half_tan)
            return (sx * cw, sy * ch)

        sn   = ps._surface_normal
        gd   = ps._gravity_dir
        rad  = ps._effective_radius()
        gnd  = ps._ground_threshold()

        sn_off = _world_to_2d_offset(
            pos[0] + sn[0] * rad * 4.0,
            pos[1] + sn[1] * rad * 4.0,
            pos[2] + sn[2] * rad * 4.0,
        )
        gd_off = _world_to_2d_offset(
            pos[0] + gd[0] * gnd * 3.0,
            pos[1] + gd[1] * gnd * 3.0,
            pos[2] + gd[2] * gnd * 3.0,
        )

        norm_ss = (float(sn_off[0]), float(sn_off[1])) if sn_off else (0.0, -80.0)
        grav_ss = (float(gd_off[0]), float(gd_off[1])) if gd_off else (0.0,  80.0)

        sdf_v = float(_py_sdf(pos))
        ref_z = max(rad, sdf_v) if sdf_v > 0 else rad
        half_tan = _m.tan(fov * 0.5)
        col_ring_px = float(rad  / (ref_z * half_tan) * ch) if ref_z > 1e-6 else 0.0
        gnd_ring_px = float(gnd  / (ref_z * half_tan) * ch) if ref_z > 1e-6 else 0.0
        col_ring_px = max(0.0, min(col_ring_px, float(min(w, h)) * 0.9))
        gnd_ring_px = max(0.0, min(gnd_ring_px, float(min(w, h)) * 0.9))

        probes, probe_r = self._build_debug_probes(pos, ps)
        n_probes = min(len(probes), 32)

        probe_ss_list  = []
        probe_sdf_list = []
        for (px2, py2, pz2), sv in probes[:n_probes]:
            off = _world_to_2d_offset(px2, py2, pz2)
            if off is not None:
                probe_ss_list.append((cw + off[0], ch + off[1]))
            else:
                probe_ss_list.append((-9999.0, -9999.0))
            probe_sdf_list.append(float(sv))

        self._dset('u_res',          (float(w), float(h)))
        self._dset('u_on_ground',    1 if ps.on_ground else 0)
        self._dset('u_enabled',      1)
        self._dset('u_sdf_val',      sdf_v)
        self._dset('u_radius',       float(rad))
        self._dset('u_gnd_thresh',   float(gnd))
        self._dset('u_speed',        float(_m.sqrt(sum(v*v for v in ps.vel))))
        self._dset('u_norm_dir_ss',  norm_ss)
        self._dset('u_grav_dir_ss',  grav_ss)
        self._dset('u_col_ring_px',  col_ring_px)
        self._dset('u_gnd_ring_px',  gnd_ring_px)
        self._dset('u_probe_count',  n_probes)
        self._dset('u_probe_radius', float(probe_r))

        for i in range(32):
            kp = f'u_probe_ss[{i}]'
            ks = f'u_probe_sdf[{i}]'
            if i < n_probes:
                if kp in self._dbg_prog: self._dbg_prog[kp].value = (float(probe_ss_list[i][0]), float(probe_ss_list[i][1]))
                if ks in self._dbg_prog: self._dbg_prog[ks].value = probe_sdf_list[i]
            else:
                if kp in self._dbg_prog: self._dbg_prog[kp].value = (-9999.0, -9999.0)
                if ks in self._dbg_prog: self._dbg_prog[ks].value = 0.0

        self.ctx.enable(moderngl.BLEND)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
        self._dbg_vao.render(moderngl.TRIANGLE_STRIP)
        self.ctx.disable(moderngl.BLEND)

    def _ensure_fbo(self):
        w, h = self.wnd.size
        if (w, h) != self._fbo_size:
            self._scene_tex.release()
            self._scene_fbo.release()
            self._scene_tex = self.ctx.texture((w, h), 3, dtype='f4')
            self._scene_tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self._scene_fbo = self.ctx.framebuffer(color_attachments=[self._scene_tex])
            self._fbo_size  = (w, h)

    def _set(self, name, val):
        if name in self.prog:
            self.prog[name].value = val
    def _speed_mul(self):
        try:
            import pyglet
            pw = getattr(self.wnd, '_window', None)
            if pw is not None:
                ks = pw._keyboard
                km = pyglet.window.key
                if ks[km.LSHIFT] or ks[km.RSHIFT]:
                    return self.SHIFT_MUL
                if ks[km.LALT] or ks[km.RALT]:
                    return self.ALT_MUL
                return 1.0
        except Exception:
            pass
        if 'shift' in _cam_input.keys_pressed:
            return self.SHIFT_MUL
        if 'alt' in _cam_input.keys_pressed:
            return self.ALT_MUL
        return 1.0
    def _set_mouse_exclusive(self, exclusive: bool):
        try:
            pw = getattr(self.wnd, '_window', None)
            if pw is not None:
                pw.set_exclusive_mouse(exclusive)
        except Exception:
            pass
    def key_event(self, key, action, modifiers):
        from moderngl_window.context.pyglet.keys import Keys
        import pyglet
        km = pyglet.window.key
        PRESS   = self.wnd.keys.ACTION_PRESS
        RELEASE = self.wnd.keys.ACTION_RELEASE
        mod_keys = {
            (km.LSHIFT, km.RSHIFT): 'shift',
            (km.LALT,   km.RALT):   'alt',
            (km.LCTRL,  km.RCTRL):  'ctrl',
        }
        for key_pair, name in mod_keys.items():
            if key in key_pair:
                if action == PRESS:   _cam_input.keys_pressed.add(name)
                elif action == RELEASE: _cam_input.keys_pressed.discard(name)
        if key == km.V and action == PRESS:
            _params.player_mode = not _params.player_mode
            if _params.player_mode:
                _player_state.vel = [0.0, 0.0, 0.0]
                _player_state.on_ground = False
                _player_state.jump_queued = False
                _player_state._smooth_pos = list(_params.cam_pos)
                _player_state._smooth_vel = [0.0, 0.0, 0.0]
                _player_state._smooth_normal = list(_player_state._surface_normal)
                _player_state._bob_phase  = 0.0
            self._set_mouse_exclusive(_params.player_mode)
            return
        if key == km.F1 and action == PRESS:
            global _debug_overlay_enabled
            _debug_overlay_enabled = not _debug_overlay_enabled
            return
        if key == km.SPACE and action == PRESS and _params.player_mode:
            _player_state.jump_queued = True
            return
        name_map = {
            Keys.W: 'w', Keys.S: 's',
            Keys.A: 'a', Keys.D: 'd',
            Keys.Q: 'q', Keys.E: 'e',
        }
        try:
            name_map[km.SPACE]    = 'space'
            name_map[km.X]        = 'x'
            name_map[km.RBRACKET] = 'rbracket'
        except Exception:
            pass
        k = name_map.get(key)
        if k is None:
            try:
                import pyglet
                km2 = pyglet.window.key
                if key == km2.F12 and action == PRESS:
                    self._pending_screenshot = True
                if key == km2.SPACE:
                    k = 'space'
                if key == km2.X:
                    k = 'x'
                if key == km2.RBRACKET:
                    k = 'rbracket'
            except Exception:
                pass
        if k is not None:
            if action == PRESS:   _cam_input.keys_pressed.add(k)
            elif action == RELEASE: _cam_input.keys_pressed.discard(k)
    def mouse_press_event(self, x, y, button):
        if button == 1:
            _cam_input.mouse_dragging = True
    def mouse_release_event(self, x, y, button):
        if button == 1:
            _cam_input.mouse_dragging = False
    def mouse_drag_event(self, x, y, dx, dy):
        if _params.player_mode:
            return
        if not _cam_input.mouse_dragging:
            return
        _params.cam_yaw   += dx * self.MOUSE_SENS_YAW
        _params.cam_pitch  = max(-self.PITCH_LIMIT, min(self.PITCH_LIMIT,
            _params.cam_pitch - dy * self.MOUSE_SENS_PITCH))
    def mouse_position_event(self, x, y, dx, dy):
        if not _params.player_mode:
            return
        _params.cam_yaw   += dx * self.MOUSE_SENS_YAW
        _params.cam_pitch  = max(-self.PITCH_LIMIT, min(self.PITCH_LIMIT,
            _params.cam_pitch - dy * self.MOUSE_SENS_PITCH))
    def mouse_scroll_event(self, x_offset, y_offset):
        mul = self._speed_mul()
        fwd, _, _ = self._calc_basis()
        p = _params.cam_pos
        spd = y_offset * self.MOUSE_SCROLL_SPD * mul
        _params.cam_pos = [p[0]+fwd[0]*spd, p[1]+fwd[1]*spd, p[2]+fwd[2]*spd]
    def _calc_basis(self):
        yaw, pitch, roll = _params.cam_yaw, _params.cam_pitch, _params.cam_roll
        fx =  math.cos(pitch) * math.sin(yaw)
        fy =  math.sin(pitch)
        fz = -math.cos(pitch) * math.cos(yaw)
        fwd = (fx, fy, fz)
        rx, ry, rz = -fz, 0.0, fx
        rlen = math.sqrt(rx*rx + rz*rz) or 1.0
        right = (rx/rlen, 0.0, rz/rlen)
        ux = right[1]*fwd[2] - right[2]*fwd[1]
        uy = right[2]*fwd[0] - right[0]*fwd[2]
        uz = right[0]*fwd[1] - right[1]*fwd[0]
        ulen = math.sqrt(ux*ux + uy*uy + uz*uz) or 1.0
        up = (ux/ulen, uy/ulen, uz/ulen)
        if roll != 0.0:
            cr, sr = math.cos(roll), math.sin(roll)
            rx2 = right[0]*cr + up[0]*sr
            ry2 = right[1]*cr + up[1]*sr
            rz2 = right[2]*cr + up[2]*sr
            ux2 = -right[0]*sr + up[0]*cr
            uy2 = -right[1]*sr + up[1]*cr
            uz2 = -right[2]*sr + up[2]*cr
            right = (rx2, ry2, rz2)
            up    = (ux2, uy2, uz2)
        return fwd, right, up
    def _update_camera_keys(self, dt):
        global _cam_vel
        ci = _cam_input
        if _params.player_mode:
            spd_mul = self.SHIFT_MUL if 'shift' in ci.keys_pressed else (self.ALT_MUL if 'alt' in ci.keys_pressed else 1.0)
            _player_state.update(dt, spd_mul)
            return
        mul = self._speed_mul()
        fwd, right, up = self._calc_basis()
        target = [0.0, 0.0, 0.0]
        spd = self.KEY_MOVE_SPD * mul
        if 'w' in ci.keys_pressed:
            target = [target[i] + fwd[i]   * spd for i in range(3)]
        if 's' in ci.keys_pressed:
            target = [target[i] - fwd[i]   * spd for i in range(3)]
        if 'a' in ci.keys_pressed:
            target = [target[i] - right[i] * spd for i in range(3)]
        if 'd' in ci.keys_pressed:
            target = [target[i] + right[i] * spd for i in range(3)]
        if 'space' in ci.keys_pressed:
            target = [target[i] + up[i]    * spd for i in range(3)]
        if 'ctrl' in ci.keys_pressed:
            target = [target[i] - up[i]    * spd for i in range(3)]
        ROLL_SPD = 1.2
        if 'x' in ci.keys_pressed:
            _params.cam_roll -= ROLL_SPD * dt * mul
        if 'rbracket' in ci.keys_pressed:
            _params.cam_roll += ROLL_SPD * dt * mul
        alpha = 1.0 - math.exp(-self.SMOOTHING * dt)
        _cam_vel = [_cam_vel[i] + (target[i] - _cam_vel[i]) * alpha for i in range(3)]
        p = _params.cam_pos
        _params.cam_pos = [p[i] + _cam_vel[i] * dt for i in range(3)]
    def render(self, t, ft):
        self._update_camera_keys(ft)
        self._ensure_fbo()
        p = _params
        fwd, right, up = self._calc_basis()
        elapsed = time.time() - self.start

        render_pos = (
            _player_state.get_render_pos(ft)
            if p.player_mode
            else list(p.cam_pos)
        )

        self._scene_fbo.use()
        self._scene_fbo.clear(0, 0, 0)
        self._set('u_time',         elapsed)
        self._set('u_resolution',   self.wnd.size)
        self._set('u_iterations',   p.iterations)
        self._set('u_scale',        p.scale)
        self._set('u_fold_x',       1.0 if p.fold_x else 0.0)
        self._set('u_fold_y',       1.0 if p.fold_y else 0.0)
        self._set('u_fold_z',       1.0 if p.fold_z else 0.0)
        self._set('u_rot_x',        p.rot_x)
        self._set('u_rot_y',        p.rot_y)
        self._set('u_rot_z',        p.rot_z)
        self._set('u_offset_x',     p.offset_x)
        self._set('u_offset_y',     p.offset_y)
        self._set('u_offset_z',     p.offset_z)
        self._set('u_julia_x',      p.julia_x)
        self._set('u_julia_y',      p.julia_y)
        self._set('u_julia_z',      p.julia_z)
        self._set('u_fractal_type', p.fractal_type)
        self._set('u_bailout',      p.bailout)
        self._set('u_min_dist',     p.min_dist)
        self._set('u_fog_density',  p.fog_density)
        self._set('u_color1',       p.color1)
        self._set('u_color2',       p.color2)
        self._set('u_color3',       p.color3)
        self._set('u_color_mode',   p.color_mode)
        self._set('u_ao_strength',  p.ao_strength)
        self._set('u_shadow_soft',  p.shadow_soft)
        self._set('u_shadows',      1 if p.shadows else 0)
        self._set('u_glow',         p.glow)
        self._set('u_cam_pos',      tuple(render_pos))
        self._set('u_cam_fwd',      fwd)
        self._set('u_cam_right',    right)
        self._set('u_cam_up',       up)
        self._set('u_animate',      1 if p.animate else 0)
        self._set('u_anim_speed',   p.anim_speed)
        self._set('u_fov',          p.fov)
        self._set('u_de_multiplier', p.de_multiplier)
        self._set('u_orbit_trap_type', p.orbit_trap_type)
        # Fog
        self._set('u_fog_color',    p.fog_color)
        # AO
        self._set('u_ao_radius',    p.ao_radius)
        self._set('u_ao_samples',   p.ao_samples)
        # Glow
        self._set('u_glow_intensity', p.glow_intensity)
        self._set('u_glow_falloff',   p.glow_falloff)
        self._set('u_glow_radius',    p.glow_radius)
        self._set('u_rim_strength',   p.rim_strength)
        self._set('u_emission',       p.emission)
        # Background
        self._set('u_bg_color1',  p.bg_color1)
        self._set('u_bg_color2',  p.bg_color2)
        self._set('u_bg_mode',    p.bg_mode)
        # AA
        self._set('u_aa_samples', p.aa_samples)
        # Mandelbox fine-tune
        self._set('u_mb_fold_limit',   p.mb_fold_limit)
        self._set('u_mb_sphere_inner', p.mb_sphere_inner)
        self._set('u_mb_sphere_outer', p.mb_sphere_outer)
        self._set('u_mb_fixed_radius', p.mb_fixed_radius)
        self._set('u_mb_color_scale',  p.mb_color_scale)
        self._set('u_mb_rot_per_iter', p.mb_rot_per_iter)
        self._set('u_mb_fold_mode',    p.mb_fold_mode)
        # Menger fine-tune
        self._set('u_ms_cross_width',  p.ms_cross_width)
        self._set('u_ms_scale',        p.ms_scale)
        self._set('u_ms_offset',       p.ms_offset)
        self._set('u_ms_twist',        p.ms_twist)
        self._set('u_ms_sharpness',    p.ms_sharpness)
        # Sierpinski fine-tune
        self._set('u_si_vertex_spread', p.si_vertex_spread)
        self._set('u_si_fold_bias',     p.si_fold_bias)
        self._set('u_si_twist',         p.si_twist)
        self._set('u_si_squash',        p.si_squash)
        self._set('u_si_vertex_jitter', p.si_vertex_jitter)
        # Octahedron fine-tune
        self._set('u_oc_ifs_scale',    p.oc_ifs_scale)
        self._set('u_oc_twist',        p.oc_twist)
        self._set('u_oc_sharpness',    p.oc_sharpness)
        self._set('u_oc_offset_uni',   p.oc_offset_uni)
        self._set('u_oc_fold_amount',  p.oc_fold_amount)
        self._set('u_oc_offset_x',     p.oc_offset_x)
        self._set('u_oc_offset_y',     p.oc_offset_y)
        self._set('u_oc_offset_z',     p.oc_offset_z)
        self._set('u_oc_rot_x',        p.oc_rot_x)
        self._set('u_oc_rot_z',        p.oc_rot_z)
        # Mandelbulb fine-tune
        self._set('u_mb2_power',        p.mb2_power)
        self._set('u_mb2_bailout',      p.mb2_bailout)
        self._set('u_mb2_julia_x',      p.mb2_julia_x)
        self._set('u_mb2_julia_y',      p.mb2_julia_y)
        self._set('u_mb2_julia_z',      p.mb2_julia_z)
        self._set('u_mb2_julia_mode',   p.mb2_julia_mode)
        self._set('u_mb2_fold_strength', p.mb2_fold_strength)
        self._set('u_mb2_fold_type',    p.mb2_fold_type)
        # Pseudo-Kleinian fine-tune
        self._set('u_kl_scale',         p.kl_scale)
        self._set('u_kl_cx',            p.kl_cx)
        self._set('u_kl_cy',            p.kl_cy)
        self._set('u_kl_cz',            p.kl_cz)
        self._set('u_kl_fold_limit',    p.kl_fold_limit)
        self._set('u_kl_sph_radius',    p.kl_sph_radius)
        self._set('u_kl_rot_per_iter',  p.kl_rot_per_iter)
        self._set('u_kl_mix_factor',    p.kl_mix_factor)
        # Mandelbox per-axis fold
        self._set('u_mb_fold_x',        p.mb_fold_x)
        self._set('u_mb_fold_y',        p.mb_fold_y)
        self._set('u_mb_fold_z',        p.mb_fold_z)
        self._set('u_mb_julia_mode',    p.mb_julia_mode)
        # Menger per-axis
        self._set('u_ms_rot_x',         p.ms_rot_x)
        self._set('u_ms_rot_z',         p.ms_rot_z)
        self._set('u_ms_scale_y',       p.ms_scale_y)
        self._set('u_ms_scale_z',       p.ms_scale_z)
        # Sierpinski per-axis
        self._set('u_si_rot_x',         p.si_rot_x)
        self._set('u_si_rot_z',         p.si_rot_z)
        # Global space operators
        self._set('u_warp_enabled',     1 if p.warp_enabled  else 0)
        self._set('u_warp_strength',    p.warp_strength)
        self._set('u_warp_freq',        p.warp_freq)
        self._set('u_warp_type',        p.warp_type)
        self._set('u_twist_axis',       p.twist_axis)
        self._set('u_twist_amount',     p.twist_amount)
        self._set('u_fold_mirror_x',    1 if p.fold_mirror_x else 0)
        self._set('u_fold_mirror_y',    1 if p.fold_mirror_y else 0)
        self._set('u_fold_mirror_z',    1 if p.fold_mirror_z else 0)
        self._set('u_rep_enabled',      1 if p.rep_enabled   else 0)
        self._set('u_rep_cell_x',       p.rep_cell_x)
        self._set('u_rep_cell_y',       p.rep_cell_y)
        self._set('u_rep_cell_z',       p.rep_cell_z)
        # Light primary
        self._set('u_light_dir',       (p.light_x, p.light_y, p.light_z))
        self._set('u_specular_power',  p.specular_power)
        self._set('u_specular_strength', p.specular_strength)
        self._set('u_ambient',         p.ambient)
        self._set('u_subsurface',      p.subsurface)
        self._set('u_fresnel_power',   p.fresnel_power)
        # Second light
        self._set('u_light2_dir',      (p.light2_x, p.light2_y, p.light2_z))
        self._set('u_light2_color',    (p.light2_r, p.light2_g, p.light2_b))
        self._set('u_light2_strength', p.light2_strength)
        # Color animation
        self._set('u_color_anim_speed', p.color_anim_speed)
        self._set('u_color_offset',    p.color_offset)
        # Raymarching
        self._set('u_step_scale',      p.step_scale)
        self._set('u_normal_eps',      p.normal_eps)
        self._set('u_reflection',      p.reflection)
        self._set('u_max_steps',       p.max_steps)
        self._set('u_max_dist',        p.max_dist)
        self._set('u_hit_eps',         p.hit_eps)
        self._set('u_shadow_steps',    p.shadow_steps)
        self._set('u_shadow_mint',     p.shadow_mint)
        self._set('u_shadow_maxt',     p.shadow_maxt)
        self._set('u_ao_step_scale',   p.ao_step_scale)
        self._set('u_rm_overrelax',    1 if p.rm_overrelax else 0)
        self._set('u_overrelax_factor', p.overrelax_factor)
        # DOF
        self._set('u_feat_ao',           1 if p.feat_ao           else 0)
        self._set('u_feat_shadows',      1 if p.feat_shadows      else 0)
        self._set('u_feat_normals_full', 1 if p.feat_normals_full else 0)
        self._set('u_feat_second_light', 1 if p.feat_second_light else 0)
        self._set('u_feat_fog',          1 if p.feat_fog          else 0)
        self._set('u_feat_glow',         1 if p.feat_glow         else 0)
        self._set('u_feat_reflection',   1 if p.feat_reflection   else 0)
        self._set('u_feat_subsurface',   1 if p.feat_subsurface   else 0)
        self._set('u_feat_orbit_trap',   1 if p.feat_orbit_trap   else 0)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        self.ctx.screen.use()
        self.ctx.clear(0, 0, 0)
        self._scene_tex.use(location=0)
        if 'u_scene' in self.post_prog:
            self.post_prog['u_scene'].value = 0
        if 'u_resolution' in self.post_prog:
            self.post_prog['u_resolution'].value = self.wnd.size
        def _pset(name, val):
            if name in self.post_prog:
                self.post_prog[name].value = val
        _pset('u_gamma',           p.gamma)
        _pset('u_exposure',        p.exposure)
        _pset('u_saturation',      p.saturation)
        self.post_vao.render(moderngl.TRIANGLE_STRIP)

        if self._pending_screenshot or p.screenshot_requested:
            self._pending_screenshot  = False
            p.screenshot_requested    = False
            self._save_screenshot()

        self._render_debug_overlay()

    def _save_screenshot(self):
        try:
            import datetime
            w, h = self.wnd.size
            data = self.ctx.screen.read(components=3)
            from PIL import Image
            img  = Image.frombytes('RGB', (w, h), data)
            img  = img.transpose(Image.FLIP_TOP_BOTTOM)
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = Path(__file__).parent / f'fractal_{ts}.png'
            img.save(str(path))
            print(f'[screenshot] saved → {path}')
            if callable(getattr(_params, '_on_screenshot', None)):
                _params._on_screenshot(str(path))
        except Exception as e:
            print(f'[screenshot] failed: {e}')

COLORS = {
    'bg':     '#1a1a2e',
    'bg2':    '#12122a',
    'panel':  '#2d2d5e',
    'accent': '#7b68ee',
    'fg':     '#e0e0ff',
    'fg2':    '#8888cc',
    'fg3':    '#aaaacc',
    'fg4':    '#6060aa',
}
FONT_MONO  = QFont('Segoe UI', 10)
FONT_SMALL = QFont('Segoe UI', 9)
FONT_TITLE = QFont('Segoe UI', 13)
FONT_TITLE.setBold(True)
FONT_BOLD  = QFont('Segoe UI', 9)
FONT_BOLD.setBold(True)

def _apply_palette(app: QApplication):
    pass
    pal = QPalette()
    bg  = QColor(COLORS['bg'])
    fg  = QColor(COLORS['fg'])
    acc = QColor(COLORS['accent'])
    pal.setColor(QPalette.Window,          bg)
    pal.setColor(QPalette.WindowText,      fg)
    pal.setColor(QPalette.Base,            QColor(COLORS['panel']))
    pal.setColor(QPalette.AlternateBase,   bg)
    pal.setColor(QPalette.Text,            fg)
    pal.setColor(QPalette.Button,          QColor(COLORS['panel']))
    pal.setColor(QPalette.ButtonText,      fg)
    pal.setColor(QPalette.Highlight,       acc)
    pal.setColor(QPalette.HighlightedText, fg)
    app.setPalette(pal)


def _css_groupbox() -> str:
    return f"""
        QGroupBox {{
            color: {COLORS['accent']};
            font: bold 9pt Consolas;
            border: 1px solid {COLORS['panel']};
            border-radius: 4px;
            margin-top: 8px;
            padding-top: 6px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 8px;
        }}
    """

def _css_check() -> str:
    return f"""
        QCheckBox {{ color: {COLORS['fg']}; font: 9pt Consolas; spacing: 6px; }}
        QCheckBox::indicator {{
            width: 14px; height: 14px;
            background: {COLORS['panel']};
            border: 1px solid {COLORS['accent']};
            border-radius: 3px;
        }}
        QCheckBox::indicator:checked {{ background: {COLORS['accent']}; }}
    """

def _css_radio() -> str:
    return f"""
        QRadioButton {{ color: {COLORS['fg']}; font: 9pt Consolas; spacing: 6px; }}
        QRadioButton::indicator {{
            width: 14px; height: 14px;
            background: {COLORS['panel']};
            border: 1px solid {COLORS['accent']};
            border-radius: 7px;
        }}
        QRadioButton::indicator:checked {{ background: {COLORS['accent']}; }}
    """

def _css_button(bg=None, hover=None) -> str:
    bg    = bg    or COLORS['panel']
    hover = hover or COLORS['accent']
    return f"""
        QPushButton {{
            background: {bg}; color: {COLORS['fg']};
            font: 8pt Consolas; border: none;
            border-radius: 4px; padding: 4px 8px;
        }}
        QPushButton:hover {{ background: {hover}; }}
        QPushButton:pressed {{ background: {COLORS['accent']}; }}
    """

def _label(text, color=None, font=None) -> QLabel:
    lbl = QLabel(text)
    lbl.setFont(font or FONT_MONO)
    lbl.setStyleSheet(f"color: {color or COLORS['fg']}; background: transparent;")
    return lbl

def _section(title) -> QGroupBox:
    grp = QGroupBox(f' {title} ')
    grp.setStyleSheet(_css_groupbox())
    grp.setFont(FONT_MONO)
    return grp

def _lbl_hint(layout, text: str):
    lbl = _label(text, COLORS['fg4'], FONT_SMALL)
    lbl.setStyleSheet(
        f"color: {COLORS['fg4']}; background: transparent;"
        "padding: 0px 4px; font-style: italic;"
    )
    layout.addWidget(lbl)

class SliderSmoother:
    TICK_MS   = 16
    SMOOTHING = 10.0

    def __init__(self, initial: float, on_update):
        self._target  = initial
        self._current = initial
        self._cb      = on_update
        self._timer   = QTimer()
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    def set_target(self, v: float):
        self._target = v
        if not self._timer.isActive():
            self._timer.start()

    def set_immediate(self, v: float):
        self._target  = v
        self._current = v
        self._timer.stop()
        self._cb(v)

    def _tick(self):
        alpha = 1.0 - math.exp(-self.SMOOTHING * self.TICK_MS * 0.001)
        self._current += (self._target - self._current) * alpha
        if abs(self._current - self._target) < 1e-5:
            self._current = self._target
            self._timer.stop()
        self._cb(self._current)


class SliderRow(QWidget):
    SLIDER_STEPS = 1000
    def __init__(self, label: str, mn: float, mx: float,
                 value: float, step: float = 0.01, parent=None):
        super().__init__(parent)
        self._mn, self._mx, self._step = mn, mx, step
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        lbl = _label(label, COLORS['fg2'], FONT_SMALL)
        lbl.setFixedWidth(80)
        lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self.SLIDER_STEPS)
        self._val_lbl = _label(f'{value:.2f}', COLORS['fg'], FONT_SMALL)
        self._val_lbl.setFixedWidth(50)
        self._val_lbl.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(lbl)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._val_lbl)
        self._callbacks = []
        self._smoother = SliderSmoother(value, self._dispatch)
        self._slider.valueChanged.connect(self._on_slider_moved)
        self.set_value(value)

    def _float_to_int(self, v: float) -> int:
        return round((v - self._mn) / (self._mx - self._mn) * self.SLIDER_STEPS)

    def _int_to_float(self, i: int) -> float:
        raw = self._mn + i / self.SLIDER_STEPS * (self._mx - self._mn)
        return round(raw / self._step) * self._step

    def _dispatch(self, v: float):
        self._val_lbl.setText(f'{v:.2f}')
        for cb in self._callbacks:
            cb(v)

    def set_value(self, v: float):
        self._slider.blockSignals(True)
        self._slider.setValue(self._float_to_int(v))
        self._slider.blockSignals(False)
        self._smoother.set_immediate(v)

    def get_value(self) -> float:
        return self._int_to_float(self._slider.value())

    def _on_slider_moved(self, _):
        self._smoother.set_target(self.get_value())

    def on_change(self, cb):
        self._callbacks.append(cb)

class ControlGUI(QMainWindow):
    CAM_SYNC_MS = 80
    def __init__(self):
        super().__init__()
        self.setWindowTitle("IFS Parameters")
        self.resize(1020, 860)
        self.setStyleSheet(f"QMainWindow, QWidget {{ background: {COLORS['bg']}; }}")
        self._feat_sections: dict = {}
        self._build()
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._sync_camera_ui)
        self._cam_timer.start(self.CAM_SYNC_MS)
    def _register_feat_section(self, feat_attr: str, widget):
        self._feat_sections.setdefault(feat_attr, []).append(widget)
        widget.setVisible(getattr(_params, feat_attr, True))

    def _make_scroll_tab(self) -> tuple:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        container = QWidget()
        container.setStyleSheet(f"background: {COLORS['bg']};")
        vbox = QVBoxLayout(container)
        vbox.setSpacing(4)
        vbox.setContentsMargins(8, 8, 8, 8)
        scroll.setWidget(container)
        return scroll, vbox

    def _build(self):
        root = QWidget()
        root.setStyleSheet(f"background: {COLORS['bg']};")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(4)

        title = _label("KALEIDOSCOPIC IFS", COLORS['accent'], FONT_TITLE)
        title.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(title)

        sub = _label("Ray-marched fractal renderer", COLORS['fg4'], FONT_SMALL)
        sub.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(sub)

        hint_text = (
            "LMB drag -> look   Scroll -> fly fwd/bwd\n"
            "W/S -> fwd/bwd   A/D -> strafe   Q/E -> up/dn\n"
            "Shift -> x5 faster   Alt -> x0.2 slower"
        )
        hint = _label(hint_text, COLORS['fg2'], FONT_SMALL)
        hint.setStyleSheet(
            f"color: {COLORS['fg2']}; background: {COLORS['bg2']};"
            "padding: 6px 8px; border-radius: 4px;"
        )
        root_layout.addWidget(hint)

        tabs = QTabWidget()
        tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid {COLORS['panel']};
                background: {COLORS['bg']};
            }}
            QTabBar::tab {{
                background: {COLORS['bg2']};
                color: {COLORS['fg2']};
                font: 8pt Consolas;
                padding: 5px 10px;
                border: 1px solid {COLORS['panel']};
                border-bottom: none;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background: {COLORS['panel']};
                color: {COLORS['accent']};
            }}
            QTabBar::tab:hover {{
                background: {COLORS['panel']};
            }}
        """)
        root_layout.addWidget(tabs, 1)

        tab_fractal, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_fractal_section()
        self._build_ifs_section()
        self._build_transform_section()
        self._vbox.addStretch()
        tabs.addTab(tab_fractal, "Fractal")

        tab_camera, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_camera_section()
        self._vbox.addStretch()
        tabs.addTab(tab_camera, "Camera")

        tab_color, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_color_section()
        self._build_background_section()
        self._vbox.addStretch()
        tabs.addTab(tab_color, "Color / BG")

        tab_light, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_lighting_section()
        self._build_light_direction_section()
        self._build_second_light_section()
        self._build_glow_section()
        self._vbox.addStretch()
        tabs.addTab(tab_light, "Lighting")

        tab_render, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_raymarching_section()
        self._build_postprocess_section()
        self._build_aa_section()
        self._vbox.addStretch()
        tabs.addTab(tab_render, "Rendering")

        tab_perf, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_performance_section()
        self._vbox.addStretch()
        tabs.addTab(tab_perf, "Performance")

        tab_presets, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_presets_section()
        self._vbox.addStretch()
        tabs.addTab(tab_presets, "Presets")


        tab_saves, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_saves_section()
        self._vbox.addStretch()
        tabs.addTab(tab_saves, "Saves")

        tab_player, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_player_section()
        self._vbox.addStretch()
        tabs.addTab(tab_player, "Player")

        tab_infinite, vbox = self._make_scroll_tab()
        self._vbox = vbox
        self._build_infinite_section()
        self._vbox.addStretch()
        tabs.addTab(tab_infinite, "Infinite")

        self.setCentralWidget(root)

    def _add_section(self, widget):
        self._vbox.addWidget(widget)
    def _build_fractal_section(self):
        grp = _section("FRACTAL TYPE")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        self._type_grp = QButtonGroup(self)
        row = QHBoxLayout()
        for i, name in enumerate(["Mandelbox", "Menger Sponge", "Sierpinski", "Octahedron IFS", "Mandelbulb", "Kleinian"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_MONO)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.fractal_type)
            self._type_grp.addButton(rb, i)
            if i % 2 == 0 and i > 0:
                layout.addLayout(row)
                row = QHBoxLayout()
            row.addWidget(rb)
        layout.addLayout(row)
        self._type_grp.idClicked.connect(lambda idx: self._on_fractal_type_changed(idx) if hasattr(self, '_fractal_panels') else setattr(_params, 'fractal_type', idx))
        iter_row = QWidget()
        iter_layout = QHBoxLayout(iter_row)
        iter_layout.setContentsMargins(0, 0, 0, 0)
        iter_lbl = _label("Iterations", COLORS['fg2'], FONT_SMALL)
        iter_lbl.setFixedWidth(80)
        iter_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._iter_slider = QSlider(Qt.Horizontal)
        self._iter_slider.setRange(1, 64)
        self._iter_slider.setValue(_params.iterations)
        self._iter_val_lbl = _label(str(_params.iterations), COLORS['fg'], FONT_SMALL)
        self._iter_val_lbl.setFixedWidth(50)
        self._iter_slider.valueChanged.connect(self._on_iter_change)
        iter_layout.addWidget(iter_lbl)
        iter_layout.addWidget(self._iter_slider, 1)
        iter_layout.addWidget(self._iter_val_lbl)
        layout.addWidget(iter_row)
        self._add_section(grp)
    def _on_iter_change(self, v):
        _params.iterations = v
        self._iter_val_lbl.setText(str(v))
    def _build_ifs_section(self):
        base_grp = _section("BASE PARAMETERS")
        base_layout = QVBoxLayout(base_grp)
        base_layout.setSpacing(2)
        for label, attr, mn, mx, val, step in [
            ("Scale",      'scale',        -3.0, 3.0,  _params.scale,        0.01),
            ("Bailout",    'bailout',       1.0, 50.0, _params.bailout,       0.1),
            ("Min Dist",   'min_dist',      0.1, 5.0,  _params.min_dist,      0.1),
            ("DE Mult",    'de_multiplier', 0.1, 3.0,  _params.de_multiplier, 0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            base_layout.addWidget(sr)
        trap_lbl = _label("Orbit trap shape:", COLORS['fg3'], FONT_SMALL)
        base_layout.addWidget(trap_lbl)
        trap_row = QHBoxLayout()
        self._trap_grp = QButtonGroup(self)
        for i, name in enumerate(["Sphere", "Plane", "Cube", "Torus"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.orbit_trap_type)
            self._trap_grp.addButton(rb, i)
            trap_row.addWidget(rb)
        self._trap_grp.idClicked.connect(lambda idx: setattr(_params, 'orbit_trap_type', idx))
        base_layout.addLayout(trap_row)
        self._add_section(base_grp)

        self._fractal_panels = {}

        mb = _section("MANDELBOX FINE-TUNE")
        mb_l = QVBoxLayout(mb)
        mb_l.setSpacing(2)
        _lbl_hint(mb_l, "Box fold per axis, sphere fold radii, Julia mode")
        for label, attr, mn, mx, val, step in [
            ("Fold Limit",    'mb_fold_limit',   0.1, 3.0,  _params.mb_fold_limit,   0.01),
            ("Fold X",        'mb_fold_x',       0.0, 3.0,  _params.mb_fold_x,       0.01),
            ("Fold Y",        'mb_fold_y',       0.0, 3.0,  _params.mb_fold_y,       0.01),
            ("Fold Z",        'mb_fold_z',       0.0, 3.0,  _params.mb_fold_z,       0.01),
            ("Sph Inner r²",  'mb_sphere_inner', 0.01,2.0,  _params.mb_sphere_inner, 0.005),
            ("Sph Outer r²",  'mb_sphere_outer', 0.1, 5.0,  _params.mb_sphere_outer, 0.01),
            ("Fixed Radius",  'mb_fixed_radius', 0.1, 4.0,  _params.mb_fixed_radius, 0.01),
            ("Color Scale",   'mb_color_scale',  0.01,5.0,  _params.mb_color_scale,  0.01),
            ("Rot/Iter",      'mb_rot_per_iter', 0.0, 0.5,  _params.mb_rot_per_iter, 0.002),
            ("Julia X",       'julia_x',        -20.0,20.0, _params.julia_x,         0.01),
            ("Julia Y",       'julia_y',        -20.0,20.0, _params.julia_y,         0.01),
            ("Julia Z",       'julia_z',        -20.0,20.0, _params.julia_z,         0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            mb_l.addWidget(sr)
        fold_mode_lbl = _label("Box fold mode:", COLORS['fg3'], FONT_SMALL)
        mb_l.addWidget(fold_mode_lbl)
        fold_mode_row = QHBoxLayout()
        self._fold_mode_grp = QButtonGroup(self)
        for i, name in enumerate(["Clamp", "Abs", "Sin"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.mb_fold_mode)
            self._fold_mode_grp.addButton(rb, i)
            fold_mode_row.addWidget(rb)
        self._fold_mode_grp.idClicked.connect(lambda idx: setattr(_params, 'mb_fold_mode', idx))
        mb_l.addLayout(fold_mode_row)
        julia_mode_lbl = _label("Julia mode:", COLORS['fg3'], FONT_SMALL)
        mb_l.addWidget(julia_mode_lbl)
        julia_mode_row = QHBoxLayout()
        self._mb_julia_mode_grp = QButtonGroup(self)
        for i, name in enumerate(["Orbit (free)", "Fixed Julia"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.mb_julia_mode)
            self._mb_julia_mode_grp.addButton(rb, i)
            julia_mode_row.addWidget(rb)
        self._mb_julia_mode_grp.idClicked.connect(lambda idx: setattr(_params, 'mb_julia_mode', idx))
        mb_l.addLayout(julia_mode_row)
        folds_lbl = _label("Axis folds:", COLORS['fg3'], FONT_SMALL)
        mb_l.addWidget(folds_lbl)
        folds_row = QHBoxLayout()
        self._fold_checks = {}
        for axis in ['X', 'Y', 'Z']:
            cb = QCheckBox(f'Fold {axis}')
            cb.setFont(FONT_MONO)
            cb.setStyleSheet(_css_check())
            cb.setChecked(getattr(_params, f'fold_{axis.lower()}'))
            cb.stateChanged.connect(
                lambda state, a=axis: setattr(_params, f'fold_{a.lower()}', bool(state))
            )
            self._fold_checks[axis] = cb
            folds_row.addWidget(cb)
        mb_l.addLayout(folds_row)
        self._fractal_panels[0] = mb
        self._add_section(mb)

        ms = _section("MENGER SPONGE FINE-TUNE")
        ms_l = QVBoxLayout(ms)
        ms_l.setSpacing(2)
        _lbl_hint(ms_l, "IFS scale per axis, cross gap, twist per axis")
        for label, attr, mn, mx, val, step in [
            ("IFS Scale",    'ms_scale',       2.0, 5.0, _params.ms_scale,       0.01),
            ("Scale Y",      'ms_scale_y',     0.0, 5.0, _params.ms_scale_y,     0.01),
            ("Scale Z",      'ms_scale_z',     0.0, 5.0, _params.ms_scale_z,     0.01),
            ("IFS Offset",   'ms_offset',      1.0, 4.0, _params.ms_offset,      0.01),
            ("Cross Width",  'ms_cross_width', 0.0, 4.0, _params.ms_cross_width, 0.01),
            ("Twist Y",      'ms_twist',       0.0, 1.3, _params.ms_twist,       0.001),
            ("Twist X",      'ms_rot_x',       0.0, 1.3, _params.ms_rot_x,       0.001),
            ("Twist Z",      'ms_rot_z',       0.0, 1.3, _params.ms_rot_z,       0.001),
            ("Edge Sharp",   'ms_sharpness',   0.1, 1.0, _params.ms_sharpness,   0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            ms_l.addWidget(sr)
        self._fractal_panels[1] = ms
        self._add_section(ms)

        si = _section("SIERPINSKI FINE-TUNE")
        si_l = QVBoxLayout(si)
        si_l.setSpacing(2)
        _lbl_hint(si_l, "Vertex spread, fold bias, squash, twist per axis")
        for label, attr, mn, mx, val, step in [
            ("Vertex Spread", 'si_vertex_spread', 0.2, 3.0, _params.si_vertex_spread, 0.01),
            ("Fold Bias",     'si_fold_bias',      1.2, 4.0, _params.si_fold_bias,     0.01),
            ("Y Squash",      'si_squash',         0.2, 3.0, _params.si_squash,        0.01),
            ("Twist Y",       'si_twist',          0.0, 1.3, _params.si_twist,         0.001),
            ("Twist X",       'si_rot_x',          0.0, 1.3, _params.si_rot_x,         0.001),
            ("Twist Z",       'si_rot_z',          0.0, 1.3, _params.si_rot_z,         0.001),
            ("Vtx Jitter",    'si_vertex_jitter',  0.0, 1.0, _params.si_vertex_jitter, 0.005),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            si_l.addWidget(sr)
        self._fractal_panels[2] = si
        self._add_section(si)

        oc = _section("OCTAHEDRON IFS FINE-TUNE")
        oc_l = QVBoxLayout(oc)
        oc_l.setSpacing(2)
        _lbl_hint(oc_l, "IFS scale, per-axis offset, rotation per axis, norm sharpness")
        for label, attr, mn, mx, val, step in [
            ("IFS Scale",     'oc_ifs_scale',   1.2, 4.0,  _params.oc_ifs_scale,   0.01),
            ("Offset Uni",    'oc_offset_uni',  0.1, 3.0,  _params.oc_offset_uni,  0.01),
            ("Offset X",      'oc_offset_x',    0.0, 4.0,  _params.oc_offset_x,    0.01),
            ("Offset Y",      'oc_offset_y',    0.0, 4.0,  _params.oc_offset_y,    0.01),
            ("Offset Z",      'oc_offset_z',    0.0, 4.0,  _params.oc_offset_z,    0.01),
            ("Twist Y",       'oc_twist',       0.0, 1.55,  _params.oc_twist,       0.001),
            ("Twist X",       'oc_rot_x',       0.0, 1.55,  _params.oc_rot_x,       0.001),
            ("Twist Z",       'oc_rot_z',       0.0, 1.55,  _params.oc_rot_z,       0.001),
            ("Norm Sharp",    'oc_sharpness',   0.5, 4.0,  _params.oc_sharpness,   0.05),
            ("Fold Amount",   'oc_fold_amount', 0.0, 1.0,  _params.oc_fold_amount, 0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            oc_l.addWidget(sr)
        self._fractal_panels[3] = oc
        self._add_section(oc)

        mb2 = _section("MANDELBULB FINE-TUNE")
        mb2_l = QVBoxLayout(mb2)
        mb2_l.setSpacing(2)
        _lbl_hint(mb2_l, "Power N (2=sphere, 8=classic), Julia mode, pre-fold")
        for label, attr, mn, mx, val, step in [
            ("Power N",       'mb2_power',        2.0, 16.0, _params.mb2_power,        0.05),
            ("Bailout",       'mb2_bailout',       1.0, 10.0, _params.mb2_bailout,      0.1),
            ("Julia X",       'mb2_julia_x',      -2.0, 2.0,  _params.mb2_julia_x,      0.005),
            ("Julia Y",       'mb2_julia_y',      -2.0, 2.0,  _params.mb2_julia_y,      0.005),
            ("Julia Z",       'mb2_julia_z',      -2.0, 2.0,  _params.mb2_julia_z,      0.005),
            ("Fold Strength", 'mb2_fold_strength', 0.0, 2.0,  _params.mb2_fold_strength, 0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            mb2_l.addWidget(sr)
        julia_lbl = _label("Julia mode:", COLORS['fg3'], FONT_SMALL)
        mb2_l.addWidget(julia_lbl)
        julia_row = QHBoxLayout()
        self._mb2_julia_grp = QButtonGroup(self)
        for i, name in enumerate(["Mandelbulb", "Julia set"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.mb2_julia_mode)
            self._mb2_julia_grp.addButton(rb, i)
            julia_row.addWidget(rb)
        self._mb2_julia_grp.idClicked.connect(lambda idx: setattr(_params, 'mb2_julia_mode', idx))
        mb2_l.addLayout(julia_row)
        fold_lbl2 = _label("Pre-fold type:", COLORS['fg3'], FONT_SMALL)
        mb2_l.addWidget(fold_lbl2)
        fold_row2 = QHBoxLayout()
        self._mb2_fold_grp = QButtonGroup(self)
        for i, name in enumerate(["None", "Box", "Abs"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.mb2_fold_type)
            self._mb2_fold_grp.addButton(rb, i)
            fold_row2.addWidget(rb)
        self._mb2_fold_grp.idClicked.connect(lambda idx: setattr(_params, 'mb2_fold_type', idx))
        mb2_l.addLayout(fold_row2)
        self._fractal_panels[4] = mb2
        self._add_section(mb2)

        kl = _section("PSEUDO-KLEINIAN FINE-TUNE")
        kl_l = QVBoxLayout(kl)
        kl_l.setSpacing(2)
        _lbl_hint(kl_l, "Sphere inversion IFS. Scale, C offset, fold limit, sphere radius.")
        for label, attr, mn, mx, val, step in [
            ("Scale",        'kl_scale',        0.5,  3.0,  _params.kl_scale,        0.01),
            ("C x",          'kl_cx',          -2.0,  2.0,  _params.kl_cx,           0.005),
            ("C y",          'kl_cy',          -2.0,  2.0,  _params.kl_cy,           0.005),
            ("C z",          'kl_cz',          -2.0,  2.0,  _params.kl_cz,           0.005),
            ("Fold Limit",   'kl_fold_limit',   0.1,  3.0,  _params.kl_fold_limit,   0.01),
            ("Sph Radius",   'kl_sph_radius',   0.05, 2.0,  _params.kl_sph_radius,   0.005),
            ("Rot/Iter",     'kl_rot_per_iter', 0.0,  1.5,  _params.kl_rot_per_iter, 0.002),
            ("DE Mix",       'kl_mix_factor',   0.0,  1.0,  _params.kl_mix_factor,   0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            kl_l.addWidget(sr)
        self._fractal_panels[5] = kl
        self._add_section(kl)

        space_grp = _section("SPACE OPERATORS")
        space_l = QVBoxLayout(space_grp)
        space_l.setSpacing(4)
        _lbl_hint(space_l, "Applied before the fractal. Stacks: repetition then mirrors then twist then warp.")

        rep_hdr = _label("INFINITE REPETITION", COLORS['fg3'], FONT_SMALL)
        space_l.addWidget(rep_hdr)
        rep_en_row = QHBoxLayout()
        self._rep_check = QCheckBox("Enable repetition")
        self._rep_check.setFont(FONT_SMALL)
        self._rep_check.setStyleSheet(_css_check())
        self._rep_check.setChecked(_params.rep_enabled)
        self._rep_check.stateChanged.connect(lambda s: setattr(_params, 'rep_enabled', bool(s)))
        rep_en_row.addWidget(self._rep_check)
        space_l.addLayout(rep_en_row)
        for label, attr, mn, mx, val, step in [
            ("Cell X", 'rep_cell_x', 0.5, 20.0, _params.rep_cell_x, 0.1),
            ("Cell Y", 'rep_cell_y', 0.5, 20.0, _params.rep_cell_y, 0.1),
            ("Cell Z", 'rep_cell_z', 0.5, 20.0, _params.rep_cell_z, 0.1),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            space_l.addWidget(sr)

        mir_hdr = _label("MIRROR PLANES", COLORS['fg3'], FONT_SMALL)
        space_l.addWidget(mir_hdr)
        mir_row = QHBoxLayout()
        for axis in ['x', 'y', 'z']:
            cb = QCheckBox(f'Mirror {axis.upper()}')
            cb.setFont(FONT_SMALL)
            cb.setStyleSheet(_css_check())
            cb.setChecked(getattr(_params, f'fold_mirror_{axis}'))
            cb.stateChanged.connect(
                lambda s, a=axis: setattr(_params, f'fold_mirror_{a}', bool(s))
            )
            setattr(self, f'_mir_{axis}', cb)
            mir_row.addWidget(cb)
        space_l.addLayout(mir_row)

        twist_hdr = _label("TWIST", COLORS['fg3'], FONT_SMALL)
        space_l.addWidget(twist_hdr)
        _lbl_hint(space_l, "Twists space around the chosen axis by amount * coordinate.")
        twist_axis_row = QHBoxLayout()
        twist_axis_lbl = _label("Axis:", COLORS['fg3'], FONT_SMALL)
        twist_axis_lbl.setFixedWidth(36)
        self._twist_axis_grp = QButtonGroup(self)
        for i, name in enumerate(["Y", "X", "Z"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.twist_axis)
            self._twist_axis_grp.addButton(rb, i)
            twist_axis_row.addWidget(rb)
        self._twist_axis_grp.idClicked.connect(lambda idx: setattr(_params, 'twist_axis', idx))
        twist_axis_row.insertWidget(0, twist_axis_lbl)
        space_l.addLayout(twist_axis_row)
        sr_tw = SliderRow("Amount", 0.0, 2.0, _params.twist_amount, 0.005)
        sr_tw.on_change(lambda v: setattr(_params, 'twist_amount', v))
        setattr(self, '_sl_twist_amount', sr_tw)
        space_l.addWidget(sr_tw)

        warp_hdr = _label("DOMAIN WARP", COLORS['fg3'], FONT_SMALL)
        space_l.addWidget(warp_hdr)
        _lbl_hint(space_l, "Sine-based space distortion before fractal evaluation.")
        warp_en_row = QHBoxLayout()
        self._warp_check = QCheckBox("Enable warp")
        self._warp_check.setFont(FONT_SMALL)
        self._warp_check.setStyleSheet(_css_check())
        self._warp_check.setChecked(_params.warp_enabled)
        self._warp_check.stateChanged.connect(lambda s: setattr(_params, 'warp_enabled', bool(s)))
        warp_en_row.addWidget(self._warp_check)
        space_l.addLayout(warp_en_row)
        warp_type_row = QHBoxLayout()
        warp_type_lbl = _label("Type:", COLORS['fg3'], FONT_SMALL)
        warp_type_lbl.setFixedWidth(36)
        self._warp_type_grp = QButtonGroup(self)
        for i, name in enumerate(["Sine", "FBM", "Curl"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.warp_type)
            self._warp_type_grp.addButton(rb, i)
            warp_type_row.addWidget(rb)
        self._warp_type_grp.idClicked.connect(lambda idx: setattr(_params, 'warp_type', idx))
        warp_type_row.insertWidget(0, warp_type_lbl)
        space_l.addLayout(warp_type_row)
        for label, attr, mn, mx, val, step in [
            ("Strength", 'warp_strength', 0.0, 2.0, _params.warp_strength, 0.01),
            ("Frequency","warp_freq",     0.1, 8.0, _params.warp_freq,     0.05),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            space_l.addWidget(sr)

        self._add_section(space_grp)

        # Wire fractal-type selector to show/hide panels
        self._type_grp.idClicked.connect(self._on_fractal_type_changed)
        self._on_fractal_type_changed(_params.fractal_type)

    def _on_fractal_type_changed(self, idx: int):
        setattr(_params, 'fractal_type', idx)
        for ftype, panel in self._fractal_panels.items():
            panel.setVisible(ftype == idx)
    def _build_transform_section(self):
        grp = _section("GLOBAL ROTATION")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        for label, attr, val in [
            ("Rot X", 'rot_x', _params.rot_x),
            ("Rot Y", 'rot_y', _params.rot_y),
            ("Rot Z", 'rot_z', _params.rot_z),
        ]:
            sr = SliderRow(label, 0.0, math.pi * 2, val, 0.01)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._add_section(grp)
    def _build_camera_section(self):
        grp = _section("CAMERA")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)
        self._cam_pos_lbl = QLabel("pos  (0.00, 0.00, 5.00)\nyaw  180.0   pitch  0.0")
        self._cam_pos_lbl.setFont(FONT_SMALL)
        self._cam_pos_lbl.setStyleSheet(
            f"color: {COLORS['fg2']}; background: {COLORS['bg2']};"
            "padding: 4px 6px; border-radius: 3px;"
        )
        layout.addWidget(self._cam_pos_lbl)
        reset_btn = QPushButton("Reset Camera")
        reset_btn.setFont(FONT_SMALL)
        reset_btn.setStyleSheet(_css_button())
        reset_btn.clicked.connect(self._reset_camera)
        layout.addWidget(reset_btn)
        self._sl_move_spd = SliderRow("Move Speed", 0.1, 10.0, FractalWindow.KEY_MOVE_SPD, 0.1)
        self._sl_move_spd.on_change(lambda v: setattr(FractalWindow, 'KEY_MOVE_SPD', v))
        layout.addWidget(self._sl_move_spd)
        self._anim_check = QCheckBox("Auto-rotate (anim)")
        self._anim_check.setFont(FONT_MONO)
        self._anim_check.setStyleSheet(_css_check())
        self._anim_check.setChecked(_params.animate)
        self._anim_check.stateChanged.connect(lambda s: setattr(_params, 'animate', bool(s)))
        layout.addWidget(self._anim_check)
        self._sl_anim_speed = SliderRow("Anim Speed", 0.0, 5.0, _params.anim_speed, 0.01)
        self._sl_anim_speed.on_change(lambda v: setattr(_params, 'anim_speed', v))
        layout.addWidget(self._sl_anim_speed)
        self._sl_fov = SliderRow("FOV", 0.5, 3.0, _params.fov, 0.01)
        self._sl_fov.on_change(lambda v: setattr(_params, 'fov', v))
        layout.addWidget(self._sl_fov)
        self._add_section(grp)
    def _reset_camera(self):
        _params.cam_pos   = [0.0, 0.0, 5.0]
        _params.cam_yaw   = 0.0
        _params.cam_pitch = 0.0
        _params.cam_roll  = 0.0
    def _sync_camera_ui(self):
        try:
            px, py, pz = _params.cam_pos
            yaw   = math.degrees(_params.cam_yaw) % 360
            pitch = math.degrees(_params.cam_pitch)
            self._cam_pos_lbl.setText(
                f"pos  ({px:+.2f}, {py:+.2f}, {pz:+.2f})\n"
                f"yaw  {yaw:.1f}   pitch  {pitch:.1f}"
            )
        except Exception:
            pass
        try:
            lbl = self._player_mode_lbl
            if _params.player_mode:
                lbl.setText("PLAYER MODE: ON")
                lbl.setStyleSheet(
                    f"color: {COLORS['accent']}; background: {COLORS['bg2']};"
                    "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
                )
            else:
                lbl.setText("PLAYER MODE: OFF")
                lbl.setStyleSheet(
                    f"color: {COLORS['fg4']}; background: {COLORS['bg2']};"
                    "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
                )
        except Exception:
            pass
    def _build_color_section(self):
        grp = _section("COLOR & PALETTE")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)
        modes_layout = QGridLayout()
        self._cmode_grp = QButtonGroup(self)
        for i, label in enumerate(["Iteration", "Orbit Trap", "Normal", "Distance"]):
            rb = QRadioButton(label)
            rb.setFont(FONT_MONO)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.color_mode)
            self._cmode_grp.addButton(rb, i)
            modes_layout.addWidget(rb, i // 2, i % 2)
        self._cmode_grp.idClicked.connect(lambda idx: setattr(_params, 'color_mode', idx))
        layout.addLayout(modes_layout)
        self._sl_fog = SliderRow("Fog Density", 0.0, 3.0, _params.fog_density, 0.01)
        self._sl_fog.on_change(lambda v: setattr(_params, 'fog_density', v))
        layout.addWidget(self._sl_fog)
        color_row = QHBoxLayout()
        self._c1_btn = self._make_color_btn("Color A", 'color1', _params.color1)
        self._c2_btn = self._make_color_btn("Color B", 'color2', _params.color2)
        self._c3_btn = self._make_color_btn("Color C", 'color3', _params.color3)
        color_row.addWidget(self._c1_btn)
        color_row.addWidget(self._c2_btn)
        color_row.addWidget(self._c3_btn)
        layout.addLayout(color_row)
        self._add_section(grp)
    def _make_color_btn(self, label: str, attr: str, rgb) -> QPushButton:
        btn = QPushButton(label)
        btn.setFont(FONT_BOLD)
        hex_col = self._rgb_to_hex(rgb)
        btn.setStyleSheet(
            f"QPushButton {{ background: {hex_col}; color: white; border: none;"
            "border-radius: 4px; padding: 4px 8px; font: bold 8pt Consolas; }}"
            f"QPushButton:hover {{ border: 2px solid {COLORS['accent']}; }}"
        )
        btn.clicked.connect(lambda _, a=attr, b=btn: self._pick_color(a, b))
        return btn
    def _rgb_to_hex(self, rgb) -> str:
        r, g, b = [int(c * 255) for c in rgb]
        return f'#{r:02x}{g:02x}{b:02x}'
    def _pick_color(self, attr: str, btn: QPushButton):
        cur = QColor(self._rgb_to_hex(getattr(_params, attr)))
        col = QColorDialog.getColor(cur, self, f"Pick {attr}")
        if col.isValid():
            rgb = (col.redF(), col.greenF(), col.blueF())
            setattr(_params, attr, rgb)
            hex_col = self._rgb_to_hex(rgb)
            btn.setStyleSheet(
                f"QPushButton {{ background: {hex_col}; color: white; border: none;"
                "border-radius: 4px; padding: 4px 8px; font: bold 8pt Consolas; }}"
                f"QPushButton:hover {{ border: 2px solid {COLORS['accent']}; }}"
            )
    def _build_glow_section(self):
        grp = _section("GLOW & EMISSION")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        _lbl_hint(layout, "Volumetric halo, rim-light and surface emission")
        for label, attr, mn, mx, val, step in [
            ("Intensity",    'glow_intensity', 0.0, 20.0, _params.glow_intensity, 0.1),
            ("Falloff",      'glow_falloff',   0.5, 30.0, _params.glow_falloff,   0.1),
            ("Core Radius",  'glow_radius',    0.1,  5.0, _params.glow_radius,    0.05),
            ("Rim Strength", 'rim_strength',   0.0,  3.0, _params.rim_strength,   0.02),
            ("Emission",     'emission',       0.0,  3.0, _params.emission,       0.02),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._add_section(grp)
        self._register_feat_section('feat_glow', grp)

    def _build_background_section(self):
        grp = _section("BACKGROUND")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)
        # mode selector
        mode_row = QHBoxLayout()
        self._bg_mode_grp = QButtonGroup(self)
        for i, lbl in enumerate(["Flat", "Gradient", "Nebula", "Stars"]):
            rb = QRadioButton(lbl)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _params.bg_mode)
            self._bg_mode_grp.addButton(rb, i)
            mode_row.addWidget(rb)
        self._bg_mode_grp.idClicked.connect(lambda idx: setattr(_params, 'bg_mode', idx))
        layout.addLayout(mode_row)
        # colour pickers
        col_row = QHBoxLayout()
        self._bg_c1_btn = self._make_color_btn("Horizon", 'bg_color1', _params.bg_color1)
        self._bg_c2_btn = self._make_color_btn("Nadir",   'bg_color2', _params.bg_color2)
        col_row.addWidget(self._bg_c1_btn)
        col_row.addWidget(self._bg_c2_btn)
        layout.addLayout(col_row)
        self._add_section(grp)

    def _build_aa_section(self):
        grp = _section("ANTI-ALIASING  &  SCREENSHOT")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)
        _lbl_hint(layout, "More samples = sharper edges, slower render")
        aa_row = QHBoxLayout()
        self._aa_grp = QButtonGroup(self)
        for i, lbl in enumerate(["Off (1x)", "RGSS 4x", "Grid 9x"]):
            rb = QRadioButton(lbl)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i + 1 == _params.aa_samples)
            self._aa_grp.addButton(rb, i + 1)
            aa_row.addWidget(rb)
        self._aa_grp.idClicked.connect(lambda idx: setattr(_params, 'aa_samples', idx))
        layout.addLayout(aa_row)

        # Screenshot button
        sep = QLabel()
        sep.setFixedHeight(4)
        layout.addWidget(sep)
        ss_row = QHBoxLayout()
        ss_btn = QPushButton("📷  Save Screenshot  (F12)")
        ss_btn.setFont(FONT_BOLD)
        ss_btn.setStyleSheet(_css_button(COLORS['panel'], COLORS['accent']))
        ss_btn.clicked.connect(self._trigger_screenshot)
        ss_row.addWidget(ss_btn)
        self._ss_status = _label("", COLORS['fg4'], FONT_SMALL)
        ss_row.addWidget(self._ss_status)
        layout.addLayout(ss_row)
        # register callback for status update
        _params._on_screenshot = self._on_screenshot_saved
        self._add_section(grp)

    def _trigger_screenshot(self):
        _params.screenshot_requested = True
        self._ss_status.setText("capturing…")

    def _on_screenshot_saved(self, path: str):
        fname = Path(path).name
        self._ss_status.setText(f"✓ {fname}")

    def _build_lighting_section(self):
        grp = _section("LIGHTING & FOG")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)

        ao_grp = _section("AMBIENT OCCLUSION")
        ao_l = QVBoxLayout(ao_grp)
        ao_l.setSpacing(2)
        for label, attr, mn, mx, val, step in [
            ("AO Strength", 'ao_strength', 0.0, 30.0, _params.ao_strength, 0.01),
            ("AO Radius",   'ao_radius',   0.01, 1.0,  _params.ao_radius,   0.005),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            ao_l.addWidget(sr)
        ao_row = QHBoxLayout()
        ao_lbl = _label("AO samples:", COLORS['fg3'], FONT_SMALL)
        ao_lbl.setFixedWidth(80)
        ao_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._ao_samples_grp = QButtonGroup(self)
        for i, n in enumerate([3, 5, 8, 12]):
            rb = QRadioButton(str(n))
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(n == _params.ao_samples)
            self._ao_samples_grp.addButton(rb, n)
            ao_row.addWidget(rb)
        self._ao_samples_grp.idClicked.connect(lambda idx: setattr(_params, 'ao_samples', idx))
        ao_row.insertWidget(0, ao_lbl)
        ao_l.addLayout(ao_row)
        layout.addWidget(ao_grp)
        self._register_feat_section('feat_ao', ao_grp)

        shadow_grp = _section("SOFT SHADOWS")
        shadow_l = QVBoxLayout(shadow_grp)
        shadow_l.setSpacing(2)
        sr_shadow = SliderRow("Shadow Soft", 1.0, 32.0, _params.shadow_soft, 0.5)
        sr_shadow.on_change(lambda v: setattr(_params, 'shadow_soft', v))
        setattr(self, '_sl_shadow_soft', sr_shadow)
        shadow_l.addWidget(sr_shadow)
        self._shadows_check = QCheckBox("Soft Shadows")
        self._shadows_check.setFont(FONT_MONO)
        self._shadows_check.setStyleSheet(_css_check())
        self._shadows_check.setChecked(_params.shadows)
        self._shadows_check.stateChanged.connect(lambda s: setattr(_params, 'shadows', bool(s)))
        shadow_l.addWidget(self._shadows_check)
        layout.addWidget(shadow_grp)
        self._register_feat_section('feat_shadows', shadow_grp)

        fog_grp = _section("FOG")
        fog_l = QVBoxLayout(fog_grp)
        fog_l.setSpacing(2)
        sr_fog_d = SliderRow("Fog Density", 0.0, 5.0, _params.fog_density, 0.01)
        sr_fog_d.on_change(lambda v: setattr(_params, 'fog_density', v))
        setattr(self, '_sl_fog_density', sr_fog_d)
        fog_l.addWidget(sr_fog_d)
        fog_row = QHBoxLayout()
        fog_lbl = _label("Fog color:", COLORS['fg3'], FONT_SMALL)
        self._fog_btn = QPushButton()
        hex_col = self._rgb_to_hex(_params.fog_color)
        self._fog_btn.setStyleSheet(
            f"QPushButton {{ background: {hex_col}; border: none; border-radius: 4px; padding: 4px 8px; }}"
            f"QPushButton:hover {{ border: 2px solid {COLORS['accent']}; }}"
        )
        self._fog_btn.setFixedHeight(24)
        self._fog_btn.clicked.connect(lambda: self._pick_color('fog_color', self._fog_btn))
        fog_row.addWidget(fog_lbl)
        fog_row.addWidget(self._fog_btn)
        fog_row.addStretch()
        fog_l.addLayout(fog_row)
        layout.addWidget(fog_grp)
        self._register_feat_section('feat_fog', fog_grp)

        sr_glow = SliderRow("Glow", 0.0, 10.0, _params.glow, 0.1)
        sr_glow.on_change(lambda v: setattr(_params, 'glow', v))
        setattr(self, '_sl_glow', sr_glow)
        layout.addWidget(sr_glow)

        self._add_section(grp)
    def _build_light_direction_section(self):
        grp = _section("LIGHT & SPECULAR")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        _lbl_hint(layout, "World-space light direction and surface response")
        for label, attr, mn, mx, val, step in [
            ("Light X",    'light_x',           -5.0, 5.0,  _params.light_x,           0.01),
            ("Light Y",    'light_y',           -5.0, 5.0,  _params.light_y,           0.01),
            ("Light Z",    'light_z',           -5.0, 5.0,  _params.light_z,           0.01),
            ("Spec Power", 'specular_power',     1.0, 128.0, _params.specular_power,    0.5),
            ("Spec Str",   'specular_strength',  0.0, 2.0,   _params.specular_strength, 0.01),
            ("Ambient",    'ambient',            0.0, 1.0,   _params.ambient,           0.01),
            ("Fresnel Pow","fresnel_power",       0.5, 15.0,  _params.fresnel_power,     0.1),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)

        sss_grp = _section("SUBSURFACE SCATTER")
        sss_l = QVBoxLayout(sss_grp)
        sss_l.setSpacing(2)
        sr_sss = SliderRow("Subsurface", 0.0, 2.0, _params.subsurface, 0.01)
        sr_sss.on_change(lambda v: setattr(_params, 'subsurface', v))
        setattr(self, '_sl_subsurface', sr_sss)
        sss_l.addWidget(sr_sss)
        layout.addWidget(sss_grp)
        self._register_feat_section('feat_subsurface', sss_grp)

        self._add_section(grp)

    def _build_raymarching_section(self):
        grp = _section("RAYMARCHING & COLOR")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)

        march_grp = _section("MARCH LOOP")
        march_l = QVBoxLayout(march_grp)
        march_l.setSpacing(2)
        _lbl_hint(march_l,
            "Max Steps: budget per ray.  Step Scale: fraction of DE used per step.\n"
            "Max Dist: ray kill distance.  Hit Eps: surface threshold (* 0.001).")
        for label, attr, mn, mx, val, step in [
            ("Max Steps",  'max_steps',  4,    512,   _params.max_steps,  1),
            ("Step Scale", 'step_scale', 0.05, 1.5,   _params.step_scale, 0.005),
            ("Max Dist",   'max_dist',   5.0,  500.0, _params.max_dist,   1.0),
            ("Hit Eps",    'hit_eps',    0.1,  10.0,  _params.hit_eps,    0.05),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, int(round(v)) if a == 'max_steps' else v))
            setattr(self, f'_sl_{attr}', sr)
            march_l.addWidget(sr)

        overrelax_row = QHBoxLayout()
        self._overrelax_check = QCheckBox("Overrelaxation (sphere tracing)")
        self._overrelax_check.setFont(FONT_SMALL)
        self._overrelax_check.setStyleSheet(_css_check())
        self._overrelax_check.setChecked(_params.rm_overrelax)
        self._overrelax_check.stateChanged.connect(
            lambda s: setattr(_params, 'rm_overrelax', bool(s))
        )
        overrelax_row.addWidget(self._overrelax_check)
        march_l.addLayout(overrelax_row)
        sr_or = SliderRow("Relax Factor", 1.0, 2.0, _params.overrelax_factor, 0.01)
        sr_or.on_change(lambda v: setattr(_params, 'overrelax_factor', v))
        setattr(self, '_sl_overrelax_factor', sr_or)
        march_l.addWidget(sr_or)
        _lbl_hint(march_l, "Overrelaxation can speed up marching but may miss thin features.")
        layout.addWidget(march_grp)

        normals_grp = _section("NORMALS")
        normals_l = QVBoxLayout(normals_grp)
        normals_l.setSpacing(2)
        _lbl_hint(normals_l,
            "Eps: offset used for finite-difference normal estimation.\n"
            "Smaller = sharper but more noise-sensitive.")
        sr_neps = SliderRow("Normal Eps", 0.00005, 0.02, _params.normal_eps, 0.00005)
        sr_neps.on_change(lambda v: setattr(_params, 'normal_eps', v))
        setattr(self, '_sl_normal_eps', sr_neps)
        normals_l.addWidget(sr_neps)

        normals_mode_row = QHBoxLayout()
        normals_lbl = _label("Sampling:", COLORS['fg3'], FONT_SMALL)
        normals_lbl.setFixedWidth(80)
        normals_lbl.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self._normals_grp = QButtonGroup(self)
        for i, name in enumerate(["3-tap (fast)", "6-tap (quality)"]):
            rb = QRadioButton(name)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == (1 if _params.feat_normals_full else 0))
            self._normals_grp.addButton(rb, i)
            normals_mode_row.addWidget(rb)
        self._normals_grp.idClicked.connect(
            lambda idx: setattr(_params, 'feat_normals_full', idx == 1)
        )
        normals_mode_row.insertWidget(0, normals_lbl)
        normals_l.addLayout(normals_mode_row)
        layout.addWidget(normals_grp)

        shadow_grp = _section("SHADOW RAY")
        shadow_l = QVBoxLayout(shadow_grp)
        shadow_l.setSpacing(2)
        _lbl_hint(shadow_l,
            "Steps: march budget for soft shadow ray.\n"
            "Min T / Max T: search interval along the light direction.")
        for label, attr, mn, mx, val, step in [
            ("Shadow Steps", 'shadow_steps', 4,    64,    _params.shadow_steps, 1),
            ("Soft Factor",  'shadow_soft',  0.5,  64.0,  _params.shadow_soft,  0.5),
            ("Min T",        'shadow_mint',  0.001, 0.5,  _params.shadow_mint,  0.001),
            ("Max T",        'shadow_maxt',  1.0,  50.0,  _params.shadow_maxt,  0.5),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, int(round(v)) if a == 'shadow_steps' else v))
            setattr(self, f'_sl_{attr}', sr)
            shadow_l.addWidget(sr)
        layout.addWidget(shadow_grp)
        self._register_feat_section('feat_shadows', shadow_grp)

        ao_march_grp = _section("AO MARCH")
        ao_march_l = QVBoxLayout(ao_march_grp)
        ao_march_l.setSpacing(2)
        _lbl_hint(ao_march_l,
            "Step Scale: multiplier on AO probe spacing.\n"
            "Higher = wider AO coverage, lower = tighter surface detail.")
        sr_aoss = SliderRow("AO Step Scale", 0.1, 5.0, _params.ao_step_scale, 0.05)
        sr_aoss.on_change(lambda v: setattr(_params, 'ao_step_scale', v))
        setattr(self, '_sl_ao_step_scale', sr_aoss)
        ao_march_l.addWidget(sr_aoss)
        layout.addWidget(ao_march_grp)
        self._register_feat_section('feat_ao', ao_march_grp)

        refl_grp = _section("REFLECTION")
        refl_l = QVBoxLayout(refl_grp)
        refl_l.setSpacing(2)
        _lbl_hint(refl_l,
            "Strength: max reflection blend.  Fresnel Power: configured in Light & Specular.")
        sr_refl = SliderRow("Strength", 0.0, 1.0, _params.reflection, 0.005)
        sr_refl.on_change(lambda v: setattr(_params, 'reflection', v))
        setattr(self, '_sl_reflection', sr_refl)
        refl_l.addWidget(sr_refl)
        layout.addWidget(refl_grp)
        self._register_feat_section('feat_reflection', refl_grp)

        color_grp = _section("COLOR ANIMATION")
        color_l = QVBoxLayout(color_grp)
        color_l.setSpacing(2)
        _lbl_hint(color_l, "Palette rotation speed and static hue offset.")
        for label, attr, mn, mx, val, step in [
            ("Anim Speed", 'color_anim_speed', 0.0, 0.5, _params.color_anim_speed, 0.002),
            ("Clr Offset", 'color_offset',     0.0, 1.0, _params.color_offset,     0.005),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            color_l.addWidget(sr)
        layout.addWidget(color_grp)

        self._add_section(grp)

    def _build_second_light_section(self):
        grp = _section("SECOND LIGHT")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        _lbl_hint(layout, "Fill light — set Strength > 0 to activate")
        for label, attr, mn, mx, val, step in [
            ("L2 Dir X",   'light2_x',        -5.0, 5.0, _params.light2_x,        0.01),
            ("L2 Dir Y",   'light2_y',        -5.0, 5.0, _params.light2_y,        0.01),
            ("L2 Dir Z",   'light2_z',        -5.0, 5.0, _params.light2_z,        0.01),
            ("L2 Red",     'light2_r',         0.0, 1.0,  _params.light2_r,        0.005),
            ("L2 Green",   'light2_g',         0.0, 1.0,  _params.light2_g,        0.005),
            ("L2 Blue",    'light2_b',         0.0, 1.0,  _params.light2_b,        0.005),
            ("L2 Strength",'light2_strength',  0.0, 3.0,  _params.light2_strength, 0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._add_section(grp)
        self._register_feat_section('feat_second_light', grp)

    def _build_postprocess_section(self):
        grp = _section("POST-PROCESS")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        _lbl_hint(layout, "Gamma, exposure, saturation")
        for label, attr, mn, mx, val, step in [
            ("Gamma",        'gamma',      0.5, 4.0, _params.gamma,      0.01),
            ("Exposure",     'exposure',   0.1, 5.0, _params.exposure,   0.01),
            ("Saturation",   'saturation', 0.0, 3.0, _params.saturation, 0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._add_section(grp)

    def _build_performance_section(self):
        grp = _section("PERFORMANCE")
        layout = QVBoxLayout(grp)
        layout.setSpacing(4)
        _lbl_hint(layout, "Disable expensive features to increase FPS")

        FEATURES = [
            ('feat_ao',           'Ambient Occlusion',  '~6x sceneDist per pixel'),
            ('feat_shadows',      'Soft Shadows',       '~32x sceneDist per pixel'),
            ('feat_normals_full', 'Full Normals (6-tap)','3-tap when off (cheaper)'),
            ('feat_orbit_trap',   'Orbit Trap Color',   'uses step count when off'),
            ('feat_second_light', 'Second Light',       'fill light calculation'),
            ('feat_fog',          'Fog',                'distance-based fog'),
            ('feat_glow',         'Glow Halo',          'miss-ray glow effect'),
            ('feat_reflection',   'Env Reflection',     'background env reflection'),
            ('feat_subsurface',   'Subsurface Scatter', 'translucency effect'),
        ]

        self._feat_checks = {}
        for attr, label, hint in FEATURES:
            row = QHBoxLayout()
            cb = QCheckBox(label)
            cb.setFont(FONT_MONO)
            cb.setStyleSheet(_css_check())
            cb.setChecked(getattr(_params, attr))
            cb.stateChanged.connect(lambda s, a=attr: self._on_feat_changed(a, bool(s)))
            self._feat_checks[attr] = cb
            hint_lbl = _label(hint, COLORS['fg4'], FONT_SMALL)
            hint_lbl.setStyleSheet(
                f"color: {COLORS['fg4']}; background: transparent;"
                "font-style: italic; padding-left: 4px;"
            )
            row.addWidget(cb)
            row.addWidget(hint_lbl, 1)
            layout.addLayout(row)

        sep = QLabel()
        sep.setFixedHeight(6)
        layout.addWidget(sep)

        preset_lbl = _label("Quality presets:", COLORS['fg3'], FONT_SMALL)
        layout.addWidget(preset_lbl)

        PERF_PRESETS = {
            "Ultra":  dict(feat_ao=True,  feat_shadows=True,  feat_normals_full=True,
                           feat_orbit_trap=True,  feat_second_light=True, feat_fog=True,
                           feat_glow=True,  feat_reflection=True,  feat_subsurface=True,
                           aa_samples=2),
            "High":   dict(feat_ao=True,  feat_shadows=True,  feat_normals_full=True,
                           feat_orbit_trap=True,  feat_second_light=True, feat_fog=True,
                           feat_glow=True,  feat_reflection=False, feat_subsurface=False,
                           aa_samples=1),
            "Medium": dict(feat_ao=True,  feat_shadows=False, feat_normals_full=True,
                           feat_orbit_trap=True,  feat_second_light=False, feat_fog=True,
                           feat_glow=True,  feat_reflection=False, feat_subsurface=False,
                           aa_samples=1),
            "Low":    dict(feat_ao=False, feat_shadows=False, feat_normals_full=False,
                           feat_orbit_trap=False, feat_second_light=False, feat_fog=False,
                           feat_glow=True,  feat_reflection=False, feat_subsurface=False,
                           aa_samples=1),
            "Potato": dict(feat_ao=False, feat_shadows=False, feat_normals_full=False,
                           feat_orbit_trap=False, feat_second_light=False, feat_fog=False,
                           feat_glow=False, feat_reflection=False, feat_subsurface=False,
                           aa_samples=1),
        }

        btn_row = QHBoxLayout()
        for name, vals in PERF_PRESETS.items():
            btn = QPushButton(name)
            btn.setFont(FONT_SMALL)
            col_map = {
                'Ultra': '#4a4a8a', 'High': '#3a6a3a', 'Medium': '#5a5a2a',
                'Low': '#6a3a1a', 'Potato': '#5a1a1a'
            }
            btn.setStyleSheet(
                f"QPushButton {{ background: {col_map[name]}; color: {COLORS['fg']};"
                "font: 8pt Consolas; border: none; border-radius: 4px; padding: 4px 6px; }}"
                f"QPushButton:hover {{ background: {COLORS['accent']}; }}"
            )
            btn.clicked.connect(lambda _, v=vals: self._apply_perf_preset(v))
            btn_row.addWidget(btn)
        layout.addLayout(btn_row)

        self._add_section(grp)

    def _on_feat_changed(self, attr: str, enabled: bool):
        setattr(_params, attr, enabled)
        sections = getattr(self, '_feat_sections', {})
        for widget in sections.get(attr, []):
            widget.setVisible(enabled)

    def _apply_perf_preset(self, vals: dict):
        for k, v in vals.items():
            setattr(_params, k, v)
        for attr, cb in self._feat_checks.items():
            cb.blockSignals(True)
            cb.setChecked(getattr(_params, attr))
            cb.blockSignals(False)
        self._aa_grp.button(_params.aa_samples).setChecked(True)
        sections = getattr(self, '_feat_sections', {})
        for attr, widgets in sections.items():
            enabled = getattr(_params, attr, True)
            for widget in widgets:
                widget.setVisible(enabled)

    def _build_saves_section(self):
        self._zkf_buttons = {}
        grp_fractal = _section("FRACTAL PRESETS (.zkf)")
        lf = QVBoxLayout(grp_fractal)
        lf.setSpacing(4)
        hint = _label(
            "Save current fractal parameters as a named preset.\n"
            "Files are stored in saves/ folder next to main.py.",
            COLORS['fg3'], FONT_SMALL
        )
        hint.setWordWrap(True)
        lf.addWidget(hint)
        row_btns = QHBoxLayout()
        btn_save_zkf = QPushButton("Save Fractal (.zkf)")
        btn_save_zkf.setFont(FONT_SMALL)
        btn_save_zkf.setStyleSheet(_css_button())
        btn_save_zkf.clicked.connect(self._save_zkf_dialog)
        btn_load_zkf = QPushButton("Load Fractal (.zkf)")
        btn_load_zkf.setFont(FONT_SMALL)
        btn_load_zkf.setStyleSheet(_css_button())
        btn_load_zkf.clicked.connect(self._load_zkf_dialog)
        row_btns.addWidget(btn_save_zkf)
        row_btns.addWidget(btn_load_zkf)
        lf.addLayout(row_btns)
        self._zkf_list_layout = QVBoxLayout()
        self._zkf_list_layout.setSpacing(2)
        lf.addLayout(self._zkf_list_layout)
        self._add_section(grp_fractal)
        self._refresh_zkf_list()
        grp_session = _section("SESSION SAVE (.zks)")
        ls = QVBoxLayout(grp_session)
        ls.setSpacing(4)
        hint2 = _label(
            "Save/load complete session: all parameters + camera\n"
            "position, yaw, pitch and player state.",
            COLORS['fg3'], FONT_SMALL
        )
        hint2.setWordWrap(True)
        ls.addWidget(hint2)
        row_sess = QHBoxLayout()
        btn_save_zks = QPushButton("Save Session (.zks)")
        btn_save_zks.setFont(FONT_SMALL)
        btn_save_zks.setStyleSheet(_css_button())
        btn_save_zks.clicked.connect(self._save_zks_dialog)
        btn_load_zks = QPushButton("Load Session (.zks)")
        btn_load_zks.setFont(FONT_SMALL)
        btn_load_zks.setStyleSheet(_css_button())
        btn_load_zks.clicked.connect(self._load_zks_dialog)
        row_sess.addWidget(btn_save_zks)
        row_sess.addWidget(btn_load_zks)
        ls.addLayout(row_sess)
        self._zks_list_layout = QVBoxLayout()
        self._zks_list_layout.setSpacing(2)
        ls.addLayout(self._zks_list_layout)
        self._add_section(grp_session)
        self._refresh_zks_list()

    def _refresh_zkf_list(self):
        while self._zkf_list_layout.count():
            item = self._zkf_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        _SAVES_DIR.mkdir(exist_ok=True)
        files = sorted(_SAVES_DIR.glob("*.zkf"))
        if not files:
            lbl = _label("No saved fractals yet.", COLORS['fg4'], FONT_SMALL)
            self._zkf_list_layout.addWidget(lbl)
            return
        for fp in files:
            row = QHBoxLayout()
            try:
                data = load_zkf(fp)
                name = data.get('name', fp.stem)
            except Exception:
                name = fp.stem
            btn = QPushButton(name)
            btn.setFont(FONT_SMALL)
            btn.setStyleSheet(_css_button())
            btn.clicked.connect(lambda _, p=fp, n=name: self._apply_zkf(p, n))
            del_btn = QPushButton("x")
            del_btn.setFont(FONT_SMALL)
            del_btn.setFixedWidth(24)
            del_btn.setStyleSheet(
                f"QPushButton {{ background: {COLORS['bg2']}; color: {COLORS['fg4']};"
                "border: none; border-radius: 3px; padding: 2px; }}"
                f"QPushButton:hover {{ color: #ff6060; }}"
            )
            del_btn.clicked.connect(lambda _, p=fp: self._delete_zkf(p))
            row.addWidget(btn, 1)
            row.addWidget(del_btn)
            w = QWidget()
            w.setLayout(row)
            w.setStyleSheet("background: transparent;")
            self._zkf_list_layout.addWidget(w)

    def _refresh_zks_list(self):
        while self._zks_list_layout.count():
            item = self._zks_list_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        _SAVES_DIR.mkdir(exist_ok=True)
        files = sorted(_SAVES_DIR.glob("*.zks"))
        if not files:
            lbl = _label("No saved sessions yet.", COLORS['fg4'], FONT_SMALL)
            self._zks_list_layout.addWidget(lbl)
            return
        for fp in files:
            row = QHBoxLayout()
            btn = QPushButton(fp.stem)
            btn.setFont(FONT_SMALL)
            btn.setStyleSheet(_css_button())
            btn.clicked.connect(lambda _, p=fp: self._apply_zks(p))
            del_btn = QPushButton("x")
            del_btn.setFont(FONT_SMALL)
            del_btn.setFixedWidth(24)
            del_btn.setStyleSheet(
                f"QPushButton {{ background: {COLORS['bg2']}; color: {COLORS['fg4']};"
                "border: none; border-radius: 3px; padding: 2px; }}"
                f"QPushButton:hover {{ color: #ff6060; }}"
            )
            del_btn.clicked.connect(lambda _, p=fp: self._delete_zks(p))
            row.addWidget(btn, 1)
            row.addWidget(del_btn)
            w = QWidget()
            w.setLayout(row)
            w.setStyleSheet("background: transparent;")
            self._zks_list_layout.addWidget(w)

    def _save_zkf_dialog(self):
        name, ok = QInputDialog.getText(self, "Save Fractal", "Preset name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        safe = "".join(c for c in name if c.isalnum() or c in ' _-').strip()
        if not safe:
            safe = "fractal"
        path = _SAVES_DIR / f"{safe}.zkf"
        try:
            save_zkf(path, name)
            self._refresh_zkf_list()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_zkf_dialog(self):
        _SAVES_DIR.mkdir(exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Fractal", str(_SAVES_DIR), "Fractal Files (*.zkf)"
        )
        if not path:
            return
        self._apply_zkf(Path(path), Path(path).stem)

    def _apply_zkf(self, path, name):
        try:
            data = load_zkf(path)
            vals = data.get('params', {})
            snap_keys = {'fractal_type', 'iterations', 'color_mode', 'shadows',
                         'animate', 'bg_mode', 'fold_x', 'fold_y', 'fold_z',
                         'mb_fold_mode', 'mb_julia_mode', 'mb2_julia_mode', 'mb2_fold_type',
                         'warp_type', 'warp_enabled', 'twist_axis',
                         'fold_mirror_x', 'fold_mirror_y', 'fold_mirror_z',
                         'rep_enabled', 'orbit_trap_type', 'rm_overrelax',
                         'feat_ao', 'feat_shadows', 'feat_normals_full', 'feat_second_light',
                         'feat_fog', 'feat_glow', 'feat_reflection', 'feat_subsurface',
                         'feat_orbit_trap', 'aa_samples', 'max_steps', 'shadow_steps', 'ao_samples'}
            for k, v in vals.items():
                if k in snap_keys:
                    cur = getattr(_params, k, None)
                    if isinstance(cur, tuple):
                        v = tuple(v) if isinstance(v, list) else v
                    setattr(_params, k, v)
            if 'fractal_type' in vals:
                self._type_grp.button(_params.fractal_type).setChecked(True)
                self._on_fractal_type_changed(_params.fractal_type)
            if 'iterations' in vals:
                self._iter_slider.setValue(_params.iterations)
            if 'color_mode' in vals:
                self._cmode_grp.button(_params.color_mode).setChecked(True)
            _interpolator.set_gui_sync(self._sync_all_sliders)
            _interpolator.start(vals)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _delete_zkf(self, path):
        try:
            path.unlink()
            self._refresh_zkf_list()
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", str(e))

    def _save_zks_dialog(self):
        name, ok = QInputDialog.getText(self, "Save Session", "Session name:")
        if not ok or not name.strip():
            return
        name = name.strip()
        safe = "".join(c for c in name if c.isalnum() or c in ' _-').strip() or "session"
        path = _SAVES_DIR / f"{safe}.zks"
        try:
            save_zks(path)
            self._refresh_zks_list()
        except Exception as e:
            QMessageBox.critical(self, "Save Error", str(e))

    def _load_zks_dialog(self):
        _SAVES_DIR.mkdir(exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", str(_SAVES_DIR), "Session Files (*.zks)"
        )
        if not path:
            return
        self._apply_zks(Path(path))

    def _apply_zks(self, path):
        try:
            load_zks(path)
            self._sync_all_sliders()
            ft = _params.fractal_type
            self._type_grp.button(ft).setChecked(True)
            self._on_fractal_type_changed(ft)
            self._iter_slider.setValue(_params.iterations)
            self._cmode_grp.button(_params.color_mode).setChecked(True)
        except Exception as e:
            QMessageBox.critical(self, "Load Error", str(e))

    def _delete_zks(self, path):
        try:
            path.unlink()
            self._refresh_zks_list()
        except Exception as e:
            QMessageBox.critical(self, "Delete Error", str(e))

    def _build_presets_section(self):
        grp = _section("PRESETS")
        layout = QGridLayout(grp)
        layout.setSpacing(4)
        presets = {
            "MB Classic": dict(
                fractal_type=0, scale=-2.0, iterations=8,
                julia_x=-0.5, julia_y=-0.5, julia_z=-0.5,
                color_mode=0, glow=1.0,
                mb_fold_limit=1.0, mb_sphere_inner=0.25, mb_sphere_outer=1.0,
                color1=(0.5,0.5,0.5), color2=(0.5,0.5,0.5), color3=(1.0,1.0,1.0)),
            "MB Fluid": dict(
                fractal_type=0, scale=2.8, iterations=10,
                julia_x=0.0, julia_y=0.0, julia_z=0.0,
                color_mode=1, glow=2.0,
                mb_fold_limit=0.6, mb_sphere_inner=0.1, mb_sphere_outer=0.9, mb_color_scale=1.2,
                color1=(0.1,0.4,0.8), color2=(0.5,0.8,0.5), color3=(0.8,0.2,0.6)),
            "MB Lava": dict(
                fractal_type=0, scale=-2.2, iterations=8,
                julia_x=0.5, julia_y=0.0, julia_z=-0.5,
                color_mode=0, glow=1.5,
                mb_fold_limit=1.0, mb_sphere_inner=0.3, mb_sphere_outer=0.8,
                color1=(1.0,0.2,0.0), color2=(1.0,0.6,0.0), color3=(0.3,0.0,0.0),
                glow_intensity=4.0, emission=0.6, rim_strength=0.6,
                bg_mode=1, bg_color1=(0.08,0.02,0.0), bg_color2=(0.0,0.0,0.0)),
            "Menger Deep": dict(
                fractal_type=1, scale=3.0, iterations=4,
                color_mode=1, ao_strength=1.5, glow=0.5,
                ms_scale=3.0, ms_offset=2.0, ms_cross_width=1.0,
                color1=(0.3,0.6,0.9), color2=(0.5,0.5,0.5), color3=(1.0,1.0,1.0)),
            "Menger Twisted": dict(
                fractal_type=1, scale=3.0, iterations=5,
                color_mode=2, glow=1.5,
                ms_scale=3.2, ms_offset=2.0, ms_twist=0.08,
                color1=(0.8,0.5,0.1), color2=(0.5,0.3,0.0), color3=(1.0,0.8,0.3)),
            "Sierp Fire": dict(
                fractal_type=2, scale=2.0, iterations=8,
                color_mode=0, glow=2.0,
                color1=(1.0,0.3,0.0), color2=(1.0,0.8,0.0), color3=(0.5,0.0,0.0),
                si_vertex_spread=1.0, si_fold_bias=2.0, si_squash=1.0),
            "Sierp Crystal": dict(
                fractal_type=2, scale=2.0, iterations=10,
                color_mode=2, glow=3.0, shadows=True,
                color1=(0.2,0.5,1.0), color2=(0.8,0.9,1.0), color3=(0.3,0.3,0.8),
                si_vertex_spread=1.3, si_fold_bias=2.0, si_squash=0.7, si_twist=0.05),
            "Sierp Abyss": dict(
                fractal_type=2, scale=2.0, iterations=11,
                color_mode=0, glow=5.0, shadows=False,
                color1=(0.1,0.0,0.3), color2=(0.6,0.0,0.8), color3=(0.0,0.0,0.1),
                si_vertex_spread=1.5, si_fold_bias=2.0, si_squash=0.5, si_twist=0.1,
                glow_intensity=9.0, glow_falloff=10.0, emission=1.0, rim_strength=1.5,
                bg_mode=3, bg_color1=(0.03,0.0,0.06), bg_color2=(0.0,0.0,0.0)),
            "Octa Soft": dict(
                fractal_type=3, scale=2.0, iterations=7,
                offset_x=1.8, offset_y=1.8, offset_z=1.8,
                color_mode=1, shadows=False, glow=3.0,
                oc_ifs_scale=2.0, oc_sharpness=2.5, oc_twist=0.06,
                color1=(0.6,0.8,1.0), color2=(0.2,0.4,0.6), color3=(1.0,1.0,1.0)),
            "Octa Plasma": dict(
                fractal_type=3, scale=2.0, iterations=8,
                offset_x=1.6, offset_y=1.6, offset_z=1.6,
                color_mode=0, shadows=False, glow=5.0,
                oc_ifs_scale=2.0, oc_sharpness=1.5, oc_twist=0.1,
                color1=(1.0,0.0,0.5), color2=(0.0,0.5,1.0), color3=(1.0,0.5,0.0),
                glow_intensity=8.0, emission=0.9, rim_strength=1.0,
                bg_mode=2, bg_color1=(0.1,0.0,0.1), bg_color2=(0.0,0.0,0.0)),
        }
        for i, (name, vals) in enumerate(presets.items()):
            btn = QPushButton(name)
            btn.setFont(FONT_SMALL)
            btn.setStyleSheet(_css_button())
            btn.clicked.connect(lambda _, v=vals: self._apply_preset(v))
            layout.addWidget(btn, i // 3, i % 3)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)
        self._add_section(grp)
    def _apply_preset(self, vals: dict):
        snap_keys = {'fractal_type', 'iterations', 'color_mode',
                     'shadows', 'animate', 'bg_mode',
                     'fold_x', 'fold_y', 'fold_z'}
        for k, v in vals.items():
            if k in snap_keys:
                setattr(_params, k, v)
        self._type_grp.button(_params.fractal_type).setChecked(True)
        self._on_fractal_type_changed(_params.fractal_type)
        self._iter_slider.setValue(_params.iterations)
        self._cmode_grp.button(_params.color_mode).setChecked(True)
        _interpolator.set_gui_sync(self._sync_all_sliders)
        _interpolator.start(vals)

    def _sync_all_sliders(self):
        self._sl_scale.set_value(_params.scale)
        self._sl_glow.set_value(_params.glow)
        self._sl_ao_strength.set_value(_params.ao_strength)
        for btn, attr in [(self._c1_btn,'color1'),(self._c2_btn,'color2'),(self._c3_btn,'color3')]:
            hex_col = self._rgb_to_hex(getattr(_params, attr))
            btn.setStyleSheet(
                f"QPushButton {{ background: {hex_col}; color: white; border: none;"
                "border-radius: 4px; padding: 4px 8px; font: bold 8pt Consolas; }}"
                f"QPushButton:hover {{ border: 2px solid {COLORS['accent']}; }}"
            )
        for attr in [
            'scale', 'bailout', 'min_dist', 'de_multiplier', 'fov',
            'mb_fold_limit','mb_sphere_inner','mb_sphere_outer','mb_fixed_radius',
            'mb_color_scale','mb_rot_per_iter',
            'julia_x','julia_y','julia_z',
            'ms_scale','ms_offset','ms_cross_width','ms_twist','ms_sharpness',
            'si_vertex_spread','si_fold_bias','si_squash','si_twist','si_vertex_jitter',
            'oc_ifs_scale','oc_sharpness','oc_twist','oc_offset_uni','oc_fold_amount',
            'offset_x','offset_y','offset_z',
            'light_x','light_y','light_z',
            'specular_power','specular_strength','ambient','subsurface','fresnel_power',
            'light2_x','light2_y','light2_z','light2_r','light2_g','light2_b','light2_strength',
            'color_anim_speed','color_offset',
            'step_scale','normal_eps','reflection',
            'max_dist','hit_eps','shadow_mint','shadow_maxt','ao_step_scale','overrelax_factor',
            'shadow_soft','shadow_steps',
            'ao_strength','ao_radius','fog_density','glow',
            'gamma','exposure','saturation',
        ]:
            sl = getattr(self, f'_sl_{attr}', None)
            if sl is not None:
                sl.set_value(getattr(_params, attr))

    def _build_infinite_section(self):
        grp_ctrl = _section("INFINITE EVOLUTION")
        layout_ctrl = QVBoxLayout(grp_ctrl)
        layout_ctrl.setSpacing(6)

        desc = _label(
            "Continuously morphs all fractal parameters using\n"
            "layered sinusoidal waves — no preset switching.\n"
            "Each run produces a unique, never-repeating form.",
            COLORS['fg2'], FONT_SMALL
        )
        desc.setStyleSheet(
            f"color: {COLORS['fg2']}; background: {COLORS['bg2']};"
            "padding: 6px 8px; border-radius: 4px;"
        )
        layout_ctrl.addWidget(desc)

        btn_row = QHBoxLayout()
        self._evo_toggle_btn = QPushButton("START EVOLUTION")
        self._evo_toggle_btn.setFont(FONT_BOLD)
        self._evo_toggle_btn.setStyleSheet(
            _css_button(COLORS['panel'], COLORS['accent'])
        )
        self._evo_toggle_btn.clicked.connect(self._toggle_evolution)
        btn_row.addWidget(self._evo_toggle_btn, 2)

        reseed_btn = QPushButton("RESEED")
        reseed_btn.setFont(FONT_SMALL)
        reseed_btn.setStyleSheet(_css_button())
        reseed_btn.clicked.connect(self._reseed_evolution)
        btn_row.addWidget(reseed_btn, 1)
        layout_ctrl.addLayout(btn_row)

        self._evo_status_lbl = _label("Status: stopped   t = 0.00", COLORS['fg4'], FONT_SMALL)
        self._evo_status_lbl.setStyleSheet(
            f"color: {COLORS['fg4']}; background: transparent; font: 8pt Consolas;"
        )
        layout_ctrl.addWidget(self._evo_status_lbl)
        _infinite_evo._status_cb = self._on_evo_tick
        self._add_section(grp_ctrl)

        grp_tune = _section("EVOLUTION PARAMETERS")
        layout_tune = QVBoxLayout(grp_tune)
        layout_tune.setSpacing(4)

        _lbl_hint(layout_tune,
            "Speed: how fast parameters oscillate.\n"
            "Mutation depth: amplitude of parameter swings.")

        self._sl_evo_speed = SliderRow("Speed", 0.01, 5.0, _infinite_evo._speed, 0.01)
        self._sl_evo_speed.on_change(lambda v: _infinite_evo.set_speed(v))
        layout_tune.addWidget(self._sl_evo_speed)

        self._sl_evo_mut = SliderRow("Mutation Depth", 0.01, 1.0, _infinite_evo._mutation, 0.01)
        self._sl_evo_mut.on_change(lambda v: _infinite_evo.set_mutation(v))
        layout_tune.addWidget(self._sl_evo_mut)

        self._add_section(grp_tune)

        grp_channel = _section("ACTIVE PARAMETER GROUPS")
        layout_channel = QVBoxLayout(grp_channel)
        layout_channel.setSpacing(2)
        _lbl_hint(layout_channel,
            "Select which parameter groups participate in evolution.\n"
            "(Changes take effect after RESEED)")
        for label in ["Shape & Scale", "Julia / Offset", "Rotation", "Glow & Color", "Lighting"]:
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.setFont(FONT_SMALL)
            cb.setStyleSheet(_css_check())
            layout_channel.addWidget(cb)
        self._add_section(grp_channel)

    def _toggle_evolution(self):
        if _infinite_evo.active:
            _infinite_evo.stop()
            self._evo_toggle_btn.setText("START EVOLUTION")
            self._evo_toggle_btn.setStyleSheet(_css_button(COLORS['panel'], COLORS['accent']))
            self._evo_status_lbl.setText(f"Status: stopped   t = {_infinite_evo._t:.2f}")
        else:
            _infinite_evo.start()
            self._evo_toggle_btn.setText("STOP EVOLUTION")
            self._evo_toggle_btn.setStyleSheet(_css_button(COLORS['accent'], COLORS['panel']))

    def _reseed_evolution(self):
        _infinite_evo.reseed()
        self._evo_status_lbl.setText("Reseeded — new oscillation pattern active.")

    def _on_evo_tick(self, t):
        self._evo_status_lbl.setText(f"Status: running   t = {t:.2f}")

    def _build_player_section(self):
        grp_mode = _section("PLAYER MODE")
        layout_mode = QVBoxLayout(grp_mode)
        layout_mode.setSpacing(6)

        hint = _label(
            "V  — toggle player / fly mode\n"
            "W A S D  — move   |   Space  — jump\n"
            "LMB drag  — look around",
            COLORS['fg2'], FONT_SMALL
        )
        hint.setStyleSheet(
            f"color: {COLORS['fg2']}; background: {COLORS['bg2']};"
            "padding: 6px 8px; border-radius: 4px;"
        )
        layout_mode.addWidget(hint)

        toggle_row = QHBoxLayout()
        self._player_mode_lbl = _label("PLAYER MODE: OFF", COLORS['fg4'], FONT_SMALL)
        self._player_mode_lbl.setStyleSheet(
            f"color: {COLORS['fg4']}; background: {COLORS['bg2']};"
            "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
        )
        toggle_btn = QPushButton("Toggle (V)")
        toggle_btn.setFont(FONT_SMALL)
        toggle_btn.setStyleSheet(_css_button())
        toggle_btn.clicked.connect(self._toggle_player_mode)
        toggle_row.addWidget(self._player_mode_lbl, 1)
        toggle_row.addWidget(toggle_btn)
        layout_mode.addLayout(toggle_row)

        reset_btn = QPushButton("Reset Velocity")
        reset_btn.setFont(FONT_SMALL)
        reset_btn.setStyleSheet(_css_button())
        reset_btn.clicked.connect(self._reset_player_velocity)
        layout_mode.addWidget(reset_btn)

        dbg_row = QHBoxLayout()
        self._dbg_lbl = _label("COLLIDER DEBUG: OFF", COLORS['fg4'], FONT_SMALL)
        self._dbg_lbl.setStyleSheet(
            f"color: {COLORS['fg4']}; background: {COLORS['bg2']};"
            "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
        )
        dbg_btn = QPushButton("Toggle (F1)")
        dbg_btn.setFont(FONT_SMALL)
        dbg_btn.setStyleSheet(_css_button())
        dbg_btn.clicked.connect(self._toggle_debug_overlay)
        dbg_row.addWidget(self._dbg_lbl, 1)
        dbg_row.addWidget(dbg_btn)
        layout_mode.addLayout(dbg_row)

        _lbl_hint(layout_mode,
            "Blue ring = collision radius  |  Cyan line = surface normal\n"
            "Red arrow = gravity dir  |  Green/Red dots = SDF probes\n"
            "Blue ring (outer) = ground detection threshold"
        )

        self._add_section(grp_mode)

        grp_grav = _section("GRAVITY MODE")
        layout_grav = QVBoxLayout(grp_grav)
        layout_grav.setSpacing(4)

        self._grav_mode_grp = QButtonGroup(self)
        grav_row = QHBoxLayout()
        for i, (label, tip) in enumerate([
            ("To Fractal", "Gravity pulls toward nearest fractal surface (normal-based)"),
            ("To Center",  "Gravity pulls toward world origin (0, 0, 0)"),
            ("Down",       "Standard gravity: always pulls along -Y axis"),
        ]):
            rb = QRadioButton(label)
            rb.setFont(FONT_SMALL)
            rb.setStyleSheet(_css_radio())
            rb.setChecked(i == _player_state.GRAVITY_MODE)
            rb.setToolTip(tip)
            self._grav_mode_grp.addButton(rb, i)
            grav_row.addWidget(rb)
        layout_grav.addLayout(grav_row)
        self._grav_mode_grp.idClicked.connect(
            lambda idx: setattr(_player_state, 'GRAVITY_MODE', idx)
        )
        self._add_section(grp_grav)

        grp_phys = _section("PHYSICS")
        layout_phys = QVBoxLayout(grp_phys)
        layout_phys.setSpacing(2)

        def _make_ps_slider(label, attr, mn, mx, step=0.05):
            sl = SliderRow(label, mn, mx, getattr(_player_state, attr), step)
            sl.on_change(lambda v, a=attr: setattr(_player_state, a, v))
            layout_phys.addWidget(sl)
            return sl

        self._sl_ps_gravity  = _make_ps_slider("Gravity",      'GRAVITY_STRENGTH', 0.1, 15.0, 0.1)
        self._sl_ps_move     = _make_ps_slider("Move Speed",   'MOVE_SPEED',        0.2, 10.0, 0.1)
        self._sl_ps_jump     = _make_ps_slider("Jump Speed",   'JUMP_SPEED',        0.2, 12.0, 0.1)
        self._sl_ps_friction = _make_ps_slider("Friction",     'FRICTION',          0.5, 30.0, 0.5)
        self._sl_ps_air      = _make_ps_slider("Air Control",  'AIR_CONTROL',       0.0,  1.0, 0.01)
        self._sl_ps_speedcap = _make_ps_slider("Speed Cap",    'SPEED_CAP',         0.5, 20.0, 0.2)
        self._sl_ps_height   = _make_ps_slider("Player Height",'PLAYER_HEIGHT',     0.01, 1.0, 0.005)
        self._sl_ps_gnd      = _make_ps_slider("Ground Dist",  'GROUND_DIST',       0.01, 1.0, 0.005)
        self._sl_ps_bias     = _make_ps_slider("Coll. Bias",   'COLLISION_BIAS',    0.1, 10.0, 0.1)

        _lbl_hint(layout_phys,
            "Coll. Bias scales collision threshold to match fractal density.\n"
            "Increase if falling through surface, decrease if hovering above it.")

        self._add_section(grp_phys)

        grp_pre = _section("PHYSICS PRESETS")
        layout_pre = QVBoxLayout(grp_pre)
        layout_pre.setSpacing(4)

        PHYS_PRESETS = {
            "Default": dict(GRAVITY_STRENGTH=1.0,  MOVE_SPEED=0.5, JUMP_SPEED=0.6,
                            FRICTION=10.0, AIR_CONTROL=0.13, SPEED_CAP=4.2,  COLLISION_BIAS=1.0),
            "Moon":    dict(GRAVITY_STRENGTH=0.5,  MOVE_SPEED=0.7, JUMP_SPEED=0.5,
                            FRICTION=5.0,  AIR_CONTROL=0.5,  SPEED_CAP=8.0,  COLLISION_BIAS=1.0),
            "Heavy":   dict(GRAVITY_STRENGTH=4.0,  MOVE_SPEED=2.0, JUMP_SPEED=0.6,
                            FRICTION=10.0, AIR_CONTROL=0.1,  SPEED_CAP=5.0,  COLLISION_BIAS=1.0),
            "Floaty":  dict(GRAVITY_STRENGTH=0.5,  MOVE_SPEED=1.0, JUMP_SPEED=0.7,
                            FRICTION=10.0, AIR_CONTROL=0.9, SPEED_CAP=4.2,  COLLISION_BIAS=1.0),
            "Ice":     dict(GRAVITY_STRENGTH=1.0,  MOVE_SPEED=0.5, JUMP_SPEED=0.6,
                            FRICTION=3.0, AIR_CONTROL=0.13, SPEED_CAP=4.2,  COLLISION_BIAS=1.0),
        }

        btn_row = QHBoxLayout()
        for name, vals in PHYS_PRESETS.items():
            btn = QPushButton(name)
            btn.setFont(FONT_SMALL)
            btn.setStyleSheet(_css_button())
            btn.clicked.connect(lambda _, v=vals: self._apply_physics_preset(v))
            btn_row.addWidget(btn)
        layout_pre.addLayout(btn_row)
        self._add_section(grp_pre)

        grp_status = _section("STATUS")
        layout_status = QVBoxLayout(grp_status)
        layout_status.setSpacing(4)

        self._pl_pos_lbl    = _label("Position:   —", COLORS['fg3'], FONT_SMALL)
        self._pl_vel_lbl    = _label("Velocity:   —", COLORS['fg3'], FONT_SMALL)
        self._pl_ground_lbl = _label("On ground:  —", COLORS['fg3'], FONT_SMALL)
        self._pl_sdf_lbl    = _label("SDF raw:    —", COLORS['fg3'], FONT_SMALL)
        self._pl_thresh_lbl = _label("Col thresh: —", COLORS['fg3'], FONT_SMALL)
        self._pl_grav_lbl   = _label("Grav dir:   —", COLORS['fg3'], FONT_SMALL)
        for lbl in (self._pl_pos_lbl, self._pl_vel_lbl, self._pl_ground_lbl,
                    self._pl_sdf_lbl, self._pl_thresh_lbl, self._pl_grav_lbl):
            lbl.setStyleSheet(
                f"color: {COLORS['fg3']}; background: transparent;"
                "font: 8pt Consolas;"
            )
            layout_status.addWidget(lbl)

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_player_status)
        self._status_timer.start(100)

        self._add_section(grp_status)

    def _toggle_player_mode(self):
        _params.player_mode = not _params.player_mode
        if _params.player_mode:
            _player_state.vel = [0.0, 0.0, 0.0]
            _player_state.jump_queued = False
            _player_state._smooth_pos    = list(_params.cam_pos)
            _player_state._smooth_vel    = [0.0, 0.0, 0.0]
            _player_state._smooth_normal = list(_player_state._surface_normal)
            _player_state._bob_phase     = 0.0
            self._player_mode_lbl.setText("PLAYER MODE: ON")
            self._player_mode_lbl.setStyleSheet(
                f"color: {COLORS['accent']}; background: {COLORS['bg2']};"
                "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
            )
        else:
            _params.player_mode = False
            self._player_mode_lbl.setText("PLAYER MODE: OFF")
            self._player_mode_lbl.setStyleSheet(
                f"color: {COLORS['fg4']}; background: {COLORS['bg2']};"
                "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
            )

    def _toggle_debug_overlay(self):
        global _debug_overlay_enabled
        _debug_overlay_enabled = not _debug_overlay_enabled
        if _debug_overlay_enabled:
            self._dbg_lbl.setText("COLLIDER DEBUG: ON")
            self._dbg_lbl.setStyleSheet(
                f"color: {COLORS['accent']}; background: {COLORS['bg2']};"
                "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
            )
        else:
            self._dbg_lbl.setText("COLLIDER DEBUG: OFF")
            self._dbg_lbl.setStyleSheet(
                f"color: {COLORS['fg4']}; background: {COLORS['bg2']};"
                "padding: 4px 8px; border-radius: 4px; font: bold 9pt Consolas;"
            )

    def _reset_player_velocity(self):
        _player_state.vel = [0.0, 0.0, 0.0]
        _player_state.jump_queued = False

    def _apply_physics_preset(self, vals: dict):
        for k, v in vals.items():
            setattr(_player_state, k, v)
        for attr, sl in [
            ('GRAVITY_STRENGTH', self._sl_ps_gravity),
            ('MOVE_SPEED',       self._sl_ps_move),
            ('JUMP_SPEED',       self._sl_ps_jump),
            ('FRICTION',         self._sl_ps_friction),
            ('AIR_CONTROL',      self._sl_ps_air),
            ('SPEED_CAP',        self._sl_ps_speedcap),
            ('COLLISION_BIAS',   self._sl_ps_bias),
        ]:
            sl.set_value(getattr(_player_state, attr))

    def _update_player_status(self):
        if not _params.player_mode:
            return
        pos = _params.cam_pos
        vel = _player_state.vel
        spd = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        try:
            d = _py_sdf(pos)
            radius = _player_state._effective_radius()
            gnd = _player_state._ground_threshold()
            sdf_str    = f"{d:.5f}"
            thresh_str = f"r={radius:.5f}  gnd={gnd:.5f}"
        except Exception:
            sdf_str = thresh_str = "err"
        self._pl_pos_lbl.setText(
            f"Position:   {pos[0]:+.3f}  {pos[1]:+.3f}  {pos[2]:+.3f}"
        )
        self._pl_vel_lbl.setText(
            f"Velocity:   {spd:.3f}  ({vel[0]:+.2f} {vel[1]:+.2f} {vel[2]:+.2f})"
        )
        self._pl_ground_lbl.setText(
            f"On ground:  {'YES' if _player_state.on_ground else 'NO'}"
        )
        self._pl_sdf_lbl.setText(f"SDF raw:    {sdf_str}")
        self._pl_thresh_lbl.setText(f"Col thresh: {thresh_str}")
        grav_names = {0: "To Fractal", 1: "To Center", 2: "Down"}
        gd = _player_state._gravity_dir
        self._pl_grav_lbl.setText(
            f"Grav dir:   {grav_names.get(_player_state.GRAVITY_MODE, '?')}"
            f"  ({gd[0]:+.2f} {gd[1]:+.2f} {gd[2]:+.2f})"
        )

def run_gl():
    mglw_settings.WINDOW = {
        'class':        'moderngl_window.context.pyglet.Window',
        'gl_version':   (3, 3),
        'title':        'Kaleidoscopic IFS Fractal' + APP_VERSION,
        'size':         (1280, 720),
        'aspect_ratio': False,
        'resizable':    True,
        'vsync':        False,
    }
    mglw.run_window_config(FractalWindow)

if __name__ == '__main__':
    gl_thread = threading.Thread(target=run_gl, daemon=True)
    gl_thread.start()
    time.sleep(0.5)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setWindowIcon(QIcon(str(Path(__file__).parent / "icon.png")))
    _apply_palette(app)
    gui = ControlGUI()
    gui.show()
    sys.exit(app.exec_())