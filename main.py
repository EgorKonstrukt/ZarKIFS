import math
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
    QButtonGroup, QGroupBox, QColorDialog,
)
from moderngl_window.conf import settings as mglw_settings

APP_VERSION = "1.1.0"

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
uniform vec3  u_color1;
uniform vec3  u_color2;
uniform vec3  u_color3;
uniform int   u_color_mode;
uniform float u_ao_strength;
uniform float u_shadow_soft;
uniform int   u_shadows;
uniform float u_glow;
uniform vec3  u_cam_pos;
uniform vec3  u_cam_fwd;
uniform vec3  u_cam_right;
uniform vec3  u_cam_up;
uniform int   u_animate;
uniform float u_anim_speed;

// --- Light ---
uniform vec3  u_light_dir;        // world-space light direction (default 1,2,1.5 normalized)
uniform float u_specular_power;   // Blinn-Phong exponent (default 32.0)
uniform float u_specular_strength;// specular contribution scale (default 0.3)
uniform float u_ambient;          // ambient light level (default 0.2)

// --- Color animation ---
uniform float u_color_anim_speed; // palette time scroll speed (default 0.05)
uniform float u_color_offset;     // static phase offset added to palette t (default 0.0)

// --- Raymarching ---
uniform float u_step_scale;       // fraction of distance to step (default 0.5)
uniform float u_normal_eps;       // finite-diff epsilon for normals (default 0.001)

// --- Glow ---
uniform float u_glow_intensity;   // overall brightness of the halo  (default 5.0)
uniform float u_glow_falloff;     // how quickly halo falls with distance (default 8.0)
uniform float u_glow_radius;      // inner "core" brightness boost (default 1.0)
uniform float u_rim_strength;     // rim-light on surface edges (default 0.4)
uniform float u_emission;         // self-emission mixed into surface colour (default 0.2)

// --- Background ---
uniform vec3  u_bg_color1;        // horizon/zenith colour (default deep blue)
uniform vec3  u_bg_color2;        // nadir colour (default black)
uniform int   u_bg_mode;          // 0=flat, 1=gradient, 2=nebula, 3=starfield

// --- Stars ---
uniform float u_star_density;     // star density 0..1 (default 0.5)
uniform float u_star_brightness;  // star brightness multiplier (default 1.0)
uniform float u_star_twinkle;     // twinkle strength 0..1 (default 0.3)
uniform float u_star_size;        // star size multiplier (default 1.0)
uniform int   u_milkyway;         // milky way on/off (default 1)

// --- Anti-aliasing ---
uniform int   u_aa_samples;       // MSAA sub-pixel count: 1,2,4  (default 1 = off)

// --- Mandelbox fine-tune ---
uniform float u_mb_fold_limit;    // box-fold clamp value (default 1.0)
uniform float u_mb_sphere_inner;  // inner sphere radius^2 (default 0.25)
uniform float u_mb_sphere_outer;  // outer sphere radius^2 (default 1.0)
uniform float u_mb_fixed_radius;  // fixed-radius factor for outer fold (default 1.0)
uniform float u_mb_color_scale;   // orbit trap color scale (default 0.5)
uniform float u_mb_rot_per_iter;  // Y-rotation added each iteration (default 0.0)

// --- Menger Sponge fine-tune ---
uniform float u_ms_cross_width;   // cross-gap width (default 2.0, range 1..3)
uniform float u_ms_scale;         // IFS scale per level (default 3.0, range 2..4)
uniform float u_ms_offset;        // IFS offset (default 2.0, range 1..3)
uniform float u_ms_twist;         // per-iteration rotation angle (default 0.0)
uniform float u_ms_sharpness;     // box corner sharpness 1=sharp 0.5=round (default 1.0)

// --- Sierpinski fine-tune ---
uniform float u_si_vertex_spread; // vertex spread factor (default 1.0)
uniform float u_si_fold_bias;     // fold distance bias (default 2.0 = classic IFS * 2)
uniform float u_si_twist;         // twist angle added per iteration (default 0.0)
uniform float u_si_squash;        // y-axis squash (default 1.0)
uniform float u_si_vertex_jitter; // random vertex perturbation amount (default 0.0)

// --- Octahedron IFS fine-tune ---
uniform float u_oc_ifs_scale;     // IFS scale (default 2.0)
uniform float u_oc_twist;         // rotation per iteration (default 0.0)
uniform float u_oc_sharpness;     // l-norm exponent (2=sphere, 1=octa, default 1.0)
uniform float u_oc_offset_uni;    // uniform offset multiplier (default 1.0)
uniform float u_oc_fold_amount;   // extra abs-fold mixing 0..1 (default 0.0)

#define PI 3.14159265358979323846
#define MAX_STEPS 200
#define MAX_DIST  100.0

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

vec2 mandelbox(vec3 pos) {
    vec3 p = pos;
    float trap = 1e10;
    float dr = 1.0;
    float foldL = u_mb_fold_limit;
    float sphIn = u_mb_sphere_inner;
    float sphOut = u_mb_sphere_outer * u_mb_fixed_radius;
    for (int i = 0; i < u_iterations; i++) {
        if (u_mb_rot_per_iter > 0.0001) p = rotY(u_mb_rot_per_iter * float(i)) * p;
        p = clamp(p, -foldL, foldL) * 2.0 - p;
        float r2 = dot(p, p);
        if (r2 < sphIn) {
            float k = sphOut / sphIn;
            p *= k; dr *= k;
        } else if (r2 < sphOut) {
            float k = sphOut / r2;
            p *= k; dr *= k;
        }
        p = p * u_scale + vec3(u_julia_x, u_julia_y, u_julia_z);
        dr = dr * abs(u_scale) + 1.0;
        trap = min(trap, dot(p,p) * u_mb_color_scale);
        if (dot(p,p) > u_bailout * u_bailout) break;
    }
    return vec2(length(p) / abs(dr), trap);
}

vec2 mengerSponge(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    float ms = u_ms_scale;
    float mo = u_ms_offset;
    for (int i = 0; i < u_iterations; i++) {
        if (u_ms_twist > 0.001) p = rotY(u_ms_twist) * p;
        p = abs(p);
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;
        p = p * ms - vec3(mo);
        p.z += mo * clamp(p.z / mo * 0.5 + 0.5, 0.0, 1.0) * u_ms_cross_width;
        s *= ms;
        trap = min(trap, dot(p, p) / (s * s));
    }
    vec3 q = abs(p) - vec3(1.0);
    float boxDist;
    if (u_ms_sharpness >= 0.99) {
        boxDist = sdBox(p, vec3(1.0));
    } else {
        float r = 1.0 - u_ms_sharpness;
        vec3 qr = max(q + r, 0.0);
        boxDist = length(qr) - r + min(max(q.x, max(q.y, q.z)), 0.0);
    }
    return vec2(boxDist / s, trap);
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
        if (u_si_twist > 0.001) p = rotY(u_si_twist) * p;
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
        trap = min(trap, dot(p, p) / (scale * scale));
    }
    return vec2(sdTetra(p, scale) / scale, trap);
}

vec2 octahedronIFS(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    float IFS_SCALE = u_oc_ifs_scale;
    vec3 off = vec3(u_offset_x, u_offset_y, u_offset_z) * u_oc_offset_uni;
    for (int i = 0; i < u_iterations; i++) {
        if (u_oc_twist > 0.001) p = rotY(u_oc_twist) * p;
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
        trap = min(trap, dot(p, p) / (s * s));
    }
    float sh = max(u_oc_sharpness, 0.5);
    float r;
    if (sh < 1.05) {
        r = abs(p.x) + abs(p.y) + abs(p.z) - 1.0;
    } else {
        r = pow(pow(abs(p.x), sh) + pow(abs(p.y), sh) + pow(abs(p.z), sh), 1.0/sh) - 1.0;
    }
    return vec2(r / s, trap);
}

vec2 sceneDist(vec3 p) {
    mat3 rx = rotX(u_rot_x), ry = rotY(u_rot_y), rz = rotZ(u_rot_z);
    p = rz * ry * rx * p;
    if (u_fractal_type == 0) return mandelbox(p);
    if (u_fractal_type == 1) return mengerSponge(p);
    if (u_fractal_type == 2) return sierpinski(p);
    return octahedronIFS(p);
}

vec3 calcNormal(vec3 p) {
    float e = u_normal_eps;
    return normalize(vec3(
        sceneDist(p+vec3(e,0,0)).x - sceneDist(p-vec3(e,0,0)).x,
        sceneDist(p+vec3(0,e,0)).x - sceneDist(p-vec3(0,e,0)).x,
        sceneDist(p+vec3(0,0,e)).x - sceneDist(p-vec3(0,0,e)).x
    ));
}

float softShadow(vec3 ro, vec3 rd, float mint, float maxt, float k) {
    float res = 1.0;
    float t = mint;
    for (int i = 0; i < 32; i++) {
        float h = sceneDist(ro + rd*t).x;
        if (h < 0.0001) return 0.0;
        res = min(res, k * h / t);
        t += clamp(h, 0.01, 0.2);
        if (t > maxt) break;
    }
    return clamp(res, 0.0, 1.0);
}

float ambientOcclusion(vec3 p, vec3 n) {
    float occ = 0.0, scale = 1.0;
    for (int i = 0; i < 5; i++) {
        float h = 0.01 + 0.12 * float(i) / 4.0;
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
    // map hash [0..1] to blackbody-like star colour: blue->white->yellow->orange
    if (h < 0.15) return vec3(0.6, 0.7, 1.0);   // blue-white (O/B)
    if (h < 0.40) return vec3(0.9, 0.95, 1.0);  // white (A)
    if (h < 0.65) return vec3(1.0, 0.97, 0.88); // yellow-white (F/G)
    if (h < 0.85) return vec3(1.0, 0.85, 0.5);  // yellow-orange (K)
    return vec3(1.0, 0.55, 0.25);               // orange-red (M)
}

float oneStar(vec2 cuv, float scale, float densityThresh, float seed) {
    vec2 uv   = cuv * scale + seed;
    vec2 cell = floor(uv);
    vec2 loc  = fract(uv) - 0.5;
    vec2 ji   = hash2(cell + seed) - 0.5;
    vec2 off  = loc - ji * 0.7;
    float br  = hash(cell + seed + 3.7);
    if (br < densityThresh) return 0.0;
    float r     = length(off);
    float sharp = mix(180.0, 60.0, br);
    float core  = exp(-r * r * sharp);
    float halo  = exp(-r * r * sharp * 0.08) * 0.12;
    return (core + halo) * (0.4 + br * 1.6);
}

vec3 starField(vec2 cuv, float t) {
    vec3 col = vec3(0.0);
    float layers[4];
    float scales[4];
    float thresh[4];
    float seeds[4];
    layers[0] = 0.0; scales[0] = 18.0; thresh[0] = 0.86; seeds[0] = 0.0;
    layers[1] = 0.0; scales[1] = 35.0; thresh[1] = 0.80; seeds[1] = 5.3;
    layers[2] = 0.0; scales[2] = 70.0; thresh[2] = 0.75; seeds[2] = 11.7;
    layers[3] = 0.0; scales[3] = 130.0; thresh[3] = 0.70; seeds[3] = 23.1;
    for (int i = 0; i < 4; i++) {
        vec2 cell = floor(cuv * scales[i] + seeds[i]);
        vec2 loc  = fract(cuv * scales[i] + seeds[i]) - 0.5;
        vec2 ji   = hash2(cell + seeds[i]) - 0.5;
        vec2 off  = loc - ji * 0.7;
        float br  = hash(cell + seeds[i] + 3.7);
        if (br >= thresh[i]) {
            float r     = length(off);
            float sharp = mix(160.0, 50.0, br);
            float core  = exp(-r * r * sharp);
            float halo  = exp(-r * r * sharp * 0.07) * 0.10;
            float lum   = (core + halo) * (0.3 + br * 1.7);
            float tw    = 1.0 + 0.08 * sin(t * (2.0 + br * 3.0) + br * 47.3);
            vec3  tint  = starTemperature(hash(cell + seeds[i] + 99.1));
            col += tint * lum * tw;
        }
    }
    return col;
}

vec3 milkyWay(vec2 cuv) {
    float band  = fbm(cuv * 0.8 + vec2(3.1, 7.4));
    float band2 = fbm(cuv * 1.6 + vec2(11.2, 2.9));
    float mw    = smoothstep(0.35, 0.65, band) * smoothstep(0.30, 0.60, band2);
    float dust  = 1.0 - smoothstep(0.45, 0.55, fbm(cuv * 2.5 + vec2(5.5, 1.1)));
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
    vec3 stars = starField(cuv, t);
    vec3 mw    = milkyWay(cuv);
    return base + stars + mw;
}

// ---- Single ray -------------------------------------------------------
vec3 castRay(vec2 uv, float t) {
    vec3 ro = u_cam_pos;
    vec3 rd = normalize(uv.x * u_cam_right + uv.y * u_cam_up + 1.5 * u_cam_fwd);

    float totalDist = 0.0;
    float minDist   = 1e10;
    float trap      = 0.0;
    int   steps     = 0;
    bool  hit       = false;

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p   = ro + rd * totalDist;
        vec2 res = sceneDist(p);
        float d  = res.x;
        trap     = res.y;
        minDist  = min(minDist, d);
        if (d < u_min_dist * 0.001) { hit = true; steps = i; break; }
        if (totalDist > MAX_DIST) break;
        totalDist += d * u_step_scale;
        steps = i;
    }

    vec3 bg = background(rd, t);
    vec3 col;
    if (hit) {
        vec3 p   = ro + rd * totalDist;
        vec3 n   = calcNormal(p);
        vec3 lightDir = normalize(u_light_dir);
        float diff    = max(dot(n, lightDir), 0.0);
        float spec    = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), u_specular_power);
        float ao      = ambientOcclusion(p, n);
        float shadow  = u_shadows == 1 ? softShadow(p, lightDir, 0.02, 10.0, u_shadow_soft) : 1.0;

        float colorParam;
        if (u_color_mode == 0)      colorParam = float(steps) / float(u_iterations * 2);
        else if (u_color_mode == 1) colorParam = clamp(sqrt(trap) * 0.5, 0.0, 1.0);
        else if (u_color_mode == 2) colorParam = n.x * 0.5 + 0.5;
        else                        colorParam = clamp(totalDist / MAX_DIST, 0.0, 1.0);

        vec3 baseCol  = palette(colorParam + u_color_offset + t * u_color_anim_speed);
        float rim     = pow(1.0 - max(dot(n, -rd), 0.0), 3.0) * u_rim_strength;
        float emitStr = u_emission * (1.0 + colorParam);

        col  = baseCol * (diff * shadow * (1.0 - u_ambient) + u_ambient) * ao;
        col += spec * u_specular_strength * shadow;
        col += baseCol * emitStr;
        col += palette(colorParam) * rim;
        col += baseCol * u_glow_intensity * 0.04;

        float fog = exp(-totalDist * u_fog_density * 0.1);
        col = mix(bg, col, fog);
    } else {
        float falloff   = max(u_glow_falloff, 0.1);
        float proximity = pow(1.0 - exp(-1.0 / (minDist * falloff + 0.001)), 2.0);
        float core      = exp(-minDist * minDist * falloff * falloff * u_glow_radius * 2.0);
        vec3  glowCol   = palette(float(steps) / float(MAX_STEPS) + t * 0.04);
        col  = bg;
        col += glowCol * proximity * u_glow_intensity * 0.25;
        col += glowCol * core      * u_glow_intensity * 0.55;
    }
    return col;
}

void main() {
    float aspect = u_resolution.x / u_resolution.y;
    float t      = u_animate == 1 ? u_time * u_anim_speed : 0.0;
    vec2  pixSize = vec2(1.0 / u_resolution.x, 1.0 / u_resolution.y);

    vec3 col = vec3(0.0);

    if (u_aa_samples <= 1) {
        vec2 uv = vec2(v_uv.x * aspect, v_uv.y);
        col = castRay(uv, t);
    } else if (u_aa_samples == 2) {
        // 2x2 rotated-grid
        const vec2 o[4] = vec2[4](vec2(-0.25,-0.25), vec2( 0.25,-0.25),
                                   vec2(-0.25, 0.25), vec2( 0.25, 0.25));
        for (int i = 0; i < 4; i++) {
            vec2 uv = vec2((v_uv.x + o[i].x * pixSize.x * 2.0) * aspect,
                            v_uv.y + o[i].y * pixSize.y * 2.0);
            col += castRay(uv, t);
        }
        col *= 0.25;
    } else {
        // 3x3 jittered 9-sample
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

    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));
    fragColor = vec4(col, 1.0);
}
"""

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
        self.animate      = True
        self.anim_speed   = 2.5

        self.mb_fold_limit    = 1.0
        self.mb_sphere_inner  = 0.25
        self.mb_sphere_outer  = 1.0
        self.mb_fixed_radius  = 1.0
        self.mb_color_scale   = 0.5
        self.mb_rot_per_iter  = 0.0

        self.ms_cross_width   = 1.0
        self.ms_scale         = 3.0
        self.ms_offset        = 2.0
        self.ms_twist         = 0.0
        self.ms_sharpness     = 1.0

        self.si_vertex_spread = 1.0
        self.si_fold_bias     = 2.0
        self.si_twist         = 0.0
        self.si_squash        = 1.0
        self.si_vertex_jitter = 0.0

        self.oc_ifs_scale     = 2.0
        self.oc_twist         = 0.0
        self.oc_sharpness     = 1.0
        self.oc_offset_uni    = 1.0
        self.oc_fold_amount   = 0.0

        self.light_x          = 1.0
        self.light_y          = 2.0
        self.light_z          = 1.5
        self.specular_power   = 32.0
        self.specular_strength= 0.3
        self.ambient          = 0.2

        self.color_anim_speed = 0.05
        self.color_offset     = 0.0

        self.step_scale       = 0.5
        self.normal_eps       = 0.001

        self.glow_intensity   = 5.0
        self.glow_falloff     = 8.0
        self.glow_radius      = 1.0
        self.rim_strength     = 0.4
        self.emission         = 0.2

        self.bg_color1        = (0.02, 0.03, 0.08)
        self.bg_color2        = (0.0,  0.0,  0.0)
        self.bg_mode          = 1   # 0=flat 1=gradient 2=nebula 3=starfield

        self.aa_samples       = 1   # 1=off 2=4xRGSS 3=9x

        self.screenshot_requested = False

_params = FractalParams()

INTERP_FLOAT_ATTRS = [
    'scale', 'rot_x', 'rot_y', 'rot_z',
    'offset_x', 'offset_y', 'offset_z',
    'julia_x', 'julia_y', 'julia_z',
    'bailout', 'min_dist', 'fog_density',
    'ao_strength', 'shadow_soft', 'glow',
    'anim_speed',
    'glow_intensity', 'glow_falloff', 'glow_radius', 'rim_strength', 'emission',
    'mb_fold_limit', 'mb_sphere_inner', 'mb_sphere_outer', 'mb_fixed_radius', 'mb_color_scale',
    'mb_rot_per_iter',
    'ms_cross_width', 'ms_scale', 'ms_offset', 'ms_twist', 'ms_sharpness',
    'si_vertex_spread', 'si_fold_bias', 'si_twist', 'si_squash', 'si_vertex_jitter',
    'oc_ifs_scale', 'oc_twist', 'oc_sharpness', 'oc_offset_uni', 'oc_fold_amount',
    'light_x', 'light_y', 'light_z',
    'specular_power', 'specular_strength', 'ambient',
    'color_anim_speed', 'color_offset',
    'step_scale', 'normal_eps',
]
INTERP_COLOR_ATTRS = ['color1', 'color2', 'color3', 'bg_color1', 'bg_color2']

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

class _CamInput:
    keys_pressed: set    = set()
    mouse_dragging: bool = False

_cam_input = _CamInput()
_cam_vel   = [0.0, 0.0, 0.0]

class FractalWindow(mglw.WindowConfig):
    title        = "Kaleidoscopic IFS Fractal " + APP_VERSION
    gl_version   = (4, 6)
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
        self.start = time.time()
        self._pending_screenshot = False
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
    def key_event(self, key, action, modifiers):
        from moderngl_window.context.pyglet.keys import Keys
        import pyglet
        km = pyglet.window.key
        PRESS   = self.wnd.keys.ACTION_PRESS
        RELEASE = self.wnd.keys.ACTION_RELEASE
        if key in (km.LSHIFT, km.RSHIFT):
            if action == PRESS:   _cam_input.keys_pressed.add('shift')
            elif action == RELEASE: _cam_input.keys_pressed.discard('shift')
        if key in (km.LALT, km.RALT):
            if action == PRESS:   _cam_input.keys_pressed.add('alt')
            elif action == RELEASE: _cam_input.keys_pressed.discard('alt')
        name_map = {
            Keys.W: 'w', Keys.S: 's',
            Keys.A: 'a', Keys.D: 'd',
            Keys.Q: 'q', Keys.E: 'e',
        }
        k = name_map.get(key)
        if k is None:
            # F12 screenshot
            try:
                import pyglet
                km2 = pyglet.window.key
                if key == km2.F12 and action == PRESS:
                    self._pending_screenshot = True
            except Exception:
                pass
            return
        if action == PRESS:
            _cam_input.keys_pressed.add(k)
        elif action == RELEASE:
            _cam_input.keys_pressed.discard(k)
    def mouse_press_event(self, x, y, button):
        if button == 1:
            _cam_input.mouse_dragging = True
    def mouse_release_event(self, x, y, button):
        if button == 1:
            _cam_input.mouse_dragging = False
    def mouse_drag_event(self, x, y, dx, dy):
        if not _cam_input.mouse_dragging:
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
    def _update_camera_keys(self, dt):
        global _cam_vel
        ci = _cam_input
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
        if 'q' in ci.keys_pressed:
            target = [target[i] - up[i]    * spd for i in range(3)]
        if 'e' in ci.keys_pressed:
            target = [target[i] + up[i]    * spd for i in range(3)]
        alpha = 1.0 - math.exp(-self.SMOOTHING * dt)
        _cam_vel = [_cam_vel[i] + (target[i] - _cam_vel[i]) * alpha for i in range(3)]
        p = _params.cam_pos
        _params.cam_pos = [p[i] + _cam_vel[i] * dt for i in range(3)]
    def render(self, t, ft):
        self._update_camera_keys(ft)
        p = _params
        fwd, right, up = self._calc_basis()
        self.ctx.clear(0, 0, 0)
        elapsed = time.time() - self.start
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
        self._set('u_cam_pos',      tuple(p.cam_pos))
        self._set('u_cam_fwd',      fwd)
        self._set('u_cam_right',    right)
        self._set('u_cam_up',       up)
        self._set('u_animate',      1 if p.animate else 0)
        self._set('u_anim_speed',   p.anim_speed)
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
        # Light
        self._set('u_light_dir',       (p.light_x, p.light_y, p.light_z))
        self._set('u_specular_power',  p.specular_power)
        self._set('u_specular_strength', p.specular_strength)
        self._set('u_ambient',         p.ambient)
        # Color animation
        self._set('u_color_anim_speed', p.color_anim_speed)
        self._set('u_color_offset',    p.color_offset)
        # Raymarching
        self._set('u_step_scale',      p.step_scale)
        self._set('u_normal_eps',      p.normal_eps)
        self.vao.render(moderngl.TRIANGLE_STRIP)

        # Screenshot: capture framebuffer after render
        if self._pending_screenshot or p.screenshot_requested:
            self._pending_screenshot  = False
            p.screenshot_requested    = False
            self._save_screenshot()

    def _save_screenshot(self):
        try:
            import datetime
            w, h = self.wnd.size
            data = self.ctx.fbo.read(components=3)
            from PIL import Image
            img  = Image.frombytes('RGB', (w, h), data)
            img  = img.transpose(Image.FLIP_TOP_BOTTOM)
            ts   = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            path = Path(__file__).parent / f'fractal_{ts}.png'
            img.save(str(path))
            print(f'[screenshot] saved → {path}')
            # notify GUI if callback registered
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
        self._build()
        self._cam_timer = QTimer(self)
        self._cam_timer.timeout.connect(self._sync_camera_ui)
        self._cam_timer.start(self.CAM_SYNC_MS)
    def _build(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { border: none; }")
        container = QWidget()
        container.setStyleSheet(f"background: {COLORS['bg']};")
        self._vbox = QVBoxLayout(container)
        self._vbox.setSpacing(4)
        self._vbox.setContentsMargins(8, 8, 8, 8)
        title = _label("KALEIDOSCOPIC IFS", COLORS['accent'], FONT_TITLE)
        title.setAlignment(Qt.AlignCenter)
        self._vbox.addWidget(title)
        sub = _label("Ray-marched fractal renderer", COLORS['fg4'], FONT_SMALL)
        sub.setAlignment(Qt.AlignCenter)
        self._vbox.addWidget(sub)
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
        self._vbox.addWidget(hint)
        self._build_fractal_section()
        self._build_ifs_section()
        self._build_transform_section()
        self._build_camera_section()
        self._build_color_section()
        self._build_background_section()
        self._build_glow_section()
        self._build_lighting_section()
        self._build_light_direction_section()
        self._build_raymarching_section()
        self._build_aa_section()
        self._build_presets_section()
        self._vbox.addStretch()
        scroll.setWidget(container)
        self.setCentralWidget(scroll)
    def _add_section(self, widget):
        self._vbox.addWidget(widget)
    def _build_fractal_section(self):
        grp = _section("FRACTAL TYPE")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        self._type_grp = QButtonGroup(self)
        row = QHBoxLayout()
        for i, name in enumerate(["Mandelbox", "Menger Sponge", "Sierpinski", "Octahedron IFS"]):
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
        # ---- Shared base params (always visible) ----
        base_grp = _section("BASE PARAMETERS")
        base_layout = QVBoxLayout(base_grp)
        base_layout.setSpacing(2)
        for label, attr, mn, mx, val, step in [
            ("Scale",    'scale',    -3.0, 3.0,  _params.scale,    0.01),
            ("Bailout",  'bailout',   1.0, 50.0, _params.bailout,   0.1),
            ("Min Dist", 'min_dist',  0.1, 5.0,  _params.min_dist,  0.1),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            base_layout.addWidget(sr)
        self._add_section(base_grp)

        # ---- Per-fractal panels (only one visible at a time) ----
        self._fractal_panels = {}

        # --- Mandelbox ---
        mb = _section("MANDELBOX FINE-TUNE")
        mb_l = QVBoxLayout(mb)
        mb_l.setSpacing(2)
        _lbl_hint(mb_l, "Box fold & sphere fold radii")
        for label, attr, mn, mx, val, step in [
            ("Fold Limit",    'mb_fold_limit',   0.1, 3.0,  _params.mb_fold_limit,   0.01),
            ("Sph Inner r²",  'mb_sphere_inner', 0.01,1.0,  _params.mb_sphere_inner, 0.005),
            ("Sph Outer r²",  'mb_sphere_outer', 0.1, 4.0,  _params.mb_sphere_outer, 0.01),
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

        # --- Menger Sponge ---
        ms = _section("MENGER SPONGE FINE-TUNE")
        ms_l = QVBoxLayout(ms)
        ms_l.setSpacing(2)
        _lbl_hint(ms_l, "IFS scale, cross gap & per-level twist")
        for label, attr, mn, mx, val, step in [
            ("IFS Scale",    'ms_scale',       2.0, 5.0, _params.ms_scale,       0.01),
            ("IFS Offset",   'ms_offset',      1.0, 4.0, _params.ms_offset,      0.01),
            ("Cross Width",  'ms_cross_width', 0.0, 4.0, _params.ms_cross_width, 0.01),
            ("Level Twist",  'ms_twist',       0.0, 0.3, _params.ms_twist,       0.001),
            ("Edge Sharp",   'ms_sharpness',   0.1, 1.0, _params.ms_sharpness,   0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            ms_l.addWidget(sr)
        self._fractal_panels[1] = ms
        self._add_section(ms)

        # --- Sierpinski ---
        si = _section("SIERPINSKI FINE-TUNE")
        si_l = QVBoxLayout(si)
        si_l.setSpacing(2)
        _lbl_hint(si_l, "Vertex spread, fold bias, squash & twist")
        for label, attr, mn, mx, val, step in [
            ("Vertex Spread", 'si_vertex_spread', 0.2, 3.0, _params.si_vertex_spread, 0.01),
            ("Fold Bias",     'si_fold_bias',      1.2, 4.0, _params.si_fold_bias,     0.01),
            ("Y Squash",      'si_squash',         0.2, 3.0, _params.si_squash,        0.01),
            ("Level Twist",   'si_twist',          0.0, 0.3, _params.si_twist,         0.001),
            ("Vtx Jitter",    'si_vertex_jitter',  0.0, 1.0, _params.si_vertex_jitter, 0.005),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            si_l.addWidget(sr)
        self._fractal_panels[2] = si
        self._add_section(si)

        # --- Octahedron IFS ---
        oc = _section("OCTAHEDRON IFS FINE-TUNE")
        oc_l = QVBoxLayout(oc)
        oc_l.setSpacing(2)
        _lbl_hint(oc_l, "IFS scale, offset, norm sharpness & twist")
        for label, attr, mn, mx, val, step in [
            ("IFS Scale",     'oc_ifs_scale',  1.2, 4.0, _params.oc_ifs_scale,  0.01),
            ("Offset X",      'offset_x',      0.1, 4.0, _params.offset_x,      0.01),
            ("Offset Y",      'offset_y',      0.1, 4.0, _params.offset_y,      0.01),
            ("Offset Z",      'offset_z',      0.1, 4.0, _params.offset_z,      0.01),
            ("Offset Uni",    'oc_offset_uni', 0.1, 3.0, _params.oc_offset_uni, 0.01),
            ("Norm Sharp",    'oc_sharpness',  0.5, 4.0, _params.oc_sharpness,  0.05),
            ("Level Twist",   'oc_twist',      0.0, 0.3, _params.oc_twist,      0.001),
            ("Fold Amount",   'oc_fold_amount',0.0, 1.0, _params.oc_fold_amount,0.01),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            oc_l.addWidget(sr)
        self._fractal_panels[3] = oc
        self._add_section(oc)

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
        self._add_section(grp)
    def _reset_camera(self):
        _params.cam_pos   = [0.0, 0.0, 5.0]
        _params.cam_yaw   = 0.0
        _params.cam_pitch = 0.0
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
        grp = _section("LIGHTING")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        for label, attr, mn, mx, val, step in [
            ("AO Strength", 'ao_strength', 0.0, 30.0, _params.ao_strength, 0.01),
            ("Shadow Soft", 'shadow_soft', 1.0, 32.0, _params.shadow_soft, 0.5),
            ("Spec Power",  'glow',        0.0, 10.0, _params.glow,        0.1),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._shadows_check = QCheckBox("Soft Shadows")
        self._shadows_check.setFont(FONT_MONO)
        self._shadows_check.setStyleSheet(_css_check())
        self._shadows_check.setChecked(_params.shadows)
        self._shadows_check.stateChanged.connect(lambda s: setattr(_params, 'shadows', bool(s)))
        layout.addWidget(self._shadows_check)
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
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._add_section(grp)

    def _build_raymarching_section(self):
        grp = _section("RAYMARCHING & COLOR")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        _lbl_hint(layout, "Marching step, normal precision and palette control")
        for label, attr, mn, mx, val, step in [
            ("Step Scale",   'step_scale',       0.1, 1.0,  _params.step_scale,       0.005),
            ("Normal Eps",   'normal_eps',       0.0001, 0.01, _params.normal_eps,     0.0001),
            ("Clr Anim Spd", 'color_anim_speed', 0.0, 0.5,  _params.color_anim_speed, 0.002),
            ("Clr Offset",   'color_offset',     0.0, 1.0,  _params.color_offset,     0.005),
        ]:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._add_section(grp)

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
            'mb_fold_limit','mb_sphere_inner','mb_sphere_outer','mb_fixed_radius',
            'mb_color_scale','mb_rot_per_iter',
            'julia_x','julia_y','julia_z',
            'ms_scale','ms_offset','ms_cross_width','ms_twist','ms_sharpness',
            'si_vertex_spread','si_fold_bias','si_squash','si_twist','si_vertex_jitter',
            'oc_ifs_scale','oc_sharpness','oc_twist','oc_offset_uni','oc_fold_amount',
            'offset_x','offset_y','offset_z',
            'light_x','light_y','light_z',
            'specular_power','specular_strength','ambient',
            'color_anim_speed','color_offset',
            'step_scale','normal_eps',
        ]:
            sl = getattr(self, f'_sl_{attr}', None)
            if sl is not None:
                sl.set_value(getattr(_params, attr))

def run_gl():
    mglw_settings.WINDOW = {
        'class':        'moderngl_window.context.pyglet.Window',
        'gl_version':   (4, 6),
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