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

APP_VERSION = "1.0.0"

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

void kifs_fold(inout vec3 p) {
    if (u_fold_x > 0.5) p.x = abs(p.x);
    if (u_fold_y > 0.5) p.y = abs(p.y);
    if (u_fold_z > 0.5) p.z = abs(p.z);
    if (p.x - p.y < 0.0) p.xy = p.yx;
    if (p.x - p.z < 0.0) p.xz = p.zx;
    if (p.y - p.z < 0.0) p.yz = p.zy;
}

void mengerFold(inout vec3 p) {
    if (p.x - p.y < 0.0) p.xy = p.yx;
    if (p.x - p.z < 0.0) p.xz = p.zx;
    if (p.y - p.z < 0.0) p.yz = p.zy;
}

float sdBox(vec3 p, vec3 b) {
    vec3 d = abs(p) - b;
    return length(max(d, 0.0)) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float sdTetra(vec3 p) {
    return (max(abs(p.x+p.y)-p.z, abs(p.x-p.y)+p.z) - 1.0) / sqrt(3.0);
}

vec2 mandelbox(vec3 pos) {
    vec3 p = pos;
    float trap = 1e10;
    float dr = 1.0;
    for (int i = 0; i < u_iterations; i++) {
        p = clamp(p, -1.0, 1.0) * 2.0 - p;
        float r2 = dot(p, p);
        if (r2 < 0.25) { float k = 4.0; p *= k; dr *= k; }
        else if (r2 < 1.0) { float k = 1.0/r2; p *= k; dr *= k; }
        p = p * u_scale + vec3(u_julia_x, u_julia_y, u_julia_z);
        dr = dr * abs(u_scale) + 1.0;
        trap = min(trap, dot(p,p));
        if (dot(p,p) > u_bailout * u_bailout) break;
    }
    return vec2(length(p) / abs(dr), trap);
}

vec2 mengerSponge(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    for (int i = 0; i < u_iterations; i++) {
        mengerFold(p);
        p *= 3.0;
        s *= 3.0;
        p -= vec3(1.0);
        trap = min(trap, dot(p,p) / (s*s));
    }
    return vec2(sdBox(p, vec3(1.0)) / s, trap);
}

vec2 sierpinski(vec3 pos) {
    vec3 p = pos;
    float scale = 1.0;
    float trap = 1e10;
    for (int i = 0; i < u_iterations; i++) {
        kifs_fold(p);
        p *= 2.0;
        p -= vec3(1.0);
        scale *= 2.0;
        trap = min(trap, dot(p,p) / (scale*scale));
    }
    return vec2(sdTetra(p) / scale, trap);
}

vec2 octahedronIFS(vec3 pos) {
    vec3 p = pos;
    float s = 1.0;
    float trap = 1e10;
    for (int i = 0; i < u_iterations; i++) {
        p = abs(p);
        if (p.x < p.y) p.xy = p.yx;
        if (p.x < p.z) p.xz = p.zx;
        if (p.y < p.z) p.yz = p.zy;
        p *= u_scale;
        p -= vec3(u_offset_x, u_offset_y, u_offset_z);
        s *= abs(u_scale);
        trap = min(trap, dot(p,p) / (s*s));
    }
    return vec2((length(p) - 0.5) / s, trap);
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
    float e = 0.001;
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

void main() {
    float aspect = u_resolution.x / u_resolution.y;
    vec2 uv = vec2(v_uv.x * aspect, v_uv.y);
    float t = u_animate == 1 ? u_time * u_anim_speed : 0.0;

    vec3 ro = u_cam_pos;
    vec3 rd = normalize(uv.x * u_cam_right + uv.y * u_cam_up + 1.5 * u_cam_fwd);

    vec3 col = vec3(0.0);
    float totalDist = 0.0;
    float minDist = 1e10;
    float trap = 0.0;
    int steps = 0;
    bool hit = false;

    for (int i = 0; i < MAX_STEPS; i++) {
        vec3 p = ro + rd * totalDist;
        vec2 res = sceneDist(p);
        float d = res.x;
        trap = res.y;
        minDist = min(minDist, d);
        if (d < u_min_dist * 0.001) { hit = true; steps = i; break; }
        if (totalDist > MAX_DIST) break;
        totalDist += d * 0.5;
        steps = i;
    }

    if (hit) {
        vec3 p = ro + rd * totalDist;
        vec3 n = calcNormal(p);
        vec3 lightDir = normalize(vec3(1.0, 2.0, 1.5));
        float diff = max(dot(n, lightDir), 0.0);
        float spec = pow(max(dot(reflect(-lightDir, n), -rd), 0.0), 32.0);
        float ao = ambientOcclusion(p, n);
        float shadow = u_shadows == 1 ? softShadow(p, lightDir, 0.02, 10.0, u_shadow_soft) : 1.0;

        float colorParam;
        if (u_color_mode == 0) colorParam = float(steps) / float(u_iterations * 2);
        else if (u_color_mode == 1) colorParam = clamp(sqrt(trap) * 0.5, 0.0, 1.0);
        else if (u_color_mode == 2) colorParam = n.x * 0.5 + 0.5;
        else colorParam = clamp(totalDist / MAX_DIST, 0.0, 1.0);

        vec3 baseCol = palette(colorParam + t * 0.05);
        col = baseCol * (diff * shadow * 0.8 + 0.2) * ao;
        col += spec * 0.3 * shadow;
        col += baseCol * u_glow * 0.2;

        float fog = exp(-totalDist * u_fog_density * 0.1);
        col = mix(vec3(0.0), col, fog);
    } else {
        float glow = exp(-minDist * 8.0) * u_glow;
        col = palette(float(steps)/float(MAX_STEPS)) * glow * 0.5;
        col += vec3(0.02, 0.03, 0.06) * (1.0 - glow);
    }

    col = pow(clamp(col, 0.0, 1.0), vec3(0.4545));
    fragColor = vec4(col, 1.0);
}
"""

class FractalParams:
    def __init__(self):
        self.iterations   = 32
        self.scale        = 3.0
        self.fold_x       = True
        self.fold_y       = True
        self.fold_z       = True
        self.rot_x        = 0.0
        self.rot_y        = 0.0
        self.rot_z        = 0.0
        self.offset_x     = 1.0
        self.offset_y     = 1.0
        self.offset_z     = 1.0
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

_params = FractalParams()

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
        self.vao.render(moderngl.TRIANGLE_STRIP)

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
        self._slider.valueChanged.connect(self._on_change)
        self.set_value(value)
    def _float_to_int(self, v: float) -> int:
        return round((v - self._mn) / (self._mx - self._mn) * self.SLIDER_STEPS)
    def _int_to_float(self, i: int) -> float:
        raw = self._mn + i / self.SLIDER_STEPS * (self._mx - self._mn)
        return round(raw / self._step) * self._step
    def set_value(self, v: float):
        self._slider.blockSignals(True)
        self._slider.setValue(self._float_to_int(v))
        self._slider.blockSignals(False)
        self._val_lbl.setText(f'{v:.2f}')
    def get_value(self) -> float:
        return self._int_to_float(self._slider.value())
    def _on_change(self, _):
        v = self.get_value()
        self._val_lbl.setText(f'{v:.2f}')
        for cb in self._callbacks:
            cb(v)
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
        self._build_lighting_section()
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
        self._type_grp.idClicked.connect(lambda idx: setattr(_params, 'fractal_type', idx))
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
        grp = _section("IFS PARAMETERS")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        sliders = [
            ("Scale",    'scale',    -3.0, 3.0,  _params.scale,    0.01),
            ("Bailout",  'bailout',   1.0, 20.0, _params.bailout,   0.1),
            ("Min Dist", 'min_dist',  0.1, 5.0,  _params.min_dist,  0.1),
        ]
        for label, attr, mn, mx, val, step in sliders:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        folds_lbl = _label("Folds:", COLORS['fg3'], FONT_SMALL)
        layout.addWidget(folds_lbl)
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
        layout.addLayout(folds_row)
        julia_lbl = _label("Julia / Offset", COLORS['fg3'], FONT_SMALL)
        layout.addWidget(julia_lbl)
        julia_sliders = [
            ("Julia X",  'julia_x',  -20.0, 20.0, _params.julia_x,  0.01),
            ("Julia Y",  'julia_y',  -20.0, 20.0, _params.julia_y,  0.01),
            ("Julia Z",  'julia_z',  -20.0, 20.0, _params.julia_z,  0.01),
            ("Offset X", 'offset_x',   0.1,  3.0, _params.offset_x, 0.01),
            ("Offset Y", 'offset_y',   0.1,  3.0, _params.offset_y, 0.01),
            ("Offset Z", 'offset_z',   0.1,  3.0, _params.offset_z, 0.01),
        ]
        for label, attr, mn, mx, val, step in julia_sliders:
            sr = SliderRow(label, mn, mx, val, step)
            sr.on_change(lambda v, a=attr: setattr(_params, a, v))
            setattr(self, f'_sl_{attr}', sr)
            layout.addWidget(sr)
        self._add_section(grp)
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
    def _build_lighting_section(self):
        grp = _section("LIGHTING")
        layout = QVBoxLayout(grp)
        layout.setSpacing(2)
        for label, attr, mn, mx, val, step in [
            ("AO Strength", 'ao_strength', 0.0, 30.0,  _params.ao_strength, 0.01),
            ("Shadow Soft", 'shadow_soft', 1.0, 32.0, _params.shadow_soft, 0.5),
            ("Glow",        'glow',        0.0, 10.0,  _params.glow,        0.1),
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
    def _build_presets_section(self):
        grp = _section("PRESETS")
        layout = QGridLayout(grp)
        layout.setSpacing(4)
        presets = {
            "Mandelbox Classic": dict(fractal_type=0, scale=-2.0, iterations=8,
                julia_x=-0.5, julia_y=-0.5, julia_z=-0.5, cam_dist=5.0,
                color_mode=0, glow=1.0),
            "Menger Deep":       dict(fractal_type=1, scale=3.0, iterations=6,
                color_mode=1, ao_strength=1.5, glow=0.5, cam_dist=6.0),
            "Sierpinski Fire":   dict(fractal_type=2, scale=2.0, iterations=10,
                color_mode=0, glow=2.0, cam_dist=4.0,
                color1=(1.0,0.3,0.0), color2=(1.0,0.8,0.0), color3=(0.5,0.0,0.0)),
            "Octahedron Void":   dict(fractal_type=3, scale=-1.8, iterations=9,
                offset_x=1.5, offset_y=1.5, offset_z=1.5,
                color_mode=2, shadows=True, glow=1.5, cam_dist=7.0),
        }
        for i, (name, vals) in enumerate(presets.items()):
            btn = QPushButton(name)
            btn.setFont(FONT_SMALL)
            btn.setStyleSheet(_css_button())
            btn.clicked.connect(lambda _, v=vals: self._apply_preset(v))
            layout.addWidget(btn, i // 2, i % 2)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        self._add_section(grp)
    def _apply_preset(self, vals: dict):
        for k, v in vals.items():
            if k == 'cam_dist':
                _params.cam_pos   = [0.0, 0.0, float(v)]
                _params.cam_yaw   = 0.0
                _params.cam_pitch = 0.0
                continue
            setattr(_params, k, v)
        self._type_grp.button(_params.fractal_type).setChecked(True)
        self._iter_slider.setValue(_params.iterations)
        self._sl_scale.set_value(_params.scale)
        self._sl_julia_x.set_value(_params.julia_x)
        self._sl_julia_y.set_value(_params.julia_y)
        self._sl_julia_z.set_value(_params.julia_z)
        self._sl_offset_x.set_value(_params.offset_x)
        self._sl_offset_y.set_value(_params.offset_y)
        self._sl_offset_z.set_value(_params.offset_z)
        self._sl_glow.set_value(_params.glow)
        self._sl_ao_strength.set_value(_params.ao_strength)
        self._cmode_grp.button(_params.color_mode).setChecked(True)
        for btn, attr in [(self._c1_btn,'color1'),(self._c2_btn,'color2'),(self._c3_btn,'color3')]:
            hex_col = self._rgb_to_hex(getattr(_params, attr))
            btn.setStyleSheet(
                f"QPushButton {{ background: {hex_col}; color: white; border: none;"
                "border-radius: 4px; padding: 4px 8px; font: bold 8pt Consolas; }}"
                f"QPushButton:hover {{ border: 2px solid {COLORS['accent']}; }}"
            )

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