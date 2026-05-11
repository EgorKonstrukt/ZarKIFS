from __future__ import annotations
import json
import math
import time
import bisect as _bisect
from pathlib import Path
from PyQt5.QtCore import Qt, QTimer, QPointF, QRectF, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QFont, QPainterPath
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QScrollArea, QSplitter, QListWidget,
    QListWidgetItem, QMenu, QInputDialog, QMessageBox,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSizePolicy, QAbstractItemView, QFrame, QGridLayout,
    QDialog, QDialogButtonBox, QLineEdit, QTreeWidget, QSlider,
    QTreeWidgetItem, QHeaderView, QStyle,
)

try:
    import app_config as _app_config
except ImportError:
    _app_config = None

ANIM_VERSION = 1

COLORS = {
    'bg':       '#0f0f1e',
    'bg2':      '#12122a',
    'bg3':      '#1a1a2e',
    'panel':    '#1e1e3a',
    'panel2':   '#252545',
    'border':   '#2d2d5e',
    'accent':   '#7b68ee',
    'accent2':  '#5a4fc8',
    'accent3':  '#9d8fff',
    'sel':      '#3a2d80',
    'track':    '#181830',
    'track2':   '#1c1c38',
    'key_free': '#7b68ee',
    'key_lin':  '#55cc88',
    'key_step': '#cc7744',
    'playhead': '#ff5555',
    'fg':       '#e0e0ff',
    'fg2':      '#8888cc',
    'fg3':      '#aaaacc',
    'fg4':      '#5050a0',
}

def _pal(widget, role):
    return widget.palette().color(role)

def _canvas_bg(widget):
    return _pal(widget, widget.palette().Base)

def _canvas_bg_alt(widget):
    return _pal(widget, widget.palette().AlternateBase)

def _canvas_text(widget):
    return _pal(widget, widget.palette().Text)

def _canvas_mid(widget):
    return _pal(widget, widget.palette().Mid)

def _canvas_midlight(widget):
    return _pal(widget, widget.palette().Midlight)

def _canvas_highlight(widget):
    return _pal(widget, widget.palette().Highlight)

def _is_dark_theme():
    app = QApplication.instance()
    if app is None:
        return True
    bg = app.palette().color(app.palette().Window)
    return bg.lightness() < 128

def _canvas_font(size=8, bold=False, mono=False):
    f = QFont('Courier New' if mono else 'Segoe UI', size)
    if bold:
        f.setBold(True)
    return f

def _c(key):
    return QColor(COLORS[key])

INTERP_NONE      = 0
INTERP_LIN       = 1
INTERP_CUBIC     = 2
INTERP_EASE_IN   = 3
INTERP_EASE_OUT  = 4
INTERP_EASE_IO   = 5
INTERP_BOUNCE    = 6
INTERP_ELASTIC   = 7
INTERP_BACK      = 8
INTERP_EXPO      = 9

INTERP_TABLE = {
    INTERP_NONE:     ('Step',               COLORS['key_step'], None),
    INTERP_LIN:      ('Linear',             COLORS['key_lin'],  None),
    INTERP_CUBIC:    ('Smooth (Catmull-Rom)',COLORS['key_free'], None),
    INTERP_EASE_IN:  ('Ease In',            '#aa88ff',          None),
    INTERP_EASE_OUT: ('Ease Out',           '#88ffcc',          None),
    INTERP_EASE_IO:  ('Ease In-Out',        '#ffcc55',          None),
    INTERP_BOUNCE:   ('Bounce',             '#ff8844',          None),
    INTERP_ELASTIC:  ('Elastic',            '#ff55aa',          None),
    INTERP_BACK:     ('Back',               '#55ddff',          None),
    INTERP_EXPO:     ('Exponential',        '#aaffaa',          None),
}

INTERP_LABELS = {k: v[0] for k, v in INTERP_TABLE.items()}
INTERP_COLORS = {k: v[1] for k, v in INTERP_TABLE.items()}

PROP_FLOAT = 'float'
PROP_COLOR = 'color'
PROP_BOOL  = 'bool'
PROP_INT   = 'int'

ALL_PROPERTIES = {
    'scale':             (PROP_FLOAT, 0.1,   10.0),
    'iterations':        (PROP_INT,   1,     20),
    'rot_x':             (PROP_FLOAT, -3.14, 3.14),
    'rot_y':             (PROP_FLOAT, -3.14, 3.14),
    'rot_z':             (PROP_FLOAT, -3.14, 3.14),
    'offset_x':          (PROP_FLOAT, -5.0,  5.0),
    'offset_y':          (PROP_FLOAT, -5.0,  5.0),
    'offset_z':          (PROP_FLOAT, -5.0,  5.0),
    'julia_x':           (PROP_FLOAT, -2.0,  2.0),
    'julia_y':           (PROP_FLOAT, -2.0,  2.0),
    'julia_z':           (PROP_FLOAT, -2.0,  2.0),
    'bailout':           (PROP_FLOAT, 1.0,   50.0),
    'fog_density':       (PROP_FLOAT, 0.0,   5.0),
    'ao_strength':       (PROP_FLOAT, 0.0,   3.0),
    'ao_radius':         (PROP_FLOAT, 0.01,  1.0),
    'shadow_soft':       (PROP_FLOAT, 0.5,   32.0),
    'glow':              (PROP_FLOAT, 0.0,   20.0),
    'glow_intensity':    (PROP_FLOAT, 0.0,   20.0),
    'glow_falloff':      (PROP_FLOAT, 0.5,   20.0),
    'glow_radius':       (PROP_FLOAT, 0.0,   5.0),
    'rim_strength':      (PROP_FLOAT, 0.0,   3.0),
    'emission':          (PROP_FLOAT, 0.0,   3.0),
    'anim_speed':        (PROP_FLOAT, 0.0,   10.0),
    'fov':               (PROP_FLOAT, 0.3,   3.0),
    'de_multiplier':     (PROP_FLOAT, 0.1,   5.0),
    'color_offset':      (PROP_FLOAT, 0.0,   1.0),
    'color_anim_speed':  (PROP_FLOAT, 0.0,   1.0),
    'step_scale':        (PROP_FLOAT, 0.1,   1.5),
    'normal_eps':        (PROP_FLOAT, 0.0001,0.01),
    'reflection':        (PROP_FLOAT, 0.0,   1.0),
    'max_dist':          (PROP_FLOAT, 10.0,  500.0),
    'hit_eps':           (PROP_FLOAT, 0.1,   5.0),
    'ambient':           (PROP_FLOAT, 0.0,   1.0),
    'specular_power':    (PROP_FLOAT, 1.0,   128.0),
    'specular_strength': (PROP_FLOAT, 0.0,   2.0),
    'subsurface':        (PROP_FLOAT, 0.0,   1.0),
    'fresnel_power':     (PROP_FLOAT, 0.5,   15.0),
    'light_x':           (PROP_FLOAT, -5.0,  5.0),
    'light_y':           (PROP_FLOAT, -5.0,  5.0),
    'light_z':           (PROP_FLOAT, -5.0,  5.0),
    'light2_strength':   (PROP_FLOAT, 0.0,   3.0),
    'light2_x':          (PROP_FLOAT, -5.0,  5.0),
    'light2_y':          (PROP_FLOAT, -5.0,  5.0),
    'light2_z':          (PROP_FLOAT, -5.0,  5.0),
    'warp_strength':     (PROP_FLOAT, 0.0,   2.0),
    'warp_freq':         (PROP_FLOAT, 0.1,   10.0),
    'twist_amount':      (PROP_FLOAT, 0.0,   5.0),
    'rep_cell_x':        (PROP_FLOAT, 0.5,   20.0),
    'rep_cell_y':        (PROP_FLOAT, 0.5,   20.0),
    'rep_cell_z':        (PROP_FLOAT, 0.5,   20.0),
    'gamma':             (PROP_FLOAT, 0.5,   4.0),
    'exposure':          (PROP_FLOAT, 0.1,   5.0),
    'saturation':        (PROP_FLOAT, 0.0,   3.0),
    'mb_fold_limit':     (PROP_FLOAT, 0.1,   5.0),
    'mb_sphere_inner':   (PROP_FLOAT, 0.01,  2.0),
    'mb_sphere_outer':   (PROP_FLOAT, 0.1,   3.0),
    'mb_fixed_radius':   (PROP_FLOAT, 0.1,   3.0),
    'mb_rot_per_iter':   (PROP_FLOAT, 0.0,   1.0),
    'ms_cross_width':    (PROP_FLOAT, 0.0,   3.0),
    'ms_scale':          (PROP_FLOAT, 1.5,   6.0),
    'ms_offset':         (PROP_FLOAT, 0.5,   5.0),
    'ms_twist':          (PROP_FLOAT, 0.0,   1.0),
    'si_vertex_spread':  (PROP_FLOAT, 0.3,   3.0),
    'si_fold_bias':      (PROP_FLOAT, 1.3,   4.0),
    'si_squash':         (PROP_FLOAT, 0.1,   3.0),
    'oc_ifs_scale':      (PROP_FLOAT, 1.3,   4.0),
    'oc_twist':          (PROP_FLOAT, 0.0,   1.0),
    'oc_fold_amount':    (PROP_FLOAT, 0.0,   1.0),
    'mb2_power':         (PROP_FLOAT, 2.0,   16.0),
    'mb2_bailout':       (PROP_FLOAT, 1.0,   10.0),
    'mb2_fold_strength': (PROP_FLOAT, 0.0,   3.0),
    'kl_scale':          (PROP_FLOAT, 0.5,   3.0),
    'kl_fold_limit':     (PROP_FLOAT, 0.1,   3.0),
    'kl_sph_radius':     (PROP_FLOAT, 0.1,   2.0),
    'kl_mix_factor':     (PROP_FLOAT, 0.0,   1.0),
    'kl_fold_limit_x':   (PROP_FLOAT, 0.0,   3.0),
    'kl_fold_limit_y':   (PROP_FLOAT, 0.0,   3.0),
    'kl_fold_limit_z':   (PROP_FLOAT, 0.0,   3.0),
    'kl_offset_x':       (PROP_FLOAT, -3.0,  3.0),
    'kl_offset_y':       (PROP_FLOAT, -3.0,  3.0),
    'kl_offset_z':       (PROP_FLOAT, -3.0,  3.0),
    'mb_scale_x':        (PROP_FLOAT, 0.0,   5.0),
    'mb_scale_y':        (PROP_FLOAT, 0.0,   5.0),
    'mb_scale_z':        (PROP_FLOAT, 0.0,   5.0),
    'mb_offset_x':       (PROP_FLOAT, -5.0,  5.0),
    'mb_offset_y':       (PROP_FLOAT, -5.0,  5.0),
    'mb_offset_z':       (PROP_FLOAT, -5.0,  5.0),
    'mb_inversion_radius': (PROP_FLOAT, 0.0, 4.0),
    'mb2_polar_mix':     (PROP_FLOAT, 0.0,   1.0),
    'mb2_rot_per_iter':  (PROP_FLOAT, 0.0,   0.5),
    'ms_offset_x':       (PROP_FLOAT, -4.0,  4.0),
    'ms_offset_y':       (PROP_FLOAT, -4.0,  4.0),
    'ms_offset_z':       (PROP_FLOAT, -4.0,  4.0),
    'ms_fold_abs_amount':(PROP_FLOAT, 0.0,   3.0),
    'si_scale_x':        (PROP_FLOAT, 0.0,   4.0),
    'si_scale_y':        (PROP_FLOAT, 0.0,   4.0),
    'si_scale_z':        (PROP_FLOAT, 0.0,   4.0),
    'si_offset_x':       (PROP_FLOAT, -3.0,  3.0),
    'si_offset_y':       (PROP_FLOAT, -3.0,  3.0),
    'si_offset_z':       (PROP_FLOAT, -3.0,  3.0),
    'si_rot_y':          (PROP_FLOAT, 0.0,   1.3),
    'oc_scale_y':        (PROP_FLOAT, 0.0,   4.0),
    'oc_scale_z':        (PROP_FLOAT, 0.0,   4.0),
    'oc_julia_x':        (PROP_FLOAT, -3.0,  3.0),
    'oc_julia_y':        (PROP_FLOAT, -3.0,  3.0),
    'oc_julia_z':        (PROP_FLOAT, -3.0,  3.0),
    'sph_inv_radius':    (PROP_FLOAT, 0.1,   5.0),
    'sph_inv_cx':        (PROP_FLOAT, -3.0,  3.0),
    'sph_inv_cy':        (PROP_FLOAT, -3.0,  3.0),
    'sph_inv_cz':        (PROP_FLOAT, -3.0,  3.0),
    'lattice_fold_x':    (PROP_FLOAT, 0.2,   8.0),
    'lattice_fold_y':    (PROP_FLOAT, 0.2,   8.0),
    'lattice_fold_z':    (PROP_FLOAT, 0.2,   8.0),
    'cam_pos_x':         (PROP_FLOAT, -50.0, 50.0),
    'cam_pos_y':         (PROP_FLOAT, -50.0, 50.0),
    'cam_pos_z':         (PROP_FLOAT, -50.0, 50.0),
    'cam_yaw':           (PROP_FLOAT, -6.28, 6.28),
    'cam_pitch':         (PROP_FLOAT, -1.55, 1.55),
    'color1':            (PROP_COLOR, None,  None),
    'color2':            (PROP_COLOR, None,  None),
    'color3':            (PROP_COLOR, None,  None),
    'bg_color1':         (PROP_COLOR, None,  None),
    'bg_color2':         (PROP_COLOR, None,  None),
    'fog_color':         (PROP_COLOR, None,  None),
    'light2_color':      (PROP_COLOR, None,  None),
    'shadows':           (PROP_BOOL,  None,  None),
    'animate':           (PROP_BOOL,  None,  None),
    'warp_enabled':      (PROP_BOOL,  None,  None),
    'rep_enabled':       (PROP_BOOL,  None,  None),
    'fold_mirror_x':     (PROP_BOOL,  None,  None),
    'fold_mirror_y':     (PROP_BOOL,  None,  None),
    'fold_mirror_z':     (PROP_BOOL,  None,  None),
    'sph_inv_enabled':   (PROP_BOOL,  None,  None),
    'lattice_fold_enabled': (PROP_BOOL, None, None),
    'mb2_abs_x':         (PROP_BOOL,  None,  None),
    'mb2_abs_y':         (PROP_BOOL,  None,  None),
    'mb2_abs_z':         (PROP_BOOL,  None,  None),
    'fractal_type':      (PROP_INT,   0,     5),
    'color_mode':        (PROP_INT,   0,     3),
    'bg_mode':           (PROP_INT,   0,     3),
    'aa_samples':        (PROP_INT,   1,     3),
    'max_steps':         (PROP_INT,   32,    512),
    'shadow_steps':      (PROP_INT,   4,     64),
    'ms_fold_type':      (PROP_INT,   0,     2),
    'oc_julia_mode':     (PROP_INT,   0,     1),
    'kl_julia_mode':     (PROP_INT,   0,     1),
}

PROP_GROUPS = {
    'Camera':       ['cam_pos_x', 'cam_pos_y', 'cam_pos_z', 'cam_yaw', 'cam_pitch', 'fov'],
    'Fractal Core': ['fractal_type', 'iterations', 'scale', 'bailout', 'de_multiplier',
                     'rot_x', 'rot_y', 'rot_z', 'offset_x', 'offset_y', 'offset_z',
                     'julia_x', 'julia_y', 'julia_z'],
    'Colors':       ['color_mode', 'color1', 'color2', 'color3', 'color_offset', 'color_anim_speed'],
    'Lighting':     ['ambient', 'specular_power', 'specular_strength', 'subsurface', 'fresnel_power',
                     'light_x', 'light_y', 'light_z', 'light2_x', 'light2_y', 'light2_z',
                     'light2_strength', 'light2_color', 'shadows'],
    'Glow & Fog':   ['glow', 'glow_intensity', 'glow_falloff', 'glow_radius', 'rim_strength',
                     'emission', 'fog_density', 'fog_color'],
    'Background':   ['bg_mode', 'bg_color1', 'bg_color2'],
    'Post-Process': ['gamma', 'exposure', 'saturation', 'aa_samples'],
    'Raymarching':  ['max_steps', 'step_scale', 'normal_eps', 'reflection', 'max_dist',
                     'hit_eps', 'shadow_steps', 'shadow_soft', 'ao_strength', 'ao_radius'],
    'Space Ops':    ['warp_enabled', 'warp_strength', 'warp_freq', 'twist_amount',
                     'rep_enabled', 'rep_cell_x', 'rep_cell_y', 'rep_cell_z',
                     'fold_mirror_x', 'fold_mirror_y', 'fold_mirror_z'],
    'Mandelbox':    ['mb_fold_limit', 'mb_sphere_inner', 'mb_sphere_outer',
                     'mb_fixed_radius', 'mb_rot_per_iter'],
    'Menger':       ['ms_scale', 'ms_offset', 'ms_twist', 'ms_cross_width'],
    'Sierpinski':   ['si_vertex_spread', 'si_fold_bias', 'si_squash'],
    'Octahedron':   ['oc_ifs_scale', 'oc_twist', 'oc_fold_amount'],
    'Mandelbulb':   ['mb2_power', 'mb2_bailout', 'mb2_fold_strength'],
    'Kleinian':     ['kl_scale', 'kl_fold_limit', 'kl_sph_radius', 'kl_mix_factor'],
    'Animation':    ['animate', 'anim_speed'],
}

KEY_R = 6


def _lerp(a, b, t):
    return a + (b - a) * t

def _smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)

def _smootherstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

def _ease_in(t):
    t = max(0.0, min(1.0, t))
    return t * t * t

def _ease_out(t):
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** 3

def _bounce_out(t):
    t = max(0.0, min(1.0, t))
    if t < 1.0 / 2.75:
        return 7.5625 * t * t
    elif t < 2.0 / 2.75:
        t -= 1.5 / 2.75
        return 7.5625 * t * t + 0.75
    elif t < 2.5 / 2.75:
        t -= 2.25 / 2.75
        return 7.5625 * t * t + 0.9375
    else:
        t -= 2.625 / 2.75
        return 7.5625 * t * t + 0.984375

def _elastic_out(t):
    t = max(0.0, min(1.0, t))
    if t in (0.0, 1.0):
        return t
    return (2.0 ** (-10.0 * t)) * math.sin((t * 10.0 - 0.75) * (2.0 * math.pi / 3.0)) + 1.0

def _back_out(t):
    t = max(0.0, min(1.0, t))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (t - 1.0) ** 3 + c1 * (t - 1.0) ** 2

def _expo_out(t):
    t = max(0.0, min(1.0, t))
    return 1.0 if t >= 1.0 else 1.0 - 2.0 ** (-10.0 * t)

_EASING_FUNCS = {
    INTERP_EASE_IN:  _ease_in,
    INTERP_EASE_OUT: _ease_out,
    INTERP_EASE_IO:  _smootherstep,
    INTERP_BOUNCE:   _bounce_out,
    INTERP_ELASTIC:  _elastic_out,
    INTERP_BACK:     _back_out,
    INTERP_EXPO:     _expo_out,
}

for _k, _fn in _EASING_FUNCS.items():
    _lbl, _col, _ = INTERP_TABLE[_k]
    INTERP_TABLE[_k] = (_lbl, _col, _fn)

_EASING_FN = {k: v[2] for k, v in INTERP_TABLE.items() if v[2] is not None}

def _catmull_rom_timed(p0, p1, p2, p3, t0, t1, t2, t3, t):
    dt12 = max(t2 - t1, 1e-9)
    tan1 = (p2 - p0) / max(t2 - t0, 1e-9)
    tan2 = (p3 - p1) / max(t3 - t1, 1e-9)
    s    = (t - t1) / dt12
    h00  =  2*s**3 - 3*s**2 + 1
    h10  =    s**3 - 2*s**2 + s
    h01  = -2*s**3 + 3*s**2
    h11  =    s**3 -   s**2
    return h00*p1 + h10*dt12*tan1 + h01*p2 + h11*dt12*tan2

def _hermite(p0, p1, m0, m1, t):
    h00 =  2*t**3 - 3*t**2 + 1
    h10 =    t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2
    h11 =    t**3 -   t**2
    return h00*p0 + h10*m0 + h01*p1 + h11*m1



def _interp_value(keys, times, time_sec, ptype):
    n = len(keys)
    if not n:
        return None
    if n == 1:
        return keys[0].value
    idx1 = _bisect.bisect_right(times, time_sec)
    if idx1 == 0:
        return keys[0].value
    if idx1 == n:
        return keys[-1].value
    idx0   = idx1 - 1
    k0     = keys[idx0]
    k1     = keys[idx1]
    interp = k0.interp
    dt     = k1.time - k0.time
    if dt < 1e-9:
        return k1.value
    t = (time_sec - k0.time) / dt
    if ptype in (PROP_BOOL, PROP_INT) or interp == INTERP_NONE:
        return k0.value
    easing_fn = _EASING_FN.get(interp)
    if easing_fn is not None:
        t_eased = easing_fn(t)
        if ptype == PROP_COLOR:
            v0, v1 = k0.value, k1.value
            return (v0[0] + (v1[0] - v0[0]) * t_eased,
                    v0[1] + (v1[1] - v0[1]) * t_eased,
                    v0[2] + (v1[2] - v0[2]) * t_eased)
        return _lerp(k0.value, k1.value, t_eased)
    if ptype == PROP_COLOR:
        if interp == INTERP_CUBIC:
            t = _smoothstep(t)
        v0, v1 = k0.value, k1.value
        return (v0[0] + (v1[0] - v0[0]) * t,
                v0[1] + (v1[1] - v0[1]) * t,
                v0[2] + (v1[2] - v0[2]) * t)
    if interp == INTERP_CUBIC:
        if idx0 > 0:
            km1  = keys[idx0 - 1]
            t0_g = km1.time
            p0v  = km1.value
        else:
            t0_g = k0.time - (k1.time - k0.time)
            p0v  = k0.value + (k0.value - k1.value)
        if idx1 < n - 1:
            kp1  = keys[idx1 + 1]
            t3_g = kp1.time
            p3v  = kp1.value
        else:
            t3_g = k1.time + (k1.time - k0.time)
            p3v  = k1.value + (k1.value - k0.value)
        return _catmull_rom_timed(p0v, k0.value, k1.value, p3v,
                                  t0_g, k0.time, k1.time, t3_g,
                                  time_sec)
    return _lerp(k0.value, k1.value, t)


class Keyframe:
    __slots__ = ('time', 'value', 'interp')
    def __init__(self, time, value, interp=INTERP_CUBIC):
        self.time   = float(time)
        self.value  = value
        self.interp = interp
    def to_list(self):
        return [self.time, self.value, self.interp]
    @classmethod
    def from_list(cls, lst):
        return cls(lst[0], lst[1], lst[2] if len(lst) > 2 else INTERP_CUBIC)


class AnimTrack:
    def __init__(self, prop):
        self.prop    = prop
        self.keys    = []
        self.enabled = True
        self._ptype  = ALL_PROPERTIES.get(prop, (PROP_FLOAT,))[0]
        self._times  = []

    def _rebuild_times(self):
        self._times = [k.time for k in self.keys]

    def add_key(self, t, value, interp=INTERP_CUBIC):
        for k in self.keys:
            if abs(k.time - t) < 1e-6:
                k.value  = value
                k.interp = interp
                return k
        kf = Keyframe(t, value, interp)
        self.keys.append(kf)
        self.keys.sort(key=lambda k: k.time)
        self._rebuild_times()
        return kf

    def remove_key(self, kf):
        if kf in self.keys:
            self.keys.remove(kf)
            self._rebuild_times()

    def evaluate(self, t):
        return _interp_value(self.keys, self._times, t, self._ptype)

    def to_dict(self):
        return {'prop': self.prop, 'enabled': self.enabled,
                'keys': [k.to_list() for k in self.keys]}

    @classmethod
    def from_dict(cls, d):
        tr = cls(d['prop'])
        tr.enabled = d.get('enabled', True)
        tr.keys    = [Keyframe.from_list(lst) for lst in d.get('keys', [])]
        tr._rebuild_times()
        return tr


class AnimClip:
    def __init__(self, name='Clip', duration=10.0):
        self.name     = name
        self.duration = float(duration)
        self.tracks   = []
        self.loop     = False
    def get_track(self, prop):
        for t in self.tracks:
            if t.prop == prop:
                return t
        return None
    def get_or_create_track(self, prop):
        t = self.get_track(prop)
        if t is None:
            t = AnimTrack(prop)
            self.tracks.append(t)
        return t
    def remove_track(self, prop):
        self.tracks = [t for t in self.tracks if t.prop != prop]
    def evaluate_all(self, time_sec):
        out = {}
        for track in self.tracks:
            if track.enabled and track.keys:
                out[track.prop] = track.evaluate(time_sec)
        return out
    def to_dict(self):
        return {'name': self.name, 'duration': self.duration,
                'loop': self.loop, 'tracks': [t.to_dict() for t in self.tracks]}
    @classmethod
    def from_dict(cls, d):
        c      = cls(d.get('name', 'Clip'), d.get('duration', 10.0))
        c.loop = d.get('loop', False)
        c.tracks = [AnimTrack.from_dict(td) for td in d.get('tracks', [])]
        return c


class AnimationState(QObject):
    changed          = pyqtSignal()
    time_changed     = pyqtSignal(float)
    clip_changed     = pyqtSignal()
    playback_changed = pyqtSignal(bool)
    TICK_MS = 16

    def __init__(self):
        super().__init__()
        self.clips      = []
        self._cur_idx   = -1
        self._time      = 0.0
        self._playing   = False
        self._speed     = 1.0
        self._last_wall = 0.0
        self._apply_cb  = None
        self._timer     = QTimer(self)
        self._timer.setInterval(self.TICK_MS)
        self._timer.timeout.connect(self._tick)

    def set_apply_callback(self, cb):
        self._apply_cb = cb

    @property
    def current_clip(self):
        if 0 <= self._cur_idx < len(self.clips):
            return self.clips[self._cur_idx]
        return None

    @property
    def time(self):
        return self._time

    @property
    def playing(self):
        return self._playing

    def set_speed(self, v):
        self._speed = max(0.01, float(v))

    def set_time(self, t):
        clip = self.current_clip
        if clip:
            t = max(0.0, min(t, clip.duration))
        self._time = t
        self.time_changed.emit(self._time)
        self._apply_frame()

    def select_clip(self, idx):
        self._cur_idx = idx
        self._time    = 0.0
        self.clip_changed.emit()
        self.changed.emit()

    def add_clip(self, name='New Clip'):
        c = AnimClip(name)
        self.clips.append(c)
        if self._cur_idx < 0:
            self._cur_idx = 0
        self.clip_changed.emit()
        self.changed.emit()
        return c

    def remove_clip(self, idx):
        if 0 <= idx < len(self.clips):
            self.clips.pop(idx)
            self._cur_idx = min(self._cur_idx, len(self.clips) - 1)
            self.clip_changed.emit()
            self.changed.emit()

    def duplicate_clip(self, idx):
        if 0 <= idx < len(self.clips):
            import copy
            c      = copy.deepcopy(self.clips[idx])
            c.name = c.name + ' (copy)'
            self.clips.insert(idx + 1, c)
            self.clip_changed.emit()
            self.changed.emit()

    def play(self):
        if not self._playing and self.current_clip:
            self._playing   = True
            self._last_wall = time.monotonic()
            self._timer.start()
            self.playback_changed.emit(True)

    def pause(self):
        if self._playing:
            self._playing = False
            self._timer.stop()
            self.playback_changed.emit(False)

    def toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def stop(self):
        self.pause()
        self.set_time(0.0)

    def _tick(self):
        now            = time.monotonic()
        dt             = (now - self._last_wall) * self._speed
        self._last_wall = now
        clip = self.current_clip
        if not clip:
            return
        self._time += dt
        if self._time >= clip.duration:
            if clip.loop:
                self._time = math.fmod(self._time, max(clip.duration, 1e-9))
            else:
                self._time = clip.duration
                self.pause()
        self.time_changed.emit(self._time)
        self._apply_frame()

    def _apply_frame(self):
        clip = self.current_clip
        if clip and self._apply_cb:
            try:
                self._apply_cb(clip.evaluate_all(self._time))
            except Exception:
                pass

    def to_dict(self):
        return {'version': ANIM_VERSION, 'current': self._cur_idx,
                'clips': [c.to_dict() for c in self.clips]}

    def from_dict(self, d):
        self.pause()
        self.clips    = [AnimClip.from_dict(cd) for cd in d.get('clips', [])]
        self._cur_idx = d.get('current', 0 if self.clips else -1)
        self._time    = 0.0
        self.clip_changed.emit()
        self.changed.emit()

    def save(self, path):
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding='utf-8')

    def load(self, path):
        self.from_dict(json.loads(Path(path).read_text(encoding='utf-8')))


_anim_state = AnimationState()


class UndoCommand:
    def redo(self): pass
    def undo(self): pass
    def merge(self, other): return False

class AddKeyCommand(UndoCommand):
    def __init__(self, track, t, value, interp=INTERP_CUBIC):
        self._track = track
        self._t     = t
        self._value = value
        self._interp= interp
        self._kf    = None
    def redo(self):
        self._kf = self._track.add_key(self._t, self._value, self._interp)
    def undo(self):
        if self._kf in self._track.keys:
            self._track.remove_key(self._kf)

class RemoveKeysCommand(UndoCommand):
    def __init__(self, removals):
        self._removals = [(tr, kf, kf.time, kf.value, kf.interp) for tr, kf in removals]
    def redo(self):
        for tr, kf, *_ in self._removals:
            tr.remove_key(kf)
    def undo(self):
        for tr, _, t, v, interp in self._removals:
            tr.add_key(t, v, interp)

class MoveKeysCommand(UndoCommand):
    def __init__(self, moves):
        self._moves = [(tr, kf, old_t, new_t) for tr, kf, old_t, new_t in moves]
    def redo(self):
        dirty = set()
        for tr, kf, _, new_t in self._moves:
            kf.time = new_t
            dirty.add(tr)
        for tr in dirty:
            tr.keys.sort(key=lambda k: k.time)
            tr._rebuild_times()
    def undo(self):
        dirty = set()
        for tr, kf, old_t, _ in self._moves:
            kf.time = old_t
            dirty.add(tr)
        for tr in dirty:
            tr.keys.sort(key=lambda k: k.time)
            tr._rebuild_times()

class SetInterpCommand(UndoCommand):
    def __init__(self, keys_modes, new_mode):
        self._entries  = [(kf, kf.interp, new_mode) for kf in keys_modes]
    def redo(self):
        for kf, _, new in self._entries:
            kf.interp = new
    def undo(self):
        for kf, old, _ in self._entries:
            kf.interp = old

class EditKeyCommand(UndoCommand):
    def __init__(self, track, kf, old_t, old_v, old_i, new_t, new_v, new_i):
        self._track = track
        self._kf    = kf
        self._old   = (old_t, old_v, old_i)
        self._new   = (new_t, new_v, new_i)
    def redo(self):
        self._kf.time, self._kf.value, self._kf.interp = self._new
        self._track.keys.sort(key=lambda k: k.time)
        self._track._rebuild_times()
    def undo(self):
        self._kf.time, self._kf.value, self._kf.interp = self._old
        self._track.keys.sort(key=lambda k: k.time)
        self._track._rebuild_times()


class UndoStack(QObject):
    changed = pyqtSignal()
    def __init__(self, max_depth=100):
        super().__init__()
        self._stack   = []
        self._pos     = -1
        self._max     = max_depth
    def push(self, cmd):
        self._stack = self._stack[:self._pos + 1]
        cmd.redo()
        self._stack.append(cmd)
        if len(self._stack) > self._max:
            self._stack.pop(0)
        self._pos = len(self._stack) - 1
        self.changed.emit()
    def undo(self):
        if self._pos >= 0:
            self._stack[self._pos].undo()
            self._pos -= 1
            self.changed.emit()
    def redo(self):
        if self._pos < len(self._stack) - 1:
            self._pos += 1
            self._stack[self._pos].redo()
            self.changed.emit()
    def can_undo(self): return self._pos >= 0
    def can_redo(self): return self._pos < len(self._stack) - 1
    def clear(self):
        self._stack = []
        self._pos   = -1
        self.changed.emit()


_undo_stack = UndoStack()


def _icon_btn(std_icon, tooltip=''):
    btn = QPushButton()
    btn.setIcon(QApplication.style().standardIcon(std_icon))
    btn.setToolTip(tooltip)
    return btn


class SliderSpinBox(QWidget):
    valueChanged = pyqtSignal(float)
    STEPS = 1000
    def __init__(self, lo, hi, decimals=3, parent=None):
        super().__init__(parent)
        self._lo       = float(lo)
        self._hi       = float(hi)
        self._decimals = decimals
        self._updating = False
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, self.STEPS)
        self._spin = QDoubleSpinBox()
        self._spin.setRange(lo, hi)
        self._spin.setDecimals(decimals)
        self._spin.setSingleStep((hi - lo) / 100.0)
        row.addWidget(self._slider, 1)
        row.addWidget(self._spin, 0)
        self._slider.valueChanged.connect(self._slider_changed)
        self._spin.valueChanged.connect(self._spin_changed)
    def _slider_changed(self, v):
        if self._updating:
            return
        val = self._lo + (self._hi - self._lo) * v / self.STEPS
        self._updating = True
        self._spin.setValue(val)
        self._updating = False
        self.valueChanged.emit(val)
    def _spin_changed(self, v):
        if self._updating:
            return
        sv = int(round((v - self._lo) / (self._hi - self._lo) * self.STEPS))
        self._updating = True
        self._slider.setValue(max(0, min(self.STEPS, sv)))
        self._updating = False
        self.valueChanged.emit(v)
    def value(self):
        return self._spin.value()
    def setValue(self, v):
        self._updating = True
        self._spin.setValue(v)
        sv = int(round((v - self._lo) / (self._hi - self._lo) * self.STEPS))
        self._slider.setValue(max(0, min(self.STEPS, sv)))
        self._updating = False


class ClipListPanel(QWidget):
    clip_selected = pyqtSignal(int)
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self._build()
        state.clip_changed.connect(self._refresh)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        hdr = QLabel("CLIPS")
        root.addWidget(hdr)
        self._list = QListWidget()
        self._list.currentRowChanged.connect(self._on_select)
        self._list.setContextMenuPolicy(Qt.CustomContextMenu)
        self._list.customContextMenuRequested.connect(self._ctx_menu)
        root.addWidget(self._list)
        btn_row = QHBoxLayout()
        btn_row.setSpacing(3)
        for icon, tip, slot in [
            (QStyle.SP_FileDialogNewFolder,    'Add Clip',       self._add_clip),
            (QStyle.SP_FileDialogContentsView, 'Duplicate Clip', self._dup_clip),
            (QStyle.SP_TrashIcon,              'Delete Clip',    self._del_clip),
        ]:
            b = _icon_btn(icon, tip)
            b.clicked.connect(slot)
            btn_row.addWidget(b)
        btn_row.addStretch()
        root.addLayout(btn_row)

    def _refresh(self):
        self._list.blockSignals(True)
        self._list.clear()
        for c in self._state.clips:
            item = QListWidgetItem(c.name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self._list.addItem(item)
        idx = self._state._cur_idx
        if 0 <= idx < self._list.count():
            self._list.setCurrentRow(idx)
        self._list.blockSignals(False)

    def _on_select(self, row):
        if row >= 0:
            self._state.select_clip(row)
            self.clip_selected.emit(row)

    def _ctx_menu(self, pos):
        menu = QMenu(self)
        row = self._list.currentRow()
        actions = [
            (QStyle.SP_FileDialogNewFolder,    'Add Clip',  self._add_clip),
            (QStyle.SP_FileDialogContentsView, 'Duplicate', self._dup_clip),
            (QStyle.SP_FileDialogDetailedView, 'Rename',    lambda: self._rename_clip(row)),
            (None, None, None),
            (QStyle.SP_TrashIcon,              'Delete',    self._del_clip),
        ]
        for icon_id, label, slot in actions:
            if icon_id is None:
                menu.addSeparator()
            else:
                a = menu.addAction(QApplication.style().standardIcon(icon_id), label)
                a.triggered.connect(slot)
        menu.exec_(self._list.mapToGlobal(pos))

    def _add_clip(self):
        name, ok = QInputDialog.getText(self, 'Add Clip', 'Clip name:', text='New Clip')
        if ok and name:
            self._state.add_clip(name)

    def _dup_clip(self):
        row = self._list.currentRow()
        if row >= 0:
            self._state.duplicate_clip(row)

    def _del_clip(self):
        row = self._list.currentRow()
        if row >= 0:
            clip = self._state.clips[row]
            if QMessageBox.question(self, 'Delete', f'Delete clip "{clip.name}"?',
                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
                self._state.remove_clip(row)

    def _rename_clip(self, row):
        if 0 <= row < len(self._state.clips):
            clip = self._state.clips[row]
            name, ok = QInputDialog.getText(self, 'Rename Clip', 'New name:', text=clip.name)
            if ok and name:
                clip.name = name
                self._state.changed.emit()
                self._refresh()


class PropSelectorDialog(QDialog):
    def __init__(self, existing, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Add Property Track')
        self._existing = set(existing)
        root = QVBoxLayout(self)
        root.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText('Search properties...')
        self._search.textChanged.connect(self._filter)
        root.addWidget(self._search)
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(['Property', 'Type'])
        self._tree.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self._tree.header().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        self._tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        root.addWidget(self._tree)
        self._populate('')
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    def _populate(self, filt):
        self._tree.clear()
        filt = filt.lower()
        for group, props in PROP_GROUPS.items():
            g_item = QTreeWidgetItem([group, ''])
            added = 0
            for p in props:
                if p in self._existing or p not in ALL_PROPERTIES:
                    continue
                if filt and filt not in p.lower() and filt not in group.lower():
                    continue
                ptype = ALL_PROPERTIES[p][0]
                child = QTreeWidgetItem([p, ptype])
                child.setData(0, Qt.UserRole, p)
                g_item.addChild(child)
                added += 1
            if added > 0:
                self._tree.addTopLevelItem(g_item)
                g_item.setExpanded(True)

    def _filter(self, text):
        self._populate(text)

    def selected_props(self):
        result = []
        for item in self._tree.selectedItems():
            p = item.data(0, Qt.UserRole)
            if p:
                result.append(p)
        return result


class TimeRuler(QWidget):
    seek_requested = pyqtSignal(float)
    RULER_H = 24
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(self.RULER_H)
        self._time       = 0.0
        self._view_start = 0.0
        self._view_end   = 10.0
        self._dragging   = False

    def set_duration(self, d):
        self.update()

    def set_time(self, t):
        self._time = t
        self.update()

    def set_view(self, start, end):
        self._view_start = start
        self._view_end   = end
        self.update()

    def _t_to_x(self, t):
        span = self._view_end - self._view_start
        if span < 1e-9:
            return 0.0
        return (t - self._view_start) / span * self.width()

    def _x_to_t(self, x):
        span = self._view_end - self._view_start
        return self._view_start + x / max(self.width(), 1) * span

    def paintEvent(self, _):
        p = QPainter(self)
        pal   = self.palette()
        dark  = _is_dark_theme()
        c_bg  = QColor(COLORS['bg3'])  if dark else pal.color(pal.Window)
        c_bdr = QColor(COLORS['border']) if dark else pal.color(pal.Mid)
        c_fg  = QColor(COLORS['fg3']) if dark else pal.color(pal.Text)
        c_dim = QColor(COLORS['fg4']) if dark else pal.color(pal.Midlight)
        c_ph  = QColor(COLORS['playhead'])
        p.fillRect(self.rect(), c_bg)
        p.setPen(QPen(c_bdr, 1))
        p.drawLine(0, self.RULER_H - 1, self.width(), self.RULER_H - 1)
        span = self._view_end - self._view_start
        if span > 0:
            step = self._nice_step(span)
            t    = math.floor(self._view_start / step) * step
            while t <= self._view_end + step:
                x        = int(self._t_to_x(t))
                is_major = abs(round(t / step) * step - t) < step * 0.01
                if is_major:
                    p.setPen(QPen(c_fg, 1))
                    p.drawLine(x, self.RULER_H - 10, x, self.RULER_H - 1)
                    p.setFont(_canvas_font(8))
                    lbl = f'{t:.1f}s' if step < 1.0 else f'{int(t)}s'
                    p.drawText(x + 2, self.RULER_H - 10, lbl)
                else:
                    p.setPen(QPen(c_dim, 1))
                    p.drawLine(x, self.RULER_H - 5, x, self.RULER_H - 1)
                t += step
        ph_x = int(self._t_to_x(self._time))
        p.setPen(QPen(c_ph, 2))
        p.drawLine(ph_x, 0, ph_x, self.RULER_H)
        p.end()

    def _nice_step(self, span):
        raw = span / 10.0
        if raw <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(raw))
        for mult in (1, 2, 5, 10):
            s = mag * mult
            if span / s <= 12:
                return s
        return mag * 10

    def mousePressEvent(self, ev):
        if ev.button() == Qt.LeftButton:
            self._dragging = True
            self.seek_requested.emit(max(0.0, self._x_to_t(ev.x())))

    def mouseMoveEvent(self, ev):
        if self._dragging:
            self.seek_requested.emit(max(0.0, self._x_to_t(ev.x())))

    def mouseReleaseEvent(self, _):
        self._dragging = False


class TrackWidget(QWidget):
    key_moved        = pyqtSignal(str, float, float)
    key_added        = pyqtSignal(str, float)
    seek_requested   = pyqtSignal(float)
    track_selected   = pyqtSignal(object)
    TRACK_H = 30
    LABEL_W = 200

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state           = state
        self._view_start      = 0.0
        self._view_end        = 10.0
        self._selected_keys   = set()
        self._selected_track  = None
        self._drag_key        = None
        self._drag_starts     = {}
        self._drag_mouse_x    = 0
        self._hover_key       = None
        self._drag_seek       = False
        self._box_select      = False
        self._box_start       = None
        self._box_end         = None
        self._mmb_pan_active  = False
        self._mmb_pan_x       = 0
        self._mmb_pan_vs      = 0.0
        self._mmb_pan_ve      = 0.0
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.setMinimumHeight(self.TRACK_H)
        state.time_changed.connect(self._on_time_changed)
        state.changed.connect(self._on_data_changed)

    def _on_time_changed(self, _):
        self.update()

    def _on_data_changed(self):
        n      = len(self._tracks())
        needed = max(self.TRACK_H, n * self.TRACK_H)
        if self.minimumHeight() != needed:
            self.setMinimumHeight(needed)
        self.update()

    def set_view(self, start, end):
        self._view_start = start
        self._view_end   = end
        self.update()

    def _clip(self):
        return self._state.current_clip

    def _tracks(self):
        c = self._clip()
        return c.tracks if c else []

    def _t_to_x(self, t):
        span = self._view_end - self._view_start
        if span < 1e-9:
            return float(self.LABEL_W)
        w = max(self.width() - self.LABEL_W, 1)
        return self.LABEL_W + (t - self._view_start) / span * w

    def _x_to_t(self, x):
        span = self._view_end - self._view_start
        w    = max(self.width() - self.LABEL_W, 1)
        return self._view_start + (x - self.LABEL_W) / w * span

    def _key_at(self, pos):
        for ti, track in enumerate(self._tracks()):
            cy = ti * self.TRACK_H + self.TRACK_H // 2
            if abs(pos.y() - cy) > self.TRACK_H // 2:
                continue
            for kf in track.keys:
                kx = self._t_to_x(kf.time)
                if abs(pos.x() - kx) <= KEY_R + 2:
                    return ti, track, kf
        return None, None, None

    def _keys_in_box(self, r):
        result = []
        for ti, track in enumerate(self._tracks()):
            cy = ti * self.TRACK_H + self.TRACK_H // 2
            if cy + KEY_R < r.top() or cy - KEY_R > r.bottom():
                continue
            for kf in track.keys:
                kx = self._t_to_x(kf.time)
                if r.left() <= kx <= r.right():
                    result.append((ti, track, kf))
        return result

    def paintEvent(self, _):
        p      = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        clip   = self._clip()
        tracks = self._tracks()
        h      = max(self.TRACK_H, len(tracks) * self.TRACK_H, self.height())
        dark   = _is_dark_theme()
        pal    = self.palette()
        c_bg0      = QColor(COLORS['track'])   if dark else pal.color(pal.Base)
        c_bg1      = QColor(COLORS['track2'])  if dark else pal.color(pal.AlternateBase)
        c_sel_bg   = QColor(COLORS['sel']).lighter(60) if dark else pal.color(pal.Highlight).lighter(170)
        c_lbl_bg   = QColor(COLORS['panel'])   if dark else pal.color(pal.Window)
        c_lbl_sel  = QColor(COLORS['panel2'])  if dark else pal.color(pal.Highlight).lighter(150)
        c_border   = QColor(COLORS['border'])  if dark else pal.color(pal.Mid)
        c_accent   = QColor(COLORS['accent'])  if dark else pal.color(pal.Highlight)
        c_fg       = QColor(COLORS['fg'])      if dark else pal.color(pal.Text)
        c_fg2      = QColor(COLORS['fg2'])     if dark else pal.color(pal.WindowText)
        c_fg4      = QColor(COLORS['fg4'])     if dark else pal.color(pal.Midlight)
        c_key_bg   = QColor(COLORS['bg'])      if dark else pal.color(pal.Base)
        c_ph       = QColor(COLORS['playhead'])
        for i, track in enumerate(tracks):
            ry       = i * self.TRACK_H
            is_sel   = (track is self._selected_track)
            bg_color = c_sel_bg if is_sel else (c_bg0 if i % 2 == 0 else c_bg1)
            p.fillRect(QRect(0, ry, self.width(), self.TRACK_H), bg_color)
            lbl_bg = c_lbl_sel if is_sel else c_lbl_bg
            p.fillRect(QRect(0, ry, self.LABEL_W, self.TRACK_H), lbl_bg)
            if is_sel:
                p.setPen(QPen(c_accent, 1))
                p.drawRect(QRect(0, ry, self.LABEL_W - 1, self.TRACK_H - 1))
            p.setPen(QPen(c_border, 1))
            p.drawLine(0, ry + self.TRACK_H - 1, self.width(), ry + self.TRACK_H - 1)
            p.drawLine(self.LABEL_W, ry, self.LABEL_W, ry + self.TRACK_H)
            p.setPen(c_fg if is_sel else (c_fg2 if track.enabled else c_fg4))
            p.setFont(_canvas_font(9))
            lbl_r = QRect(8, ry, self.LABEL_W - 12, self.TRACK_H)
            p.drawText(lbl_r, Qt.AlignVCenter | Qt.AlignLeft, track.prop)
            p.setPen(c_fg4)
            p.setFont(_canvas_font(8))
            ptype = ALL_PROPERTIES.get(track.prop, (PROP_FLOAT,))[0]
            p.drawText(QRect(0, ry, self.LABEL_W - 6, self.TRACK_H), Qt.AlignVCenter | Qt.AlignRight, ptype)
            if not track.enabled:
                continue
            cy = ry + self.TRACK_H // 2
            if len(track.keys) > 1:
                p.setPen(QPen(c_border, 1, Qt.DotLine))
                x0 = max(self._t_to_x(track.keys[0].time), self.LABEL_W)
                x1 = self._t_to_x(track.keys[-1].time)
                p.drawLine(int(x0), cy, int(x1), cy)
            for kf in track.keys:
                cx = int(self._t_to_x(kf.time))
                if cx < self.LABEL_W - KEY_R or cx > self.width() + KEY_R:
                    continue
                icolor = QColor(INTERP_COLORS.get(kf.interp, COLORS['key_free']))
                sel    = (i, id(kf)) in self._selected_keys
                hover  = self._hover_key == (i, id(kf))
                path   = QPainterPath()
                path.moveTo(cx,          cy - KEY_R)
                path.lineTo(cx + KEY_R,  cy)
                path.lineTo(cx,          cy + KEY_R)
                path.lineTo(cx - KEY_R,  cy)
                path.closeSubpath()
                if sel:
                    p.setBrush(QBrush(QColor(COLORS['accent3'])))
                    p.setPen(QPen(QColor('#ffffff'), 1.5))
                elif hover:
                    p.setBrush(QBrush(icolor.lighter(130)))
                    p.setPen(QPen(QColor('#ffffff'), 1))
                else:
                    p.setBrush(QBrush(icolor))
                    p.setPen(QPen(c_key_bg, 1))
                p.drawPath(path)
        if not tracks:
            p.setPen(c_fg4)
            p.setFont(_canvas_font(9))
            p.drawText(self.rect(), Qt.AlignCenter, 'No tracks — click "+ Add Track" to begin')
        if clip:
            ph_x = int(self._t_to_x(self._state.time))
            p.setPen(QPen(c_ph, 1))
            p.drawLine(ph_x, 0, ph_x, h)
        if self._box_select and self._box_start and self._box_end:
            r = QRectF(self._box_start, self._box_end).normalized()
            p.setBrush(QBrush(QColor(COLORS['accent'] + '44')))
            p.setPen(QPen(QColor(COLORS['accent']), 1, Qt.DashLine))
            p.drawRect(r)
        p.end()

    def _track_at_row(self, y):
        row    = int(y) // self.TRACK_H
        tracks = self._tracks()
        if 0 <= row < len(tracks):
            return row, tracks[row]
        return None, None

    def _select_track(self, track):
        if self._selected_track is not track:
            self._selected_track = track
            self.track_selected.emit(track)
            self.update()

    def _selected_key_pairs(self):
        result = []
        for ti, track in enumerate(self._tracks()):
            for kf in track.keys:
                if (ti, id(kf)) in self._selected_keys:
                    result.append((track, kf))
        return result

    def mousePressEvent(self, ev):
        self.setFocus()
        pos = QPointF(ev.pos())
        ti, track, kf = self._key_at(pos)
        if ev.button() == Qt.LeftButton:
            if ev.x() <= self.LABEL_W:
                row_i, row_track = self._track_at_row(ev.y())
                if row_track is not None:
                    self._select_track(row_track)
                return
            if kf is not None:
                key_id = (ti, id(kf))
                if ev.modifiers() & Qt.ControlModifier:
                    if key_id in self._selected_keys:
                        self._selected_keys.discard(key_id)
                    else:
                        self._selected_keys.add(key_id)
                elif ev.modifiers() & Qt.ShiftModifier:
                    self._selected_keys.add(key_id)
                else:
                    if key_id not in self._selected_keys:
                        self._selected_keys = {key_id}
                self._select_track(track)
                self._drag_key     = True
                self._drag_starts  = {id(k): k.time for _, k in self._selected_key_pairs()}
                self._drag_mouse_x = ev.x()
                self.update()
            else:
                self._selected_keys.clear()
                self._box_select  = True
                self._box_start   = pos
                self._box_end     = pos
                self._drag_seek   = False
                self.update()
        elif ev.button() == Qt.MiddleButton:
            self._mmb_pan_active = True
            self._mmb_pan_x      = ev.x()
            self._mmb_pan_vs     = self._view_start
            self._mmb_pan_ve     = self._view_end
            self.setCursor(Qt.SizeHorCursor)
        elif ev.button() == Qt.RightButton:
            if ti is None:
                ti, track = self._track_at_row(ev.y())
            self._ctx_menu(ev.pos(), ti, track, kf)

    def mouseMoveEvent(self, ev):
        pos = QPointF(ev.pos())
        ti, _, kf = self._key_at(pos)
        new_hover = (ti, id(kf)) if kf is not None else None
        if new_hover != self._hover_key:
            self._hover_key = new_hover
            self.update()
        if self._drag_key and (ev.buttons() & Qt.LeftButton):
            dx   = ev.x() - self._drag_mouse_x
            span = self._view_end - self._view_start
            dt   = dx / max(self.width() - self.LABEL_W, 1) * span
            clip = self._clip()
            pairs = self._selected_key_pairs()
            for track, dkf in pairs:
                orig_t = self._drag_starts.get(id(dkf), dkf.time)
                new_t  = max(0.0, orig_t + dt)
                if clip:
                    new_t = min(new_t, clip.duration)
                dkf.time = new_t
            for track, _ in pairs:
                track.keys.sort(key=lambda k: k.time)
            self.update()
        elif self._box_select and (ev.buttons() & Qt.LeftButton):
            self._box_end = pos
            r = QRectF(self._box_start, self._box_end).normalized()
            self._selected_keys = {(ti2, id(kf2)) for ti2, _, kf2 in self._keys_in_box(r)}
            self.update()
        elif self._drag_seek and (ev.buttons() & (Qt.LeftButton | Qt.MiddleButton)):
            self.seek_requested.emit(max(0.0, self._x_to_t(ev.x())))
        if self._mmb_pan_active and (ev.buttons() & Qt.MiddleButton):
            dx   = ev.x() - self._mmb_pan_x
            span = self._mmb_pan_ve - self._mmb_pan_vs
            dt   = -dx / max(self.width() - self.LABEL_W, 1) * span
            new_vs = max(0.0, self._mmb_pan_vs + dt)
            new_ve = new_vs + span
            self._view_start = new_vs
            self._view_end   = new_ve
            parent = self.parent()
            while parent:
                if hasattr(parent, '_sync_view'):
                    parent._sync_view()
                    break
                parent = parent.parent() if hasattr(parent, 'parent') else None
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag_key:
            pairs  = self._selected_key_pairs()
            moves  = []
            for track, kf in pairs:
                orig_t = self._drag_starts.get(id(kf), kf.time)
                if abs(orig_t - kf.time) > 1e-9:
                    moves.append((track, kf, orig_t, kf.time))
                    kf.time = orig_t
            if moves:
                _undo_stack.push(MoveKeysCommand(moves))
            self._state.changed.emit()
        if self._box_select:
            self._box_select = False
            self._box_start  = None
            self._box_end    = None
        self._drag_key  = False
        self._drag_seek = False
        if self._mmb_pan_active:
            self._mmb_pan_active = False
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def _track_index_of(self, track):
        for i, t in enumerate(self._tracks()):
            if t is track:
                return i
        return -1

    def mouseDoubleClickEvent(self, ev):
        if ev.x() <= self.LABEL_W:
            return
        pos = QPointF(ev.pos())
        ti, track, kf = self._key_at(pos)
        if kf is not None:
            self._edit_key_dialog(track, kf)
        else:
            row    = int(ev.y() // self.TRACK_H)
            tracks = self._tracks()
            if 0 <= row < len(tracks):
                self.key_added.emit(tracks[row].prop, max(0.0, self._x_to_t(ev.x())))

    def keyPressEvent(self, ev):
        key  = ev.key()
        mods = ev.modifiers()
        if key == Qt.Key_Delete or key == Qt.Key_Backspace:
            self._delete_selected()
        elif key == Qt.Key_A and not mods:
            self._select_all_keys()
        elif key == Qt.Key_I and not mods:
            self._insert_key_on_selected_track()
        elif key == Qt.Key_F and not mods:
            parent = self.parent()
            while parent:
                if hasattr(parent, '_frame_all'):
                    parent._frame_all()
                    break
                parent = parent.parent() if hasattr(parent, 'parent') else None
        elif key == Qt.Key_Z and (mods & Qt.ControlModifier):
            if mods & Qt.ShiftModifier:
                _undo_stack.redo()
            else:
                _undo_stack.undo()
            self._state.changed.emit()
        elif key == Qt.Key_Y and (mods & Qt.ControlModifier):
            _undo_stack.redo()
            self._state.changed.emit()
        elif key == Qt.Key_Escape:
            self._selected_keys.clear()
            self.update()
        else:
            super().keyPressEvent(ev)

    def _select_all_keys(self):
        self._selected_keys = set()
        for ti, track in enumerate(self._tracks()):
            for kf in track.keys:
                self._selected_keys.add((ti, id(kf)))
        self.update()

    def _insert_key_on_selected_track(self):
        if self._selected_track:
            self.key_added.emit(self._selected_track.prop, self._state.time)

    def _delete_selected(self):
        pairs = self._selected_key_pairs()
        if not pairs:
            return
        cmd = RemoveKeysCommand(pairs)
        _undo_stack.push(cmd)
        self._selected_keys.clear()
        self._state.changed.emit()

    def _ctx_menu(self, pos, ti, track, kf):
        menu = QMenu(self)
        if kf is not None:
            sel_pairs = self._selected_key_pairs()
            if len(sel_pairs) > 1:
                a_del = menu.addAction(QApplication.style().standardIcon(QStyle.SP_TrashIcon),
                                       f'Delete {len(sel_pairs)} Keyframes')
                a_del.triggered.connect(self._delete_selected)
                menu.addSeparator()
                im = menu.addMenu('Set Interpolation (all selected)')
                for mode, label in INTERP_LABELS.items():
                    ac = im.addAction(label)
                    ac.triggered.connect(lambda _, m=mode: self._set_interp_selected(m))
            else:
                a = menu.addAction(QApplication.style().standardIcon(QStyle.SP_TrashIcon), 'Delete Keyframe')
                a.triggered.connect(lambda: self._delete_key_cmd(track, kf))
                menu.addSeparator()
                im = menu.addMenu('Interpolation')
                for mode, label in INTERP_LABELS.items():
                    ac = im.addAction(label)
                    ac.setCheckable(True)
                    ac.setChecked(kf.interp == mode)
                    ac.triggered.connect(lambda _, m=mode, k=kf: self._set_interp_cmd([k], m))
                a2 = menu.addAction(QApplication.style().standardIcon(QStyle.SP_FileDialogDetailedView), 'Edit Value...')
                a2.triggered.connect(lambda: self._edit_key_dialog(track, kf))
        else:
            px = int(pos.x())
            if px > self.LABEL_W:
                t      = max(0.0, self._x_to_t(px))
                tracks = self._tracks()
                if ti is not None and 0 <= ti < len(tracks):
                    trk = tracks[ti]
                    a = menu.addAction(
                        QApplication.style().standardIcon(QStyle.SP_FileDialogNewFolder),
                        f'Add Key at {t:.3f}s')
                    a.triggered.connect(lambda: self.key_added.emit(trk.prop, t))
        if track is not None:
            menu.addSeparator()
            lbl_tog  = 'Disable Track' if track.enabled else 'Enable Track'
            icon_tog = QStyle.SP_MediaStop if track.enabled else QStyle.SP_MediaPlay
            a3 = menu.addAction(QApplication.style().standardIcon(icon_tog), lbl_tog)
            a3.triggered.connect(lambda: self._toggle_track(track))
            a4 = menu.addAction(QApplication.style().standardIcon(QStyle.SP_TrashIcon), 'Delete Track')
            a4.triggered.connect(lambda: self._remove_track(track))
        if menu.actions():
            menu.exec_(self.mapToGlobal(pos))

    def _delete_key_cmd(self, track, kf):
        _undo_stack.push(RemoveKeysCommand([(track, kf)]))
        self._selected_keys.discard((self._track_index_of(track), id(kf)))
        self._state.changed.emit()

    def _set_interp_cmd(self, keys, mode):
        _undo_stack.push(SetInterpCommand(keys, mode))
        self._state.changed.emit()

    def _set_interp_selected(self, mode):
        keys = [kf for _, kf in self._selected_key_pairs()]
        if keys:
            _undo_stack.push(SetInterpCommand(keys, mode))
            self._state.changed.emit()

    def _delete_key(self, track, kf):
        self._delete_key_cmd(track, kf)

    def _set_interp(self, kf, mode):
        self._set_interp_cmd([kf], mode)

    def _toggle_track(self, track):
        track.enabled = not track.enabled
        self._state.changed.emit()

    def _remove_track(self, track):
        clip = self._clip()
        if clip:
            clip.remove_track(track.prop)
            if self._selected_track is track:
                self._selected_track = None
                self.track_selected.emit(None)
            self._state.changed.emit()

    def _edit_key_dialog(self, track, kf):
        ptype_info = ALL_PROPERTIES.get(track.prop, (PROP_FLOAT, 0.0, 1.0))
        dlg        = KeyEditDialog(track.prop, kf, ptype_info, self)
        if dlg.exec_() == QDialog.Accepted:
            cmd = EditKeyCommand(track, kf, kf.time, kf.value, kf.interp,
                                 dlg.time_val, dlg.prop_val, dlg.interp_val)
            _undo_stack.push(cmd)
            self._state.changed.emit()


class KeyEditDialog(QDialog):
    def __init__(self, prop, kf, ptype_info, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f'Edit Keyframe: {prop}')

        self.time_val   = kf.time
        self.prop_val   = kf.value
        self.interp_val = kf.interp
        ptype = ptype_info[0]
        root  = QVBoxLayout(self)
        root.setSpacing(8)
        grid  = QGridLayout()
        grid.setSpacing(6)
        grid.addWidget(QLabel('Time (s):'), 0, 0)
        self._t_spin = QDoubleSpinBox()
        self._t_spin.setRange(0.0, 9999.0)
        self._t_spin.setDecimals(4)
        self._t_spin.setValue(kf.time)
        grid.addWidget(self._t_spin, 0, 1)
        grid.addWidget(QLabel('Interpolation:'), 1, 0)
        self._interp_cb = QComboBox()
        for mode, label in INTERP_LABELS.items():
            self._interp_cb.addItem(label, mode)
        self._interp_cb.setCurrentIndex(kf.interp)
        grid.addWidget(self._interp_cb, 1, 1)
        self._ptype = ptype
        if ptype == PROP_FLOAT:
            lo, hi = ptype_info[1], ptype_info[2]
            grid.addWidget(QLabel('Value:'), 2, 0)
            self._v_spin = QDoubleSpinBox()
            self._v_spin.setRange(lo if lo is not None else -9999.0,
                                  hi if hi is not None else  9999.0)
            self._v_spin.setDecimals(6)
            self._v_spin.setValue(float(kf.value))
            grid.addWidget(self._v_spin, 2, 1)
        elif ptype == PROP_INT:
            lo, hi = ptype_info[1], ptype_info[2]
            grid.addWidget(QLabel('Value:'), 2, 0)
            self._v_spin = QSpinBox()
            self._v_spin.setRange(int(lo) if lo is not None else -9999,
                                  int(hi) if hi is not None else  9999)
            self._v_spin.setValue(int(kf.value))
            grid.addWidget(self._v_spin, 2, 1)
        elif ptype == PROP_BOOL:
            grid.addWidget(QLabel('Value:'), 2, 0)
            self._v_check = QCheckBox()
            self._v_check.setChecked(bool(kf.value))
            grid.addWidget(self._v_check, 2, 1)
        elif ptype == PROP_COLOR:
            val = kf.value if isinstance(kf.value, (list, tuple)) and len(kf.value) == 3 else (0.0, 0.0, 0.0)
            self._color_spins = []
            for row_i, (ch, init) in enumerate(zip(('R:', 'G:', 'B:'), val)):
                grid.addWidget(QLabel(ch), 2 + row_i, 0)
                sp = QDoubleSpinBox()
                sp.setRange(0.0, 1.0)
                sp.setDecimals(4)
                sp.setValue(float(init))
                grid.addWidget(sp, 2 + row_i, 1)
                self._color_spins.append(sp)
        root.addLayout(grid)
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)

        root.addWidget(btns)

    def _accept(self):
        self.time_val   = self._t_spin.value()
        self.interp_val = self._interp_cb.currentData()
        pt = self._ptype
        if pt in (PROP_FLOAT, PROP_INT):
            self.prop_val = self._v_spin.value()
        elif pt == PROP_BOOL:
            self.prop_val = self._v_check.isChecked()
        elif pt == PROP_COLOR:
            self.prop_val = tuple(sp.value() for sp in self._color_spins)
        self.accept()


class CurveEditor(QWidget):
    MARGIN   = 48
    KEY_R    = 5
    SNAP_PX  = 8

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state        = state
        self._track        = None
        self._view_start   = 0.0
        self._view_end     = 10.0
        self._val_min      = -1.0
        self._val_max      = 1.0
        self._sel_keys     = set()
        self._drag_kf      = None
        self._drag_starts  = {}
        self._drag_px      = QPointF()
        self._hover_kf     = None
        self._box_sel      = False
        self._box_p0       = None
        self._box_p1       = None
        self._pan_active   = False
        self._pan_px       = QPointF()
        self._pan_vs       = 0.0
        self._pan_ve       = 0.0
        self._pan_vmin     = 0.0
        self._pan_vmax     = 0.0
        self.setMinimumHeight(160)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
        state.time_changed.connect(self._on_time)
        state.changed.connect(self._on_data)

    def _on_time(self, _):
        self.update()

    def _on_data(self):
        self._prune_sel()
        self.update()

    def _prune_sel(self):
        if self._track:
            valid = {id(k) for k in self._track.keys}
            self._sel_keys &= valid
        else:
            self._sel_keys.clear()

    def set_track(self, track):
        self._track    = track
        self._sel_keys = set()
        self._drag_kf  = None
        if track and track.keys:
            self._fit_values()
        self.update()

    def _fit_values(self):
        if not self._track:
            return
        vals = [k.value for k in self._track.keys if isinstance(k.value, (int, float))]
        if not vals:
            return
        lo, hi = min(vals), max(vals)
        pad = max((hi - lo) * 0.15, 0.2)
        self._val_min = lo - pad
        self._val_max = hi + pad

    def set_view(self, start, end):
        self._view_start = start
        self._view_end   = end
        self.update()

    def _m(self):
        return self.MARGIN

    def _iw(self):
        return max(self.width() - 2 * self._m(), 1)

    def _ih(self):
        return max(self.height() - 2 * self._m(), 1)

    def _t_to_x(self, t):
        span = self._view_end - self._view_start
        if span < 1e-9:
            return float(self._m())
        return self._m() + (t - self._view_start) / span * self._iw()

    def _x_to_t(self, x):
        iw = self._iw()
        if iw < 1:
            return self._view_start
        return self._view_start + (x - self._m()) / iw * (self._view_end - self._view_start)

    def _v_to_y(self, v):
        span = self._val_max - self._val_min
        if span < 1e-9:
            return self.height() / 2.0
        return self._m() + (1.0 - (v - self._val_min) / span) * self._ih()

    def _y_to_v(self, y):
        ih = self._ih()
        if ih < 1:
            return self._val_min
        return self._val_min + (1.0 - (y - self._m()) / ih) * (self._val_max - self._val_min)

    def _key_at(self, pos):
        if not self._track:
            return None
        best_d, best_k = self.SNAP_PX + 1, None
        for kf in self._track.keys:
            try:
                kx = self._t_to_x(kf.time)
                ky = self._v_to_y(float(kf.value))
            except Exception:
                continue
            d = math.hypot(pos.x() - kx, pos.y() - ky)
            if d < best_d:
                best_d, best_k = d, kf
        return best_k

    def _keys_in_box(self, r):
        result = set()
        if not self._track:
            return result
        for kf in self._track.keys:
            try:
                kx = self._t_to_x(kf.time)
                ky = self._v_to_y(float(kf.value))
            except Exception:
                continue
            if r.contains(QPointF(kx, ky)):
                result.add(id(kf))
        return result

    def _nice_val_step(self, span):
        if span <= 0:
            return 1.0
        raw = span / 5.0
        mag = 10 ** math.floor(math.log10(max(raw, 1e-15)))
        for m in (1, 2, 5, 10):
            s = mag * m
            if span / s <= 7:
                return s
        return mag * 10

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        dark  = _is_dark_theme()
        pal   = self.palette()
        c_bg     = QColor(COLORS['bg'])      if dark else pal.color(pal.Base)
        c_grid   = QColor(COLORS['panel2'])  if dark else pal.color(pal.AlternateBase)
        c_border = QColor(COLORS['border'])  if dark else pal.color(pal.Mid)
        c_fg2    = QColor(COLORS['fg2'])     if dark else pal.color(pal.WindowText)
        c_fg4    = QColor(COLORS['fg4'])     if dark else pal.color(pal.Midlight)
        c_accent = QColor(COLORS['accent'])  if dark else pal.color(pal.Highlight)
        c_ph     = QColor(COLORS['playhead'])
        p.fillRect(self.rect(), c_bg)
        m  = self._m()
        iw = self._iw()
        ih = self._ih()
        t_span = self._view_end - self._view_start
        v_span = self._val_max  - self._val_min
        if t_span > 0:
            from PyQt5.QtCore import Qt as _Qt
            step_t = self._nice_t_step(t_span)
            ti = math.floor(self._view_start / step_t) * step_t
            while ti <= self._view_end + step_t:
                x = self._t_to_x(ti)
                if m <= x <= m + iw:
                    p.setPen(QPen(c_grid, 1, Qt.DotLine))
                    p.drawLine(QPointF(x, m), QPointF(x, m + ih))
                    p.setPen(QPen(c_fg4, 1))
                    p.setFont(_canvas_font(8))
                    lbl = f'{ti:.2g}s'
                    p.drawText(QRectF(x - 20, m + ih + 2, 40, m - 4), Qt.AlignHCenter | Qt.AlignTop, lbl)
                ti += step_t
        if v_span > 0:
            step_v = self._nice_val_step(v_span)
            vi = math.floor(self._val_min / step_v) * step_v
            while vi <= self._val_max + step_v:
                y = self._v_to_y(vi)
                if m <= y <= m + ih:
                    p.setPen(QPen(c_grid, 1, Qt.DotLine))
                    p.drawLine(QPointF(m, y), QPointF(m + iw, y))
                    p.setPen(QPen(c_fg4, 1))
                    p.setFont(_canvas_font(8))
                    p.drawText(QRectF(0, y - 8, m - 4, 16), Qt.AlignRight | Qt.AlignVCenter, f'{vi:.4g}')
                vi += step_v
        p.setPen(QPen(c_border, 1))
        p.drawRect(QRectF(m, m, iw, ih))
        track = self._track
        if not track or not track.keys:
            p.setPen(c_fg4)
            p.setFont(_canvas_font(9))
            msg = 'Select a float track to edit curve' if not track else 'No keyframes'
            p.drawText(self.rect(), Qt.AlignCenter, msg)
            p.end()
            return
        ptype = ALL_PROPERTIES.get(track.prop, (PROP_FLOAT,))[0]
        if ptype not in (PROP_FLOAT, PROP_INT):
            p.setPen(c_fg4)
            p.setFont(_canvas_font(9))
            p.drawText(self.rect(), Qt.AlignCenter, f'No curve for type: {ptype}')
            p.end()
            return
        inner = QRectF(m, m, iw, ih)
        p.save()
        p.setClipRect(inner)
        steps   = max(200, int(iw) * 2)
        path    = QPainterPath()
        started = False
        for i in range(steps + 1):
            t = self._view_start + t_span * i / steps
            v = track.evaluate(t)
            if v is None:
                continue
            x = self._t_to_x(t)
            y = self._v_to_y(float(v))
            if not started:
                path.moveTo(x, y)
                started = True
            else:
                path.lineTo(x, y)
        p.setPen(QPen(c_accent, 1.5))
        p.drawPath(path)
        ph_x = self._t_to_x(self._state.time)
        p.setPen(QPen(c_ph, 1))
        p.drawLine(QPointF(ph_x, m), QPointF(ph_x, m + ih))
        cur_v = track.evaluate(self._state.time)
        if cur_v is not None:
            try:
                cy = self._v_to_y(float(cur_v))
                p.setBrush(QBrush(c_ph))
                p.setPen(Qt.NoPen)
                p.drawEllipse(QPointF(ph_x, cy), 3, 3)
            except Exception:
                pass
        for kf in track.keys:
            try:
                kx = self._t_to_x(kf.time)
                ky = self._v_to_y(float(kf.value))
            except Exception:
                continue
            icolor = QColor(INTERP_COLORS.get(kf.interp, COLORS['key_free']))
            is_sel   = id(kf) in self._sel_keys
            is_hover = self._hover_kf is kf
            if is_sel:
                p.setBrush(QBrush(QColor(COLORS['accent3'])))
                p.setPen(QPen(QColor('#ffffff'), 1.5))
                kr = self.KEY_R + 2
            elif is_hover:
                p.setBrush(QBrush(icolor.lighter(140)))
                p.setPen(QPen(QColor('#ffffff'), 1))
                kr = self.KEY_R + 1
            else:
                p.setBrush(QBrush(icolor))
                p.setPen(QPen(c_bg, 1))
                kr = self.KEY_R
            p.drawEllipse(QPointF(kx, ky), kr, kr)
        if self._box_sel and self._box_p0 and self._box_p1:
            br = QRectF(self._box_p0, self._box_p1).normalized()
            p.setBrush(QBrush(QColor(COLORS['accent'] + '44')))
            p.setPen(QPen(c_accent, 1, Qt.DashLine))
            p.drawRect(br)
        p.restore()
        for kf in track.keys:
            if not (id(kf) in self._sel_keys or self._hover_kf is kf):
                continue
            try:
                kx = self._t_to_x(kf.time)
                ky = self._v_to_y(float(kf.value))
            except Exception:
                continue
            kr = self.KEY_R + (2 if id(kf) in self._sel_keys else 1)
            p.setPen(QPen(c_fg4, 1))
            p.setFont(_canvas_font(8))
            lbl = f't={kf.time:.3f}  v={kf.value:.4g}'
            tx  = kx + kr + 3
            ty  = max(m + 10, min(ky - 3, m + ih - 2))
            p.drawText(QPointF(tx, ty), lbl)
        p.setPen(QPen(c_fg2, 1))
        p.setFont(_canvas_font(9, bold=True))
        p.drawText(QRectF(m + 4, m + 4, iw - 8, 16), Qt.AlignLeft | Qt.AlignTop, track.prop)
        p.end()

    def _nice_t_step(self, span):
        raw = span / 8.0
        if raw <= 0:
            return 1.0
        mag = 10 ** math.floor(math.log10(max(raw, 1e-15)))
        for mult in (1, 2, 5, 10):
            s = mag * mult
            if span / s <= 10:
                return s
        return mag * 10

    def mousePressEvent(self, ev):
        self.setFocus()
        pos = QPointF(ev.pos())
        kf  = self._key_at(pos)
        if ev.button() == Qt.LeftButton:
            if kf is not None:
                kid = id(kf)
                if ev.modifiers() & Qt.ControlModifier:
                    if kid in self._sel_keys:
                        self._sel_keys.discard(kid)
                    else:
                        self._sel_keys.add(kid)
                elif ev.modifiers() & Qt.ShiftModifier:
                    self._sel_keys.add(kid)
                else:
                    if kid not in self._sel_keys:
                        self._sel_keys = {kid}
                self._drag_kf    = True
                self._drag_starts = {id(k): (k.time, k.value)
                                     for k in self._track.keys if id(k) in self._sel_keys}
                self._drag_px    = pos
            else:
                self._sel_keys = set()
                self._box_sel  = True
                self._box_p0   = pos
                self._box_p1   = pos
            self.update()
        elif ev.button() == Qt.MiddleButton:
            self._pan_active = True
            self._pan_px     = pos
            self._pan_vs     = self._view_start
            self._pan_ve     = self._view_end
            self._pan_vmin   = self._val_min
            self._pan_vmax   = self._val_max
        elif ev.button() == Qt.RightButton:
            self._ctx_menu(pos, kf)

    def mouseMoveEvent(self, ev):
        pos = QPointF(ev.pos())
        kf  = self._key_at(pos)
        if kf is not self._hover_kf:
            self._hover_kf = kf
            self.update()
        if self._drag_kf and (ev.buttons() & Qt.LeftButton) and self._track:
            dx     = pos.x() - self._drag_px.x()
            dy     = pos.y() - self._drag_px.y()
            t_span = self._view_end  - self._view_start
            v_span = self._val_max   - self._val_min
            dt     = dx / max(self._iw(), 1) * t_span
            dv     = -dy / max(self._ih(), 1) * v_span
            clip   = self._state.current_clip
            pinfo  = ALL_PROPERTIES.get(self._track.prop, (PROP_FLOAT, None, None))
            lo, hi = pinfo[1], pinfo[2]
            for k in self._track.keys:
                if id(k) not in self._sel_keys:
                    continue
                ot, ov = self._drag_starts.get(id(k), (k.time, k.value))
                nt = max(0.0, ot + dt)
                if clip:
                    nt = min(nt, clip.duration)
                nv = ov + dv
                if lo is not None:
                    nv = max(lo, nv)
                if hi is not None:
                    nv = min(hi, nv)
                k.time  = nt
                k.value = nv
            self._track.keys.sort(key=lambda k: k.time)
            self.update()
        elif self._box_sel and (ev.buttons() & Qt.LeftButton):
            self._box_p1   = pos
            r = QRectF(self._box_p0, self._box_p1).normalized()
            self._sel_keys = self._keys_in_box(r)
            self.update()
        elif self._pan_active and (ev.buttons() & Qt.MiddleButton):
            dx     = pos.x() - self._pan_px.x()
            dy     = pos.y() - self._pan_px.y()
            t_span = self._pan_ve  - self._pan_vs
            v_span = self._pan_vmax - self._pan_vmin
            dt     = -dx / max(self._iw(), 1) * t_span
            dv     =  dy / max(self._ih(), 1) * v_span
            self._view_start = max(0.0, self._pan_vs  + dt)
            self._view_end   = self._pan_ve  + dt
            self._val_min    = self._pan_vmin + dv
            self._val_max    = self._pan_vmax + dv
            self.update()

    def mouseReleaseEvent(self, ev):
        if self._drag_kf and self._track:
            moves = []
            for k in self._track.keys:
                if id(k) not in self._sel_keys:
                    continue
                ot, ov = self._drag_starts.get(id(k), (k.time, k.value))
                if abs(ot - k.time) > 1e-9 or abs(float(ov) - float(k.value)) > 1e-9:
                    moves.append((k, ot, ov, k.time, k.value))
                    k.time, k.value = ot, ov
            if moves:
                track = self._track
                class _MoveCmd(UndoCommand):
                    def __init__(self, t, mv):
                        self._t, self._mv = t, mv
                    def redo(self):
                        for k, _, _, nt, nv in self._mv:
                            k.time, k.value = nt, nv
                        self._t.keys.sort(key=lambda k: k.time)
                    def undo(self):
                        for k, ot, ov, _, _ in self._mv:
                            k.time, k.value = ot, ov
                        self._t.keys.sort(key=lambda k: k.time)
                _undo_stack.push(_MoveCmd(track, moves))
            self._state.changed.emit()
        if self._box_sel:
            self._box_sel = False
            self._box_p0  = None
            self._box_p1  = None
        self._drag_kf    = False
        self._pan_active = False
        self.update()

    def mouseDoubleClickEvent(self, ev):
        if ev.button() != Qt.LeftButton or not self._track:
            return
        pos = QPointF(ev.pos())
        kf  = self._key_at(pos)
        if kf is not None:
            self._edit_key_dialog(kf)
        else:
            t = self._x_to_t(pos.x())
            v = self._y_to_v(pos.y())
            clip = self._state.current_clip
            if clip:
                t = max(0.0, min(t, clip.duration))
            pinfo = ALL_PROPERTIES.get(self._track.prop, (PROP_FLOAT, None, None))
            lo, hi = pinfo[1], pinfo[2]
            if lo is not None: v = max(lo, v)
            if hi is not None: v = min(hi, v)
            _undo_stack.push(AddKeyCommand(self._track, t, v))
            self._state.changed.emit()

    def keyPressEvent(self, ev):
        key  = ev.key()
        mods = ev.modifiers()
        if key in (Qt.Key_Delete, Qt.Key_Backspace):
            self._delete_selected()
        elif key == Qt.Key_A and not mods:
            self._sel_keys = {id(k) for k in self._track.keys} if self._track else set()
            self.update()
        elif key == Qt.Key_F and not mods:
            self._fit_values()
            self.update()
        elif key == Qt.Key_Z and (mods & Qt.ControlModifier):
            if mods & Qt.ShiftModifier:
                _undo_stack.redo()
            else:
                _undo_stack.undo()
            self._state.changed.emit()
        elif key == Qt.Key_Y and (mods & Qt.ControlModifier):
            _undo_stack.redo()
            self._state.changed.emit()
        elif key == Qt.Key_Escape:
            self._sel_keys.clear()
            self.update()
        else:
            super().keyPressEvent(ev)

    def wheelEvent(self, ev):
        pos     = QPointF(ev.pos())
        delta   = ev.angleDelta().y()
        factor  = 0.85 if delta > 0 else 1.0 / 0.85
        mods    = ev.modifiers()
        if mods & Qt.ControlModifier:
            tc = self._x_to_t(pos.x())
            span = (self._view_end - self._view_start) * factor
            frac = (tc - self._view_start) / max(self._view_end - self._view_start, 1e-9)
            self._view_start = tc - frac * span
            self._view_end   = self._view_start + span
        elif mods & Qt.ShiftModifier:
            vc   = self._y_to_v(pos.y())
            span = (self._val_max - self._val_min) * factor
            frac = (vc - self._val_min) / max(self._val_max - self._val_min, 1e-9)
            self._val_min = vc - frac * span
            self._val_max = self._val_min + span
        else:
            vc   = self._y_to_v(pos.y())
            span = (self._val_max - self._val_min) * factor
            frac = (vc - self._val_min) / max(self._val_max - self._val_min, 1e-9)
            self._val_min = vc - frac * span
            self._val_max = self._val_min + span
        self.update()
        ev.accept()

    def _ctx_menu(self, pos, kf):
        menu = QMenu(self)
        if kf is not None:
            sel_kfs = [k for k in (self._track.keys if self._track else []) if id(k) in self._sel_keys]
            if len(sel_kfs) > 1:
                a = menu.addAction(QApplication.style().standardIcon(QStyle.SP_TrashIcon),
                                   f'Delete {len(sel_kfs)} Keyframes')
                a.triggered.connect(self._delete_selected)
                menu.addSeparator()
                im = menu.addMenu('Set Interpolation')
                for mode, label in INTERP_LABELS.items():
                    ac = im.addAction(label)
                    ac.triggered.connect(lambda _, m=mode: self._set_interp_sel(m))
            else:
                a = menu.addAction(QApplication.style().standardIcon(QStyle.SP_TrashIcon), 'Delete Keyframe')
                a.triggered.connect(lambda: self._delete_kf(kf))
                menu.addSeparator()
                im = menu.addMenu('Interpolation')
                for mode, label in INTERP_LABELS.items():
                    ac = im.addAction(label)
                    ac.setCheckable(True)
                    ac.setChecked(kf.interp == mode)
                    ac.triggered.connect(lambda _, m=mode, k=kf: self._set_interp_kf(k, m))
                a2 = menu.addAction(QApplication.style().standardIcon(QStyle.SP_FileDialogDetailedView), 'Edit...')
                a2.triggered.connect(lambda: self._edit_key_dialog(kf))
        else:
            t = self._x_to_t(pos.x())
            v = self._y_to_v(pos.y())
            if self._track:
                a = menu.addAction(QApplication.style().standardIcon(QStyle.SP_FileDialogNewFolder),
                                   f'Add Key  t={t:.3f}  v={v:.4g}')
                a.triggered.connect(lambda: self._add_key_at(t, v))
            menu.addSeparator()
            a_fit = menu.addAction('Fit Values (F)')
            a_fit.triggered.connect(lambda: (self._fit_values(), self.update()))
            a_all = menu.addAction('Select All (A)')
            a_all.triggered.connect(lambda: (self._sel_keys.__class__.__init__(self._sel_keys) or
                                             self._sel_keys.update({id(k) for k in (self._track.keys if self._track else [])}) or
                                             self.update()))
        if menu.actions():
            menu.exec_(self.mapToGlobal(pos.toPoint()))

    def _add_key_at(self, t, v):
        if not self._track:
            return
        clip  = self._state.current_clip
        pinfo = ALL_PROPERTIES.get(self._track.prop, (PROP_FLOAT, None, None))
        lo, hi = pinfo[1], pinfo[2]
        if clip:
            t = max(0.0, min(t, clip.duration))
        if lo is not None: v = max(lo, v)
        if hi is not None: v = min(hi, v)
        _undo_stack.push(AddKeyCommand(self._track, t, v))
        self._state.changed.emit()

    def _delete_selected(self):
        if not self._track or not self._sel_keys:
            return
        pairs = [(self._track, k) for k in self._track.keys if id(k) in self._sel_keys]
        if pairs:
            _undo_stack.push(RemoveKeysCommand(pairs))
            self._sel_keys.clear()
            self._state.changed.emit()

    def _delete_kf(self, kf):
        if not self._track:
            return
        _undo_stack.push(RemoveKeysCommand([(self._track, kf)]))
        self._sel_keys.discard(id(kf))
        self._state.changed.emit()

    def _set_interp_kf(self, kf, mode):
        _undo_stack.push(SetInterpCommand([kf], mode))
        self._state.changed.emit()

    def _set_interp_sel(self, mode):
        if not self._track:
            return
        keys = [k for k in self._track.keys if id(k) in self._sel_keys]
        if keys:
            _undo_stack.push(SetInterpCommand(keys, mode))
            self._state.changed.emit()

    def _edit_key_dialog(self, kf):
        if not self._track:
            return
        ptype_info = ALL_PROPERTIES.get(self._track.prop, (PROP_FLOAT, 0.0, 1.0))
        dlg        = KeyEditDialog(self._track.prop, kf, ptype_info, self)
        if dlg.exec_() == QDialog.Accepted:
            cmd = EditKeyCommand(self._track, kf, kf.time, kf.value, kf.interp,
                                 dlg.time_val, dlg.prop_val, dlg.interp_val)
            _undo_stack.push(cmd)
            self._state.changed.emit()


class PlaybackBar(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._build()
        state.time_changed.connect(self._on_time)
        state.playback_changed.connect(self._on_play)
        state.clip_changed.connect(self._on_clip)

    def _build(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(6, 4, 6, 4)
        root.setSpacing(4)
        self._back_btn = _icon_btn(QStyle.SP_MediaSeekBackward, 'Go to Start')
        self._stop_btn = _icon_btn(QStyle.SP_MediaStop,         'Stop')
        self._play_btn = _icon_btn(QStyle.SP_MediaPlay,         'Play / Pause')
        self._fwd_btn  = _icon_btn(QStyle.SP_MediaSeekForward,  'Go to End')
        self._back_btn.clicked.connect(lambda: self._state.set_time(0.0))
        self._stop_btn.clicked.connect(self._state.stop)
        self._play_btn.clicked.connect(self._state.toggle_play)
        self._fwd_btn.clicked.connect(self._goto_end)
        for btn in (self._back_btn, self._stop_btn, self._play_btn, self._fwd_btn):
            root.addWidget(btn)
        root.addSpacing(8)
        root.addWidget(QLabel('Time:'))
        self._time_lbl = QLabel('0.000s')
        root.addWidget(self._time_lbl)
        root.addSpacing(12)
        root.addWidget(QLabel('Spd:'))
        self._speed_spin = QDoubleSpinBox()
        self._speed_spin.setRange(0.01, 10.0)
        self._speed_spin.setSingleStep(0.1)
        self._speed_spin.setDecimals(2)
        self._speed_spin.setValue(1.0)
        self._speed_spin.valueChanged.connect(self._state.set_speed)
        root.addWidget(self._speed_spin)
        root.addSpacing(8)
        self._loop_btn = QPushButton()
        self._loop_btn.setIcon(QApplication.style().standardIcon(QStyle.SP_BrowserReload))
        self._loop_btn.setCheckable(True)
        self._loop_btn.setToolTip('Loop')
        self._loop_btn.toggled.connect(self._set_loop)
        root.addWidget(self._loop_btn)
        root.addSpacing(8)
        root.addWidget(QLabel('Dur:'))
        self._dur_spin = QDoubleSpinBox()
        self._dur_spin.setRange(0.5, 9999.0)
        self._dur_spin.setSingleStep(1.0)
        self._dur_spin.setDecimals(2)
        self._dur_spin.valueChanged.connect(self._set_duration)
        root.addWidget(self._dur_spin)
        root.addStretch()

    def _on_time(self, t):
        self._time_lbl.setText(f'{t:.3f}s')

    def _on_play(self, playing):
        self._play_btn.setIcon(QApplication.style().standardIcon(
            QStyle.SP_MediaPause if playing else QStyle.SP_MediaPlay))

    def _on_clip(self):
        clip = self._state.current_clip
        if clip:
            self._dur_spin.blockSignals(True)
            self._dur_spin.setValue(clip.duration)
            self._dur_spin.blockSignals(False)
            self._loop_btn.blockSignals(True)
            self._loop_btn.setChecked(clip.loop)
            self._loop_btn.blockSignals(False)

    def _set_duration(self, v):
        clip = self._state.current_clip
        if clip:
            clip.duration = v
            self._state.changed.emit()

    def _set_loop(self, v):
        clip = self._state.current_clip
        if clip:
            clip.loop = v

    def _goto_end(self):
        clip = self._state.current_clip
        if clip:
            self._state.set_time(clip.duration)


class InspectorPanel(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state = state

        self._build()
        state.changed.connect(self._refresh)
        state.clip_changed.connect(self._refresh)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        root.setSpacing(4)
        hdr = QLabel("INSPECTOR")

        root.addWidget(hdr)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._content = QWidget()
        self._v_layout = QVBoxLayout(self._content)
        self._v_layout.setContentsMargins(0, 0, 0, 0)
        self._v_layout.setSpacing(4)
        self._v_layout.addStretch()
        self._scroll.setWidget(self._content)
        root.addWidget(self._scroll)

    def _clear(self):
        while self._v_layout.count() > 1:
            item = self._v_layout.takeAt(0)
            if item and item.widget():
                item.widget().deleteLater()

    def _refresh(self):
        self._clear()
        clip = self._state.current_clip
        if not clip:
            return
        for label, value in [
            ('Clip',     clip.name),
            ('Duration', f'{clip.duration:.2f}s'),
            ('Tracks',   str(len(clip.tracks))),
            ('Loop',     'Yes' if clip.loop else 'No'),
        ]:
            row = QWidget()

            h = QHBoxLayout(row)
            h.setContentsMargins(6, 3, 6, 3)
            lbl = QLabel(label + ':')

            val = QLabel(value)

            h.addWidget(lbl)
            h.addStretch()
            h.addWidget(val)
            self._v_layout.insertWidget(self._v_layout.count() - 1, row)


class AnimationEditorWindow(QMainWindow):
    closed = pyqtSignal()
    WIN_W  = 1300
    WIN_H  = 700

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self._state      = state
        self._view_start = 0.0
        self._view_end   = 10.0
        self.setWindowTitle('Animation Editor')
        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        state.time_changed.connect(self._on_time)
        state.changed.connect(self._on_change)
        state.clip_changed.connect(self._on_clip_change)
        if _app_config:
            _app_config.restore_window_geometry("anim_editor", self,
                                                default_w=self.WIN_W, default_h=self.WIN_H)
            _app_config.register_theme_change_callback(self._on_theme_changed)
        else:
            self.resize(self.WIN_W, self.WIN_H)
        QTimer.singleShot(0, self._on_clip_change)

    def _on_theme_changed(self, theme_name: str):
        if _app_config:
            _app_config.save_window_geometry("anim_editor", self)
        geom = self.geometry()
        for sig, slot in [
            (self._state.time_changed,    self._on_time),
            (self._state.changed,         self._on_change),
            (self._state.clip_changed,    self._on_clip_change),
            (_undo_stack.changed,         self._refresh_undo_btns),
        ]:
            try:
                sig.disconnect(slot)
            except (RuntimeError, TypeError):
                pass
        old_central = self.centralWidget()
        from PyQt5.QtWidgets import QToolBar, QStatusBar
        for tb in self.findChildren(QToolBar):
            self.removeToolBar(tb)
            tb.deleteLater()
        old_sb = self.statusBar()
        old_sb.deleteLater()
        self.setStatusBar(QStatusBar(self))
        self._build_ui()
        self._build_toolbar()
        self._build_statusbar()
        if old_central:
            old_central.deleteLater()
        self._state.time_changed.connect(self._on_time)
        self._state.changed.connect(self._on_change)
        self._state.clip_changed.connect(self._on_clip_change)
        self.setGeometry(geom)
        QTimer.singleShot(0, self._on_clip_change)

    def _build_toolbar(self):
        tb = self.addToolBar('Main')
        tb.setMovable(False)

        tb.addAction(self.style().standardIcon(QStyle.SP_FileDialogNewFolder), 'New Clip').triggered.connect(self._new_clip)
        tb.addAction(self.style().standardIcon(QStyle.SP_DialogOpenButton),    'Open').triggered.connect(self._open)
        tb.addAction(self.style().standardIcon(QStyle.SP_DialogSaveButton),    'Save').triggered.connect(self._save)
        tb.addSeparator()
        self._undo_act = tb.addAction(self.style().standardIcon(QStyle.SP_ArrowBack),    'Undo  Ctrl+Z')
        self._redo_act = tb.addAction(self.style().standardIcon(QStyle.SP_ArrowForward), 'Redo  Ctrl+Y / Ctrl+Shift+Z')
        self._undo_act.triggered.connect(lambda: (_undo_stack.undo(), self._state.changed.emit()))
        self._redo_act.triggered.connect(lambda: (_undo_stack.redo(), self._state.changed.emit()))
        try:
            _undo_stack.changed.disconnect(self._refresh_undo_btns)
        except (RuntimeError, TypeError):
            pass
        _undo_stack.changed.connect(self._refresh_undo_btns)
        self._refresh_undo_btns()
        tb.addSeparator()
        tb.addAction(self.style().standardIcon(QStyle.SP_FileIcon),            'Add Key at Playhead  K').triggered.connect(self._add_key_at_current)
        tb.addAction(self.style().standardIcon(QStyle.SP_FileDialogListView),  'Add Track').triggered.connect(self._add_track_dialog)
        tb.addSeparator()
        tb.addAction(self.style().standardIcon(QStyle.SP_DesktopIcon),         'Fit View  F').triggered.connect(self._frame_all)

    def _refresh_undo_btns(self):
        self._undo_act.setEnabled(_undo_stack.can_undo())
        self._redo_act.setEnabled(_undo_stack.can_redo())

    def _build_statusbar(self):
        sb = self.statusBar()
        self._status_lbl = QLabel('Ready')
        sb.addWidget(self._status_lbl)
        hint = QLabel('Space: Play  |  K/I: Add Key  |  Del: Delete  |  A: Sel All  |  F: Fit  |  Ctrl+Z/Y: Undo/Redo  |  Timeline: MMB drag=pan view  |  Curve: LMB drag pts, DblClick=add, Scroll=zoom val, Ctrl+Scroll=zoom time, MMB=pan')

        sb.addPermanentWidget(hint)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        ml = QVBoxLayout(central)
        ml.setContentsMargins(0, 0, 0, 0)
        ml.setSpacing(0)
        self._playback_bar = PlaybackBar(self._state)
        self._playback_bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ml.addWidget(self._playback_bar)
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ml.addWidget(sep)
        hsplit = QSplitter(Qt.Horizontal)
        hsplit.setHandleWidth(3)
        self._clip_panel = ClipListPanel(self._state)
        self._clip_panel.setMinimumWidth(160)
        self._clip_panel.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        hsplit.addWidget(self._clip_panel)
        center = QWidget()
        clayout = QVBoxLayout(center)
        clayout.setContentsMargins(0, 0, 0, 0)
        clayout.setSpacing(0)
        clayout.addWidget(self._build_track_toolbar())
        vsplit = QSplitter(Qt.Vertical)
        vsplit.setHandleWidth(3)
        tl_container = QWidget()
        tl_layout    = QVBoxLayout(tl_container)
        tl_layout.setContentsMargins(0, 0, 0, 0)
        tl_layout.setSpacing(0)
        self._ruler = TimeRuler()
        self._ruler.seek_requested.connect(self._state.set_time)
        tl_layout.addWidget(self._ruler)
        self._track_scroll = QScrollArea()
        self._track_scroll.setWidgetResizable(True)
        self._track_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._track_widget = TrackWidget(self._state)
        self._track_widget.key_added.connect(self._on_key_added)
        self._track_widget.seek_requested.connect(self._state.set_time)
        self._track_widget.track_selected.connect(self._on_track_selected)
        self._track_scroll.setWidget(self._track_widget)
        tl_layout.addWidget(self._track_scroll)
        tl_layout.addWidget(self._build_zoom_bar())
        vsplit.addWidget(tl_container)
        self._curve_editor = CurveEditor(self._state)
        vsplit.addWidget(self._curve_editor)
        vsplit.setSizes([360, 160])
        clayout.addWidget(vsplit)
        hsplit.addWidget(center)
        self._inspector = InspectorPanel(self._state)
        self._inspector.setMinimumWidth(160)
        self._inspector.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        hsplit.addWidget(self._inspector)
        hsplit.setSizes([200, 880, 200])
        ml.addWidget(hsplit, 1)
        self._sync_view()

    def _build_track_toolbar(self):
        bar = QWidget()
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(4)
        add_btn = QPushButton('+ Add Track')
        add_btn.clicked.connect(self._add_track_dialog)
        row.addWidget(add_btn)
        cap_btn = QPushButton('Capture All')
        cap_btn.setToolTip('Capture current property values as keyframe at playhead')
        cap_btn.clicked.connect(self._capture_all)
        row.addWidget(cap_btn)
        row.addStretch()
        row.addWidget(QLabel('Filter:'))
        self._filter_edit = QLineEdit()
        self._filter_edit.setPlaceholderText('Filter tracks...')
        row.addWidget(self._filter_edit)
        return bar

    def _build_zoom_bar(self):
        bar = QWidget()
        bar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row = QHBoxLayout(bar)
        row.setContentsMargins(6, 2, 6, 2)
        row.setSpacing(4)
        zm = _icon_btn(QStyle.SP_ArrowLeft,  'Zoom Out')
        zp = _icon_btn(QStyle.SP_ArrowRight, 'Zoom In')
        zm.clicked.connect(self._zoom_out)
        zp.clicked.connect(self._zoom_in)
        row.addWidget(zm)
        row.addWidget(zp)
        self._zoom_lbl = QLabel()
        row.addWidget(self._zoom_lbl)
        row.addStretch()
        fit_btn = QPushButton('Fit View')
        fit_btn.clicked.connect(self._frame_all)
        row.addWidget(fit_btn)
        return bar

    def _sync_view(self):
        self._ruler.set_view(self._view_start, self._view_end)
        self._track_widget.set_view(self._view_start, self._view_end)
        self._curve_editor.set_view(self._view_start, self._view_end)
        span = self._view_end - self._view_start
        self._zoom_lbl.setText(f'{self._view_start:.1f}s — {self._view_end:.1f}s  ({span:.1f}s)')

    def _zoom_in(self):
        span   = self._view_end - self._view_start
        center = (self._view_start + self._view_end) * 0.5
        span   = max(0.5, span * 0.6)
        self._view_start = center - span * 0.5
        self._view_end   = center + span * 0.5
        self._sync_view()

    def _zoom_out(self):
        span   = self._view_end - self._view_start
        center = (self._view_start + self._view_end) * 0.5
        span   = min(9999.0, span * 1.6)
        self._view_start = max(0.0, center - span * 0.5)
        self._view_end   = self._view_start + span
        self._sync_view()

    def _frame_all(self):
        clip = self._state.current_clip
        if not clip:
            return
        self._view_start = 0.0
        self._view_end   = max(clip.duration, 0.5)
        self._sync_view()

    def wheelEvent(self, ev):
        if ev.angleDelta().y() > 0:
            self._zoom_in()
        else:
            self._zoom_out()
        ev.accept()

    def _on_track_selected(self, track):
        ptype = ALL_PROPERTIES.get(track.prop, (PROP_FLOAT,))[0] if track else None
        if track and ptype in (PROP_FLOAT, PROP_INT):
            self._curve_editor.set_track(track)
        else:
            self._curve_editor.set_track(None)

    def keyPressEvent(self, ev):
        key  = ev.key()
        mods = ev.modifiers()
        if key == Qt.Key_Space:
            self._state.toggle_play()
        elif key == Qt.Key_Z and (mods & Qt.ControlModifier):
            if mods & Qt.ShiftModifier:
                _undo_stack.redo()
            else:
                _undo_stack.undo()
            self._state.changed.emit()
        elif key == Qt.Key_Y and (mods & Qt.ControlModifier):
            _undo_stack.redo()
            self._state.changed.emit()
        elif key == Qt.Key_S and (mods & Qt.ControlModifier):
            self._save()
        elif key == Qt.Key_O and (mods & Qt.ControlModifier):
            self._open()
        elif key == Qt.Key_F:
            self._frame_all()
        elif key == Qt.Key_Home:
            self._state.set_time(0.0)
        elif key == Qt.Key_End:
            clip = self._state.current_clip
            if clip:
                self._state.set_time(clip.duration)
        elif key == Qt.Key_Left:
            step = 1.0 / 24.0 if not (mods & Qt.ShiftModifier) else 1.0
            self._state.set_time(max(0.0, self._state.time - step))
        elif key == Qt.Key_Right:
            clip = self._state.current_clip
            step = 1.0 / 24.0 if not (mods & Qt.ShiftModifier) else 1.0
            t_max = clip.duration if clip else 9999.0
            self._state.set_time(min(t_max, self._state.time + step))
        elif key == Qt.Key_K or key == Qt.Key_I:
            self._add_key_at_current()
        else:
            super().keyPressEvent(ev)

    def _on_time(self, t):
        self._ruler.set_time(t)
        self._status_lbl.setText(f't = {t:.4f}s  |  Undo: {_undo_stack._pos + 1}/{len(_undo_stack._stack)}')

    def _on_change(self):
        self._track_widget.update()
        self._curve_editor.update()

    def _on_clip_change(self):
        clip = self._state.current_clip
        if clip:
            self._ruler.set_duration(clip.duration)
            self._view_end = max(clip.duration, 0.5)
            self._sync_view()
        self._track_widget.update()
        self._track_widget._selected_track = None
        self._track_widget._selected_keys  = set()
        self._curve_editor.set_track(None)

    def _add_track_dialog(self):
        clip = self._state.current_clip
        if not clip:
            QMessageBox.warning(self, 'No Clip', 'Create or select a clip first.')
            return
        existing = {t.prop for t in clip.tracks}
        dlg = PropSelectorDialog(existing, self)
        if dlg.exec_() == QDialog.Accepted:
            for prop in dlg.selected_props():
                clip.get_or_create_track(prop)
            self._state.changed.emit()

    def _add_key_at_current(self):
        clip = self._state.current_clip
        if not clip:
            return
        t     = self._state.time
        count = 0
        for track in clip.tracks:
            if not track.enabled:
                continue
            val = _get_param_value(track.prop)
            if val is not None:
                _undo_stack.push(AddKeyCommand(track, t, val))
                count += 1
        self._state.changed.emit()
        self._status_lbl.setText(f'Added {count} keyframes at t={t:.3f}s')

    def _capture_all(self):
        self._add_key_at_current()

    def _on_key_added(self, prop, t):
        clip = self._state.current_clip
        if not clip:
            return
        track = clip.get_or_create_track(prop)
        val   = _get_param_value(prop)
        if val is None:
            ptype = ALL_PROPERTIES.get(prop, (PROP_FLOAT,))[0]
            val   = (0.0, 0.0, 0.0) if ptype == PROP_COLOR else 0.0
        _undo_stack.push(AddKeyCommand(track, t, val))
        self._state.changed.emit()

    def _new_clip(self):
        name, ok = QInputDialog.getText(self, 'New Clip', 'Clip name:', text='New Clip')
        if ok and name:
            self._state.add_clip(name)

    def _save(self):
        path, _ = QFileDialog.getSaveFileName(self, 'Save Animation', '',
                                              'ZarAnim (*.zanim);;JSON (*.json)')
        if path:
            try:
                self._state.save(path)
                self._status_lbl.setText(f'Saved: {path}')
            except Exception as e:
                QMessageBox.critical(self, 'Save Error', str(e))

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Open Animation', '',
                                              'ZarAnim (*.zanim);;JSON (*.json)')
        if path:
            try:
                self._state.load(path)
                self._status_lbl.setText(f'Loaded: {path}')
            except Exception as e:
                QMessageBox.critical(self, 'Load Error', str(e))

    def closeEvent(self, ev):
        if _app_config:
            _app_config.save_window_geometry("anim_editor", self)
            _app_config.unregister_theme_change_callback(self._on_theme_changed)
        self._state.pause()
        self.closed.emit()
        super().closeEvent(ev)


_params_ref = None


def set_params_ref(params_obj):
    global _params_ref
    _params_ref = params_obj
    _apply_dispatch_cache.clear()
    _anim_state.set_apply_callback(apply_anim_to_params)


def _get_param_value(prop):
    p = _params_ref
    if p is None:
        return None
    try:
        if prop == 'cam_pos_x':
            return float(p.cam_pos[0])
        if prop == 'cam_pos_y':
            return float(p.cam_pos[1])
        if prop == 'cam_pos_z':
            return float(p.cam_pos[2])
        if prop == 'light2_color':
            return (float(p.light2_r), float(p.light2_g), float(p.light2_b))
        if hasattr(p, prop):
            v = getattr(p, prop)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                return tuple(float(c) for c in v[:3])
            return v
    except Exception:
        pass
    return None


_apply_dispatch_cache = {}


def _build_apply_dispatch(p):
    dispatch = {}
    for prop in ALL_PROPERTIES:
        if prop == 'cam_pos_x':
            dispatch[prop] = ('cam_pos', 0)
        elif prop == 'cam_pos_y':
            dispatch[prop] = ('cam_pos', 1)
        elif prop == 'cam_pos_z':
            dispatch[prop] = ('cam_pos', 2)
        elif prop == 'light2_color':
            dispatch[prop] = 'light2_color'
        elif hasattr(p, prop):
            cur = getattr(p, prop)
            if isinstance(cur, bool):
                dispatch[prop] = ('attr_bool', prop)
            elif isinstance(cur, int):
                dispatch[prop] = ('attr_int', prop)
            elif isinstance(cur, float):
                dispatch[prop] = ('attr_float', prop)
            elif isinstance(cur, tuple):
                dispatch[prop] = ('attr_tuple', prop)
            elif isinstance(cur, list):
                dispatch[prop] = ('attr_list', prop, len(cur))
            else:
                dispatch[prop] = ('attr_typed', prop, type(cur))
    return dispatch


def apply_anim_to_params(values):
    p = _params_ref
    if p is None:
        return
    global _apply_dispatch_cache
    dispatch = _apply_dispatch_cache.get(id(p))
    if dispatch is None:
        dispatch = _build_apply_dispatch(p)
        _apply_dispatch_cache[id(p)] = dispatch
    for prop, val in values.items():
        entry = dispatch.get(prop)
        if entry is None:
            continue
        try:
            if entry == 'light2_color':
                if isinstance(val, (tuple, list)) and len(val) == 3:
                    p.light2_r = float(val[0])
                    p.light2_g = float(val[1])
                    p.light2_b = float(val[2])
            elif entry[0] == 'cam_pos':
                p.cam_pos[entry[1]] = float(val)
            elif entry[0] == 'attr_bool':
                setattr(p, entry[1], bool(val))
            elif entry[0] == 'attr_int':
                setattr(p, entry[1], int(round(float(val))))
            elif entry[0] == 'attr_float':
                setattr(p, entry[1], float(val))
            elif entry[0] == 'attr_tuple':
                if isinstance(val, (tuple, list)):
                    setattr(p, entry[1], tuple(float(v) for v in val))
            elif entry[0] == 'attr_list':
                lst = getattr(p, entry[1])
                n   = entry[2]
                for i, v in enumerate(val):
                    if i < n:
                        lst[i] = float(v)
            elif entry[0] == 'attr_typed':
                setattr(p, entry[1], entry[2](val))
        except Exception:
            pass

_anim_window = None


def open_animation_editor():
    global _anim_window
    try:
        if _anim_window is not None and not _anim_window.isHidden():
            _anim_window.raise_()
            _anim_window.activateWindow()
            return _anim_window
        _anim_window = AnimationEditorWindow(_anim_state)
        _anim_window.show()
    except Exception as e:
        import traceback
        traceback.print_exc()
    return _anim_window