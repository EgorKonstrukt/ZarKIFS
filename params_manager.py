from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Tuple, List, Optional
from enum import IntEnum


class PropType(IntEnum):
    FLOAT = 0
    COLOR = 1
    BOOL = 2
    INT = 3


@dataclass
class ParamDef:
    name: str
    prop_type: PropType
    min_val: float = 0.0
    max_val: float = 1.0
    default: Any = None
    label: str = ""
    category: str = "General"


PARAM_DEFINITIONS: Dict[str, ParamDef] = {
    'iterations':        ParamDef('iterations',        PropType.INT,   1,     20,     8,      'Iterations',          'Fractal'),
    'scale':             ParamDef('scale',             PropType.FLOAT, 0.1,   10.0,   3.0,    'Scale',               'Fractal'),
    'fold_x':            ParamDef('fold_x',            PropType.BOOL,  0,     1,      True,   'Fold X',              'Fractal'),
    'fold_y':            ParamDef('fold_y',            PropType.BOOL,  0,     1,      True,   'Fold Y',              'Fractal'),
    'fold_z':            ParamDef('fold_z',            PropType.BOOL,  0,     1,      True,   'Fold Z',              'Fractal'),
    'rot_x':             ParamDef('rot_x',             PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation X',          'Fractal'),
    'rot_y':             ParamDef('rot_y',             PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation Y',          'Fractal'),
    'rot_z':             ParamDef('rot_z',             PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation Z',          'Fractal'),
    'offset_x':          ParamDef('offset_x',          PropType.FLOAT, -5.0,  5.0,    2.0,    'Offset X',            'Fractal'),
    'offset_y':          ParamDef('offset_y',          PropType.FLOAT, -5.0,  5.0,    2.0,    'Offset Y',            'Fractal'),
    'offset_z':          ParamDef('offset_z',          PropType.FLOAT, -5.0,  5.0,    2.0,    'Offset Z',            'Fractal'),
    'julia_x':           ParamDef('julia_x',           PropType.FLOAT, -2.0,  2.0,    -0.5,   'Julia X',             'Fractal'),
    'julia_y':           ParamDef('julia_y',           PropType.FLOAT, -2.0,  2.0,    -0.5,   'Julia Y',             'Fractal'),
    'julia_z':           ParamDef('julia_z',           PropType.FLOAT, -2.0,  2.0,    -0.5,   'Julia Z',             'Fractal'),
    'fractal_type':      ParamDef('fractal_type',      PropType.INT,   0,     10,     0,      'Fractal Type',        'Fractal'),
    'bailout':           ParamDef('bailout',           PropType.FLOAT, 1.0,   50.0,   100.0,  'Bailout',             'Fractal'),
    'min_dist':          ParamDef('min_dist',          PropType.FLOAT, 0.1,   10.0,   1.0,    'Min Distance',        'Fractal'),
    'fog_density':       ParamDef('fog_density',       PropType.FLOAT, 0.0,   5.0,    0.5,    'Fog Density',         'Fractal'),
    'color1':            ParamDef('color1',            PropType.COLOR, 0.0,   1.0,    (0.5, 0.5, 0.5), 'Color 1',            'Colors'),
    'color2':            ParamDef('color2',            PropType.COLOR, 0.0,   1.0,    (0.5, 0.5, 0.5), 'Color 2',            'Colors'),
    'color3':            ParamDef('color3',            PropType.COLOR, 0.0,   1.0,    (1.0, 1.0, 1.0), 'Color 3',            'Colors'),
    'color_mode':        ParamDef('color_mode',        PropType.INT,   0,     5,      0,      'Color Mode',          'Colors'),
    'ao_strength':       ParamDef('ao_strength',       PropType.FLOAT, 0.0,   3.0,    1.0,    'AO Strength',         'Lighting'),
    'ao_radius':         ParamDef('ao_radius',         PropType.FLOAT, 0.01,  1.0,    0.12,   'AO Radius',           'Lighting'),
    'ao_samples':        ParamDef('ao_samples',        PropType.INT,   1,     16,     5,      'AO Samples',          'Lighting'),
    'shadow_soft':       ParamDef('shadow_soft',       PropType.FLOAT, 0.5,   32.0,   8.0,    'Shadow Softness',     'Lighting'),
    'shadows':           ParamDef('shadows',           PropType.BOOL,  0,     1,      True,   'Shadows',             'Lighting'),
    'glow':              ParamDef('glow',              PropType.FLOAT, 0.0,   20.0,   5.0,    'Glow',                'Lighting'),
    'glow_intensity':    ParamDef('glow_intensity',    PropType.FLOAT, 0.0,   20.0,   5.0,    'Glow Intensity',      'Lighting'),
    'glow_falloff':      ParamDef('glow_falloff',      PropType.FLOAT, 0.5,   20.0,   8.0,    'Glow Falloff',        'Lighting'),
    'glow_radius':       ParamDef('glow_radius',       PropType.FLOAT, 0.0,   5.0,    1.0,    'Glow Radius',         'Lighting'),
    'rim_strength':      ParamDef('rim_strength',      PropType.FLOAT, 0.0,   3.0,    0.4,    'Rim Strength',        'Lighting'),
    'emission':          ParamDef('emission',          PropType.FLOAT, 0.0,   3.0,    0.2,    'Emission',            'Lighting'),
    'cam_pos':           ParamDef('cam_pos',           PropType.COLOR, -10.0, 10.0,   [0.0, 0.0, 5.0], 'Camera Position',     'Camera'),
    'cam_yaw':           ParamDef('cam_yaw',           PropType.FLOAT, -3.14, 3.14,   0.0,    'Camera Yaw',          'Camera'),
    'cam_pitch':         ParamDef('cam_pitch',         PropType.FLOAT, -1.57, 1.57,   0.0,    'Camera Pitch',        'Camera'),
    'cam_roll':          ParamDef('cam_roll',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Camera Roll',         'Camera'),
    'animate':           ParamDef('animate',           PropType.BOOL,  0,     1,      True,   'Animate',             'Animation'),
    'anim_speed':        ParamDef('anim_speed',        PropType.FLOAT, 0.0,   10.0,   2.5,    'Animation Speed',     'Animation'),
    'color_anim_speed':  ParamDef('color_anim_speed',  PropType.FLOAT, 0.0,   1.0,    0.05,   'Color Anim Speed',    'Animation'),
    'color_offset':      ParamDef('color_offset',      PropType.FLOAT, 0.0,   1.0,    0.0,    'Color Offset',        'Animation'),
    'mb_fold_limit':     ParamDef('mb_fold_limit',     PropType.FLOAT, 0.1,   5.0,    1.0,    'Fold Limit',          'Mandelbox'),
    'mb_sphere_inner':   ParamDef('mb_sphere_inner',   PropType.FLOAT, 0.01,  2.0,    0.25,   'Sphere Inner',        'Mandelbox'),
    'mb_sphere_outer':   ParamDef('mb_sphere_outer',   PropType.FLOAT, 0.1,   3.0,    1.0,    'Sphere Outer',        'Mandelbox'),
    'mb_fixed_radius':   ParamDef('mb_fixed_radius',   PropType.FLOAT, 0.1,   3.0,    1.0,    'Fixed Radius',        'Mandelbox'),
    'mb_color_scale':    ParamDef('mb_color_scale',    PropType.FLOAT, 0.1,   10.0,   5.0,    'Color Scale',         'Mandelbox'),
    'mb_rot_per_iter':   ParamDef('mb_rot_per_iter',   PropType.FLOAT, 0.0,   1.0,    0.0,    'Rot Per Iter',        'Mandelbox'),
    'mb_fold_mode':      ParamDef('mb_fold_mode',      PropType.INT,   0,     2,      0,      'Fold Mode',           'Mandelbox'),
    'mb_fold_x':         ParamDef('mb_fold_x',         PropType.FLOAT, -5.0,  5.0,    0.0,    'Fold X',              'Mandelbox'),
    'mb_fold_y':         ParamDef('mb_fold_y',         PropType.FLOAT, -5.0,  5.0,    0.0,    'Fold Y',              'Mandelbox'),
    'mb_fold_z':         ParamDef('mb_fold_z',         PropType.FLOAT, -5.0,  5.0,    0.0,    'Fold Z',              'Mandelbox'),
    'mb_julia_mode':     ParamDef('mb_julia_mode',     PropType.INT,   0,     3,      1,      'Julia Mode',          'Mandelbox'),
    'mb_scale_x':        ParamDef('mb_scale_x',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale X',             'Mandelbox'),
    'mb_scale_y':        ParamDef('mb_scale_y',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Y',             'Mandelbox'),
    'mb_scale_z':        ParamDef('mb_scale_z',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Z',             'Mandelbox'),
    'mb_offset_x':       ParamDef('mb_offset_x',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset X',            'Mandelbox'),
    'mb_offset_y':       ParamDef('mb_offset_y',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Y',            'Mandelbox'),
    'mb_offset_z':       ParamDef('mb_offset_z',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Z',            'Mandelbox'),
    'mb_inversion_radius': ParamDef('mb_inversion_radius', PropType.FLOAT, 0.0, 5.0, 0.0,    'Inversion Radius',    'Mandelbox'),
    'ms_cross_width':    ParamDef('ms_cross_width',    PropType.FLOAT, 0.0,   3.0,    1.0,    'Cross Width',         'Menger'),
    'ms_scale':          ParamDef('ms_scale',          PropType.FLOAT, 1.5,   6.0,    3.0,    'Scale',               'Menger'),
    'ms_offset':         ParamDef('ms_offset',         PropType.FLOAT, 0.5,   5.0,    2.0,    'Offset',              'Menger'),
    'ms_twist':          ParamDef('ms_twist',          PropType.FLOAT, 0.0,   1.0,    0.0,    'Twist',               'Menger'),
    'ms_sharpness':      ParamDef('ms_sharpness',      PropType.FLOAT, 0.1,   3.0,    1.0,    'Sharpness',           'Menger'),
    'ms_rot_x':          ParamDef('ms_rot_x',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation X',          'Menger'),
    'ms_rot_z':          ParamDef('ms_rot_z',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation Z',          'Menger'),
    'ms_scale_y':        ParamDef('ms_scale_y',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Y',             'Menger'),
    'ms_scale_z':        ParamDef('ms_scale_z',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Z',             'Menger'),
    'ms_offset_x':       ParamDef('ms_offset_x',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset X',            'Menger'),
    'ms_offset_y':       ParamDef('ms_offset_y',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Y',            'Menger'),
    'ms_offset_z':       ParamDef('ms_offset_z',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Z',            'Menger'),
    'ms_fold_type':      ParamDef('ms_fold_type',      PropType.INT,   0,     3,      0,      'Fold Type',           'Menger'),
    'ms_fold_abs_amount': ParamDef('ms_fold_abs_amount', PropType.FLOAT, 0.0, 1.0, 0.5,     'Fold Abs Amount',     'Menger'),
    'si_vertex_spread':  ParamDef('si_vertex_spread',  PropType.FLOAT, 0.3,   3.0,    1.0,    'Vertex Spread',       'Sierpinski'),
    'si_fold_bias':      ParamDef('si_fold_bias',      PropType.FLOAT, 1.3,   4.0,    2.0,    'Fold Bias',           'Sierpinski'),
    'si_twist':          ParamDef('si_twist',          PropType.FLOAT, 0.0,   1.0,    0.0,    'Twist',               'Sierpinski'),
    'si_squash':         ParamDef('si_squash',         PropType.FLOAT, 0.1,   3.0,    1.0,    'Squash',              'Sierpinski'),
    'si_vertex_jitter':  ParamDef('si_vertex_jitter',  PropType.FLOAT, 0.0,   1.0,    0.0,    'Vertex Jitter',       'Sierpinski'),
    'si_rot_x':          ParamDef('si_rot_x',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation X',          'Sierpinski'),
    'si_rot_y':          ParamDef('si_rot_y',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation Y',          'Sierpinski'),
    'si_rot_z':          ParamDef('si_rot_z',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation Z',          'Sierpinski'),
    'si_scale_x':        ParamDef('si_scale_x',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale X',             'Sierpinski'),
    'si_scale_y':        ParamDef('si_scale_y',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Y',             'Sierpinski'),
    'si_scale_z':        ParamDef('si_scale_z',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Z',             'Sierpinski'),
    'si_offset_x':       ParamDef('si_offset_x',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset X',            'Sierpinski'),
    'si_offset_y':       ParamDef('si_offset_y',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Y',            'Sierpinski'),
    'si_offset_z':       ParamDef('si_offset_z',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Z',            'Sierpinski'),
    'oc_ifs_scale':      ParamDef('oc_ifs_scale',      PropType.FLOAT, 1.3,   4.0,    2.0,    'IFS Scale',           'Octahedron'),
    'oc_twist':          ParamDef('oc_twist',          PropType.FLOAT, 0.0,   1.0,    0.0,    'Twist',               'Octahedron'),
    'oc_sharpness':      ParamDef('oc_sharpness',      PropType.FLOAT, 0.1,   3.0,    1.0,    'Sharpness',           'Octahedron'),
    'oc_offset_uni':     ParamDef('oc_offset_uni',     PropType.FLOAT, 0.0,   3.0,    1.0,    'Offset Uniform',      'Octahedron'),
    'oc_fold_amount':    ParamDef('oc_fold_amount',    PropType.FLOAT, 0.0,   1.0,    0.0,    'Fold Amount',         'Octahedron'),
    'oc_offset_x':       ParamDef('oc_offset_x',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset X',            'Octahedron'),
    'oc_offset_y':       ParamDef('oc_offset_y',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Y',            'Octahedron'),
    'oc_offset_z':       ParamDef('oc_offset_z',       PropType.FLOAT, -5.0,  5.0,    0.0,    'Offset Z',            'Octahedron'),
    'oc_rot_x':          ParamDef('oc_rot_x',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation X',          'Octahedron'),
    'oc_rot_z':          ParamDef('oc_rot_z',          PropType.FLOAT, -3.14, 3.14,   0.0,    'Rotation Z',          'Octahedron'),
    'oc_scale_y':        ParamDef('oc_scale_y',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Y',             'Octahedron'),
    'oc_scale_z':        ParamDef('oc_scale_z',        PropType.FLOAT, 0.0,   5.0,    0.0,    'Scale Z',             'Octahedron'),
    'oc_julia_mode':     ParamDef('oc_julia_mode',     PropType.INT,   0,     3,      0,      'Julia Mode',          'Octahedron'),
    'oc_julia_x':        ParamDef('oc_julia_x',        PropType.FLOAT, -2.0,  2.0,    0.0,    'Julia X',             'Octahedron'),
    'oc_julia_y':        ParamDef('oc_julia_y',        PropType.FLOAT, -2.0,  2.0,    0.0,    'Julia Y',             'Octahedron'),
    'oc_julia_z':        ParamDef('oc_julia_z',        PropType.FLOAT, -2.0,  2.0,    0.0,    'Julia Z',             'Octahedron'),
    'mb2_power':         ParamDef('mb2_power',         PropType.FLOAT, 2.0,   16.0,   8.0,    'Power',               'Mandelbulb'),
    'mb2_bailout':       ParamDef('mb2_bailout',       PropType.FLOAT, 1.0,   10.0,   2.0,    'Bailout',             'Mandelbulb'),
    'mb2_julia_x':       ParamDef('mb2_julia_x',       PropType.FLOAT, -2.0,  2.0,    0.0,    'Julia X',             'Mandelbulb'),
    'mb2_julia_y':       ParamDef('mb2_julia_y',       PropType.FLOAT, -2.0,  2.0,    0.0,    'Julia Y',             'Mandelbulb'),
    'mb2_julia_z':       ParamDef('mb2_julia_z',       PropType.FLOAT, -2.0,  2.0,    0.0,    'Julia Z',             'Mandelbulb'),
    'mb2_julia_mode':    ParamDef('mb2_julia_mode',    PropType.INT,   0,     3,      0,      'Julia Mode',          'Mandelbulb'),
    'mb2_fold_strength': ParamDef('mb2_fold_strength', PropType.FLOAT, 0.0,   3.0,    0.0,    'Fold Strength',       'Mandelbulb'),
    'mb2_fold_type':     ParamDef('mb2_fold_type',     PropType.INT,   0,     3,      0,      'Fold Type',           'Mandelbulb'),
    'mb2_polar_mix':     ParamDef('mb2_polar_mix',     PropType.FLOAT, 0.0,   1.0,    0.0,    'Polar Mix',           'Mandelbulb'),
    'mb2_rot_per_iter':  ParamDef('mb2_rot_per_iter',  PropType.FLOAT, 0.0,   1.0,    0.0,    'Rot Per Iter',        'Mandelbulb'),
    'mb2_abs_x':         ParamDef('mb2_abs_x',         PropType.BOOL,  0,     1,      False,  'Abs X',               'Mandelbulb'),
    'mb2_abs_y':         ParamDef('mb2_abs_y',         PropType.BOOL,  0,     1,      False,  'Abs Y',               'Mandelbulb'),
    'mb2_abs_z':         ParamDef('mb2_abs_z',         PropType.BOOL,  0,     1,      False,  'Abs Z',               'Mandelbulb'),
    'kl_scale':          ParamDef('kl_scale',          PropType.FLOAT, 0.5,   3.0,    1.5,    'Scale',               'Kleinian'),
    'kl_cx':             ParamDef('kl_cx',             PropType.FLOAT, -2.0,  2.0,    0.3,    'Constant X',          'Kleinian'),
    'kl_cy':             ParamDef('kl_cy',             PropType.FLOAT, -2.0,  2.0,    0.3,    'Constant Y',          'Kleinian'),
    'kl_cz':             ParamDef('kl_cz',             PropType.FLOAT, -2.0,  2.0,    0.0,    'Constant Z',          'Kleinian'),
    'kl_fold_limit':     ParamDef('kl_fold_limit',     PropType.FLOAT, 0.1,   3.0,    1.0,    'Fold Limit',          'Kleinian'),
    'kl_sph_radius':     ParamDef('kl_sph_radius',     PropType.FLOAT, 0.1,   2.0,    0.5,    'Sphere Radius',       'Kleinian'),
    'kl_rot_per_iter':   ParamDef('kl_rot_per_iter',   PropType.FLOAT, 0.0,   1.0,    0.0,    'Rot Per Iter',        'Kleinian'),
    'kl_mix_factor':     ParamDef('kl_mix_factor',     PropType.FLOAT, 0.0,   1.0,    0.0,    'Mix Factor',          'Kleinian'),
    'kl_fold_limit_x':   ParamDef('kl_fold_limit_x',   PropType.FLOAT, 0.0,   3.0,    0.0,    'Fold Limit X',        'Kleinian'),
    'kl_fold_limit_y':   ParamDef('kl_fold_limit_y',   PropType.FLOAT, 0.0,   3.0,    0.0,    'Fold Limit Y',        'Kleinian'),
    'kl_fold_limit_z':   ParamDef('kl_fold_limit_z',   PropType.FLOAT, 0.0,   3.0,    0.0,    'Fold Limit Z',        'Kleinian'),
    'kl_julia_mode':     ParamDef('kl_julia_mode',     PropType.INT,   0,     3,      0,      'Julia Mode',          'Kleinian'),
    'kl_offset_x':       ParamDef('kl_offset_x',       PropType.FLOAT, -3.0,  3.0,    0.0,    'Offset X',            'Kleinian'),
    'kl_offset_y':       ParamDef('kl_offset_y',       PropType.FLOAT, -3.0,  3.0,    0.0,    'Offset Y',            'Kleinian'),
    'kl_offset_z':       ParamDef('kl_offset_z',       PropType.FLOAT, -3.0,  3.0,    0.0,    'Offset Z',            'Kleinian'),
    'qj_cx':             ParamDef('qj_cx',             PropType.FLOAT, -2.0,  2.0,    -0.2,   'Constant X',          'Quaternion'),
    'qj_cy':             ParamDef('qj_cy',             PropType.FLOAT, -2.0,  2.0,    0.6,    'Constant Y',          'Quaternion'),
    'qj_cz':             ParamDef('qj_cz',             PropType.FLOAT, -2.0,  2.0,    0.2,    'Constant Z',          'Quaternion'),
    'qj_cw':             ParamDef('qj_cw',             PropType.FLOAT, -2.0,  2.0,    0.2,    'Constant W',          'Quaternion'),
    'qj_w_slice':        ParamDef('qj_w_slice',        PropType.FLOAT, -2.0,  2.0,    0.0,    'W Slice',             'Quaternion'),
    'qj_bailout':        ParamDef('qj_bailout',        PropType.FLOAT, 1.0,   10.0,   4.0,    'Bailout',             'Quaternion'),
    'qj_slice_rot_xw':   ParamDef('qj_slice_rot_xw',   PropType.FLOAT, -3.14, 3.14,   0.0,    'Slice Rot XW',        'Quaternion'),
    'qj_slice_rot_yw':   ParamDef('qj_slice_rot_yw',   PropType.FLOAT, -3.14, 3.14,   0.0,    'Slice Rot YW',        'Quaternion'),
    'qj_slice_rot_zw':   ParamDef('qj_slice_rot_zw',   PropType.FLOAT, -3.14, 3.14,   0.0,    'Slice Rot ZW',        'Quaternion'),
    'warp_enabled':      ParamDef('warp_enabled',      PropType.BOOL,  0,     1,      False,  'Warp Enabled',        'Space Operators'),
    'warp_strength':     ParamDef('warp_strength',     PropType.FLOAT, 0.0,   2.0,    0.3,    'Warp Strength',       'Space Operators'),
    'warp_freq':         ParamDef('warp_freq',         PropType.FLOAT, 0.1,   10.0,   1.0,    'Warp Frequency',      'Space Operators'),
    'warp_type':         ParamDef('warp_type',         PropType.INT,   0,     3,      0,      'Warp Type',           'Space Operators'),
    'twist_axis':        ParamDef('twist_axis',        PropType.INT,   0,     2,      0,      'Twist Axis',          'Space Operators'),
    'twist_amount':      ParamDef('twist_amount',      PropType.FLOAT, 0.0,   5.0,    0.0,    'Twist Amount',        'Space Operators'),
    'fold_mirror_x':     ParamDef('fold_mirror_x',     PropType.BOOL,  0,     1,      False,  'Fold Mirror X',       'Space Operators'),
    'fold_mirror_y':     ParamDef('fold_mirror_y',     PropType.BOOL,  0,     1,      False,  'Fold Mirror Y',       'Space Operators'),
    'fold_mirror_z':     ParamDef('fold_mirror_z',     PropType.BOOL,  0,     1,      False,  'Fold Mirror Z',       'Space Operators'),
    'rep_enabled':       ParamDef('rep_enabled',       PropType.BOOL,  0,     1,      False,  'Repeat Enabled',      'Space Operators'),
    'rep_cell_x':        ParamDef('rep_cell_x',        PropType.FLOAT, 0.5,   20.0,   4.0,    'Repeat Cell X',       'Space Operators'),
    'rep_cell_y':        ParamDef('rep_cell_y',        PropType.FLOAT, 0.5,   20.0,   4.0,    'Repeat Cell Y',       'Space Operators'),
    'rep_cell_z':        ParamDef('rep_cell_z',        PropType.FLOAT, 0.5,   20.0,   4.0,    'Repeat Cell Z',       'Space Operators'),
    'orbit_trap_type':   ParamDef('orbit_trap_type',   PropType.INT,   0,     3,      0,      'Orbit Trap Type',     'Lighting'),
    'de_multiplier':     ParamDef('de_multiplier',     PropType.FLOAT, 0.1,   5.0,    1.0,    'DE Multiplier',       'Lighting'),
    'light_x':           ParamDef('light_x',           PropType.FLOAT, -5.0,  5.0,    1.0,    'Light X',             'Lighting'),
    'light_y':           ParamDef('light_y',           PropType.FLOAT, -5.0,  5.0,    2.0,    'Light Y',             'Lighting'),
    'light_z':           ParamDef('light_z',           PropType.FLOAT, -5.0,  5.0,    1.5,    'Light Z',             'Lighting'),
    'specular_power':    ParamDef('specular_power',    PropType.FLOAT, 1.0,   128.0,  32.0,   'Specular Power',      'Lighting'),
    'specular_strength': ParamDef('specular_strength', PropType.FLOAT, 0.0,   2.0,    0.3,    'Specular Strength',   'Lighting'),
    'ambient':           ParamDef('ambient',           PropType.FLOAT, 0.0,   1.0,    0.2,    'Ambient',             'Lighting'),
    'subsurface':        ParamDef('subsurface',        PropType.FLOAT, 0.0,   1.0,    0.0,    'Subsurface',          'Lighting'),
    'fresnel_power':     ParamDef('fresnel_power',     PropType.FLOAT, 0.5,   15.0,   5.0,    'Fresnel Power',       'Lighting'),
    'light2_x':          ParamDef('light2_x',          PropType.FLOAT, -5.0,  5.0,    -1.0,   'Light2 X',            'Lighting'),
    'light2_y':          ParamDef('light2_y',          PropType.FLOAT, -5.0,  5.0,    -0.5,   'Light2 Y',            'Lighting'),
    'light2_z':          ParamDef('light2_z',          PropType.FLOAT, -5.0,  5.0,    1.0,    'Light2 Z',            'Lighting'),
    'light2_r':          ParamDef('light2_r',          PropType.FLOAT, 0.0,   1.0,    0.2,    'Light2 Red',          'Lighting'),
    'light2_g':          ParamDef('light2_g',          PropType.FLOAT, 0.0,   1.0,    0.3,    'Light2 Green',        'Lighting'),
    'light2_b':          ParamDef('light2_b',          PropType.FLOAT, 0.0,   1.0,    0.5,    'Light2 Blue',         'Lighting'),
    'light2_strength':   ParamDef('light2_strength',   PropType.FLOAT, 0.0,   3.0,    0.0,    'Light2 Strength',     'Lighting'),
    'step_scale':        ParamDef('step_scale',        PropType.FLOAT, 0.1,   1.5,    0.85,   'Step Scale',          'Raymarching'),
    'normal_eps':        ParamDef('normal_eps',        PropType.FLOAT, 0.0001, 0.01, 0.0005, 'Normal Epsilon',      'Raymarching'),
    'reflection':        ParamDef('reflection',        PropType.FLOAT, 0.0,   1.0,    0.0,    'Reflection',          'Raymarching'),
    'max_steps':         ParamDef('max_steps',         PropType.INT,   10,    1000,   200,    'Max Steps',           'Raymarching'),
    'max_dist':          ParamDef('max_dist',          PropType.FLOAT, 10.0,  500.0,  100.0,  'Max Distance',        'Raymarching'),
    'hit_eps':           ParamDef('hit_eps',           PropType.FLOAT, 0.1,   5.0,    0.5,    'Hit Epsilon',         'Raymarching'),
    'shadow_steps':      ParamDef('shadow_steps',      PropType.INT,   4,     256,    32,     'Shadow Steps',        'Raymarching'),
    'shadow_mint':       ParamDef('shadow_mint',       PropType.FLOAT, 0.001, 1.0,    0.02,   'Shadow Min T',        'Raymarching'),
    'shadow_maxt':       ParamDef('shadow_maxt',       PropType.FLOAT, 0.1,   100.0,  10.0,   'Shadow Max T',        'Raymarching'),
    'ao_step_scale':     ParamDef('ao_step_scale',     PropType.FLOAT, 0.1,   3.0,    1.0,    'AO Step Scale',       'Raymarching'),
    'rm_overrelax':      ParamDef('rm_overrelax',      PropType.BOOL,  0,     1,      False,  'Overrelaxation',      'Raymarching'),
    'overrelax_factor':  ParamDef('overrelax_factor',  PropType.FLOAT, 1.0,   2.0,    1.2,    'Overrelax Factor',    'Raymarching'),
    'fov':               ParamDef('fov',               PropType.FLOAT, 0.3,   3.0,    1.5,    'FOV',                 'Camera'),
    'fog_color':         ParamDef('fog_color',         PropType.COLOR, 0.0,   1.0,    (0.02, 0.03, 0.08), 'Fog Color',          'Lighting'),
    'gamma':             ParamDef('gamma',             PropType.FLOAT, 0.5,   4.0,    2.2,    'Gamma',               'Post Process'),
    'exposure':          ParamDef('exposure',          PropType.FLOAT, 0.1,   5.0,    1.0,    'Exposure',            'Post Process'),
    'saturation':        ParamDef('saturation',        PropType.FLOAT, 0.0,   3.0,    1.0,    'Saturation',          'Post Process'),
    'feat_ao':           ParamDef('feat_ao',           PropType.BOOL,  0,     1,      True,   'AO',                  'Features'),
    'feat_shadows':      ParamDef('feat_shadows',      PropType.BOOL,  0,     1,      True,   'Shadows',             'Features'),
    'feat_normals_full': ParamDef('feat_normals_full', PropType.BOOL,  0,     1,      True,   'Full Normals',        'Features'),
    'feat_second_light': ParamDef('feat_second_light', PropType.BOOL,  0,     1,      True,   'Second Light',        'Features'),
    'feat_fog':          ParamDef('feat_fog',          PropType.BOOL,  0,     1,      True,   'Fog',                 'Features'),
    'feat_glow':         ParamDef('feat_glow',         PropType.BOOL,  0,     1,      True,   'Glow',                'Features'),
    'feat_reflection':   ParamDef('feat_reflection',   PropType.BOOL,  0,     1,      True,   'Reflection',          'Features'),
    'feat_subsurface':   ParamDef('feat_subsurface',   PropType.BOOL,  0,     1,      True,   'Subsurface',          'Features'),
    'feat_orbit_trap':   ParamDef('feat_orbit_trap',   PropType.BOOL,  0,     1,      True,   'Orbit Trap',          'Features'),
    'bg_color1':         ParamDef('bg_color1',         PropType.COLOR, 0.0,   1.0,    (0.02, 0.03, 0.08), 'BG Color 1',         'Background'),
    'bg_color2':         ParamDef('bg_color2',         PropType.COLOR, 0.0,   1.0,    (0.0,  0.0,  0.0), 'BG Color 2',         'Background'),
    'bg_mode':           ParamDef('bg_mode',           PropType.INT,   0,     3,      1,      'BG Mode',             'Background'),
    'aa_samples':        ParamDef('aa_samples',        PropType.INT,   1,     9,      1,      'AA Samples',          'Post Process'),
    'render_scale':      ParamDef('render_scale',      PropType.INT,   10,    400,    100,    'Render Scale',        'Performance'),
    'dyn_res_enabled':   ParamDef('dyn_res_enabled',   PropType.BOOL,  0,     1,      False,  'Dynamic Resolution',  'Performance'),
    'dyn_res_target_fps': ParamDef('dyn_res_target_fps', PropType.INT, 15,   144,    60,     'Target FPS',          'Performance'),
    'dyn_res_min_fps':   ParamDef('dyn_res_min_fps',   PropType.INT,   10,    60,     30,     'Min FPS',             'Performance'),
    'vsync':             ParamDef('vsync',             PropType.BOOL,  0,     1,      False,  'VSync',               'Performance'),
    'fps_cap':           ParamDef('fps_cap',           PropType.INT,   0,     240,    0,      'FPS Cap',             'Performance'),
    'vr_mode':           ParamDef('vr_mode',           PropType.BOOL,  0,     1,      False,  'VR Mode',             'VR'),
    'vr_ipd':            ParamDef('vr_ipd',            PropType.FLOAT, 0.04,  0.08,   0.063,  'IPD',                 'VR'),
    'vr_render_scale':   ParamDef('vr_render_scale',   PropType.INT,   50,    200,    100,    'VR Render Scale',     'VR'),
    'vr_dyn_res_enabled': ParamDef('vr_dyn_res_enabled', PropType.BOOL, 0,   1,      False,  'VR Dynamic Resolution', 'VR'),
    'vr_comfort_vignette': ParamDef('vr_comfort_vignette', PropType.BOOL, 0, 1,  False,  'Comfort Vignette',    'VR'),
    'vr_prediction_mult': ParamDef('vr_prediction_mult', PropType.FLOAT, 0.5, 2.0,  1.0,    'Prediction Mult',     'VR'),
    'vr_supersampling':  ParamDef('vr_supersampling',  PropType.INT,   1,     4,      1,      'Supersampling',       'VR'),
    'sph_inv_enabled':   ParamDef('sph_inv_enabled',   PropType.BOOL,  0,     1,      False,  'Sphere Inversion',    'Space Operators'),
    'sph_inv_radius':    ParamDef('sph_inv_radius',    PropType.FLOAT, 0.1,   5.0,    1.0,    'Inversion Radius',    'Space Operators'),
    'sph_inv_cx':        ParamDef('sph_inv_cx',        PropType.FLOAT, -5.0,  5.0,    0.0,    'Inversion Center X',  'Space Operators'),
    'sph_inv_cy':        ParamDef('sph_inv_cy',        PropType.FLOAT, -5.0,  5.0,    0.0,    'Inversion Center Y',  'Space Operators'),
    'sph_inv_cz':        ParamDef('sph_inv_cz',        PropType.FLOAT, -5.0,  5.0,    0.0,    'Inversion Center Z',  'Space Operators'),
    'lattice_fold_enabled': ParamDef('lattice_fold_enabled', PropType.BOOL, 0, 1,  False,  'Lattice Fold',        'Space Operators'),
    'lattice_fold_x':    ParamDef('lattice_fold_x',    PropType.FLOAT, 0.5,   10.0,   2.0,    'Lattice Fold X',      'Space Operators'),
    'lattice_fold_y':    ParamDef('lattice_fold_y',    PropType.FLOAT, 0.5,   10.0,   2.0,    'Lattice Fold Y',      'Space Operators'),
    'lattice_fold_z':    ParamDef('lattice_fold_z',    PropType.FLOAT, 0.5,   10.0,   2.0,    'Lattice Fold Z',      'Space Operators'),
}


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
    'mb2_polar_mix', 'mb2_rot_per_iter',
    'kl_fold_limit_x', 'kl_fold_limit_y', 'kl_fold_limit_z',
    'kl_offset_x', 'kl_offset_y', 'kl_offset_z',
    'qj_cx', 'qj_cy', 'qj_cz', 'qj_cw', 'qj_w_slice', 'qj_bailout',
    'qj_slice_rot_xw', 'qj_slice_rot_yw', 'qj_slice_rot_zw',
    'sph_inv_radius', 'sph_inv_cx', 'sph_inv_cy', 'sph_inv_cz',
    'lattice_fold_x', 'lattice_fold_y', 'lattice_fold_z',
    'warp_strength', 'warp_freq',
    'rep_cell_x', 'rep_cell_y', 'rep_cell_z',
    'mb_scale_x', 'mb_scale_y', 'mb_scale_z',
    'mb_offset_x', 'mb_offset_y', 'mb_offset_z',
    'mb_inversion_radius',
    'ms_offset_x', 'ms_offset_y', 'ms_offset_z',
    'si_scale_x', 'si_scale_y', 'si_scale_z',
    'si_offset_x', 'si_offset_y', 'si_offset_z',
    'si_rot_y',
    'oc_scale_y', 'oc_scale_z',
    'oc_julia_x', 'oc_julia_y', 'oc_julia_z',
    'vr_ipd', 'vr_prediction_mult',
]


INTERP_COLOR_ATTRS = ['color1', 'color2', 'color3', 'bg_color1', 'bg_color2', 'fog_color']


FRACTAL_FIELDS = [
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
    'qj_cx', 'qj_cy', 'qj_cz', 'qj_cw', 'qj_w_slice', 'qj_bailout',
    'qj_slice_rot_xw', 'qj_slice_rot_yw', 'qj_slice_rot_zw',
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


SESSION_EXTRA_FIELDS = ['cam_pos', 'cam_yaw', 'cam_pitch', 'cam_roll', 'player_mode']


def get_param_default(name: str) -> Any:
    if name in PARAM_DEFINITIONS:
        return PARAM_DEFINITIONS[name].default
    return None


def get_param_range(name: str) -> Tuple[float, float]:
    if name in PARAM_DEFINITIONS:
        defn = PARAM_DEFINITIONS[name]
        return (defn.min_val, defn.max_val)
    return (0.0, 1.0)


def get_param_type(name: str) -> PropType:
    if name in PARAM_DEFINITIONS:
        return PARAM_DEFINITIONS[name].prop_type
    return PropType.FLOAT


def get_all_param_names() -> List[str]:
    return list(PARAM_DEFINITIONS.keys())


def get_params_by_category(category: str) -> Dict[str, ParamDef]:
    return {
        name: defn 
        for name, defn in PARAM_DEFINITIONS.items() 
        if defn.category == category
    }


def get_all_categories() -> List[str]:
    categories = set()
    for defn in PARAM_DEFINITIONS.values():
        categories.add(defn.category)
    return sorted(categories)
