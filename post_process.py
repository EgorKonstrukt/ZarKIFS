import moderngl
from pathlib import Path as _Path
_SHADER_DIR = _Path(__file__).parent

POST_VERT_SRC = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

POST_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform vec2  u_resolution;
uniform float u_gamma;
uniform float u_exposure;
uniform float u_saturation;
uniform int   u_flip_y;
uniform int   u_fxaa_enabled;
uniform float u_fxaa_span_max;
uniform float u_fxaa_reduce_mul;
uniform float u_fxaa_reduce_min;
vec3 applyFXAA(vec2 uv, vec2 rcpFrame) {
    float spanMax   = max(u_fxaa_span_max, 1.0);
    const vec3 lumW = vec3(0.299, 0.587, 0.114);
    vec3 nw = texture(u_scene, uv + vec2(-1.0, -1.0) * rcpFrame).rgb;
    vec3 ne = texture(u_scene, uv + vec2( 1.0, -1.0) * rcpFrame).rgb;
    vec3 sw = texture(u_scene, uv + vec2(-1.0,  1.0) * rcpFrame).rgb;
    vec3 se = texture(u_scene, uv + vec2( 1.0,  1.0) * rcpFrame).rgb;
    vec3 m  = texture(u_scene, uv).rgb;
    float lumNW = dot(nw, lumW);
    float lumNE = dot(ne, lumW);
    float lumSW = dot(sw, lumW);
    float lumSE = dot(se, lumW);
    float lumM  = dot(m,  lumW);
    float lumMin = min(lumM, min(min(lumNW, lumNE), min(lumSW, lumSE)));
    float lumMax = max(lumM, max(max(lumNW, lumNE), max(lumSW, lumSE)));
    if (lumMax - lumMin < lumMax * 0.125) return m;
    vec2 dir;
    dir.x = -((lumNW + lumNE) - (lumSW + lumSE));
    dir.y =  ((lumNW + lumSW) - (lumNE + lumSE));
    float dirReduce = max((lumNW + lumNE + lumSW + lumSE) * (0.25 * u_fxaa_reduce_mul), u_fxaa_reduce_min);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-spanMax), vec2(spanMax)) * rcpFrame;
    vec3 rgbA = 0.5 * (texture(u_scene, uv + dir * (1.0/3.0 - 0.5)).rgb +
                       texture(u_scene, uv + dir * (2.0/3.0 - 0.5)).rgb);
    vec3 rgbB = rgbA * 0.5 + 0.25 * (texture(u_scene, uv + dir * -0.5).rgb +
                                       texture(u_scene, uv + dir *  0.5).rgb);
    float lumB = dot(rgbB, lumW);
    return (lumB < lumMin || lumB > lumMax) ? rgbA : rgbB;
}
vec3 tonemap(vec3 col) {
    col *= u_exposure;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation);
    return pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1)));
}
void main() {
    vec2 uv = (u_flip_y != 0) ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    vec3 col = (u_fxaa_enabled != 0)
        ? applyFXAA(uv, 1.0 / u_resolution)
        : texture(u_scene, uv).rgb;
    fragColor = vec4(tonemap(col), 1.0);
}
"""

TAA_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_current;
uniform sampler2D u_history;
uniform vec2  u_resolution;
uniform vec2  u_jitter;
uniform float u_blend_alpha;
uniform int   u_taa_first;
vec3 clipHistory(vec3 history, vec3 c, vec2 uv, vec2 rcp) {
    vec3 n = texture(u_current, uv + vec2( 0.0,  1.0) * rcp).rgb;
    vec3 s = texture(u_current, uv + vec2( 0.0, -1.0) * rcp).rgb;
    vec3 e = texture(u_current, uv + vec2( 1.0,  0.0) * rcp).rgb;
    vec3 w = texture(u_current, uv + vec2(-1.0,  0.0) * rcp).rgb;
    vec3 cmin = min(c, min(min(n, s), min(e, w)));
    vec3 cmax = max(c, max(max(n, s), max(e, w)));
    vec3 center = (cmin + cmax) * 0.5;
    vec3 extent = (cmax - cmin) * 0.625;
    return clamp(history, center - extent, center + extent);
}
void main() {
    vec2 rcpRes = 1.0 / u_resolution;
    vec2 uvCur  = v_uv - u_jitter;
    vec3 cur    = texture(u_current, uvCur).rgb;
    if (u_taa_first != 0) { fragColor = vec4(cur, 1.0); return; }
    vec3 hist = clipHistory(texture(u_history, v_uv).rgb, cur, uvCur, rcpRes);
    fragColor = vec4(mix(hist, cur, u_blend_alpha), 1.0);
}
"""

TONEMAP_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform float u_gamma;
uniform float u_exposure;
uniform float u_saturation;
uniform int   u_flip_y;
void main() {
    vec2 uv  = (u_flip_y != 0) ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    vec3 col = texture(u_scene, uv).rgb * u_exposure;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1))), 1.0);
}
"""

TONEMAP_SHARPEN_FRAG_SRC = """
#version 330 core
in vec2 v_uv;
out vec4 fragColor;
uniform sampler2D u_scene;
uniform vec2  u_resolution;
uniform float u_gamma;
uniform float u_exposure;
uniform float u_saturation;
uniform int   u_flip_y;
uniform float u_sharpen_strength;
void main() {
    vec2 uv  = (u_flip_y != 0) ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    vec2 rcp = 1.0 / u_resolution;
    vec3 c   = texture(u_scene, uv).rgb;
    vec3 n   = texture(u_scene, uv + vec2( 0.0,  1.0) * rcp).rgb;
    vec3 s   = texture(u_scene, uv + vec2( 0.0, -1.0) * rcp).rgb;
    vec3 e   = texture(u_scene, uv + vec2( 1.0,  0.0) * rcp).rgb;
    vec3 w   = texture(u_scene, uv + vec2(-1.0,  0.0) * rcp).rgb;
    vec3 lap = c * 4.0 - n - s - e - w;
    vec3 col = clamp(c + lap * u_sharpen_strength, 0.0, 1.0) * u_exposure;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation);
    fragColor = vec4(pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1))), 1.0);
}
"""

FSR_EASU_FRAG_SRC = (_SHADER_DIR / "shaders/postprocess/fsr_easu_frag.glsl").read_text(encoding="utf-8")

FSR_RCAS_FRAG_SRC = (_SHADER_DIR / "shaders/postprocess/fsr_rcas_frag.glsl").read_text(encoding="utf-8")

TAAU_UPSAMPLE_FRAG_SRC = FSR_EASU_FRAG_SRC

FXAA_DEFAULTS        = {'span_max': 8.0, 'reduce_mul': 0.125, 'reduce_min': 0.0078125}
TAA_SHARPEN_DEFAULT  = 0.15
TAAU_SHARPEN_DEFAULT = 0.25
TAAU_SCALE_DEFAULT   = 0.5
TAAU_VALID_SCALES    = (0.25, 0.333, 0.5, 0.667, 0.75)

FSR_EASU_SHARPNESS_DEFAULT = 0.5
FSR_RCAS_SHARPNESS_DEFAULT = 0.25
FSR_QUALITY_PRESETS = {
    'Ultra Quality': {'scale': 0.9, 'easu': 0.35, 'rcas': 0.15},
    'Quality':       {'scale': 0.667,'easu': 0.50, 'rcas': 0.25},
    'Balanced':      {'scale': 0.585,'easu': 0.65, 'rcas': 0.30},
    'Performance':   {'scale': 0.5,  'easu': 0.80, 'rcas': 0.35},
    'Ultra Perf':    {'scale': 0.333,'easu': 1.00, 'rcas': 0.40},
}


def _halton(base, n):
    out = []
    for i in range(1, n + 1):
        f, r, j = 1.0, 0.0, i
        while j > 0:
            f /= base; r += f * (j % base); j //= base
        out.append(r)
    return out


TAA_JITTER_SEQ = tuple((x - 0.5, y - 0.5) for x, y in zip(_halton(2, 16), _halton(3, 16)))
_TAA_SEQ_LEN   = len(TAA_JITTER_SEQ)


def _make_fbo(ctx, w, h):
    tex = ctx.texture((w, h), 3, dtype='f4')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    return {'tex': tex, 'fbo': ctx.framebuffer(color_attachments=[tex])}


def _free_fbo(b):
    if b is None:
        return
    for k in ('fbo', 'tex'):
        try: b[k].release()
        except Exception: pass


class PostProcessor:
    TAA_BLEND_ALPHA = 0.1

    def __init__(self, ctx: moderngl.Context, quad_vbo: moderngl.Buffer):
        self._ctx = ctx
        self._prog_fxaa  = ctx.program(vertex_shader=POST_VERT_SRC, fragment_shader=POST_FRAG_SRC)
        self._prog_taa   = ctx.program(vertex_shader=POST_VERT_SRC, fragment_shader=TAA_FRAG_SRC)
        self._prog_tone  = ctx.program(vertex_shader=POST_VERT_SRC, fragment_shader=TONEMAP_FRAG_SRC)
        self._prog_sharp = ctx.program(vertex_shader=POST_VERT_SRC, fragment_shader=TONEMAP_SHARPEN_FRAG_SRC)
        self._prog_fsr_easu = ctx.program(vertex_shader=POST_VERT_SRC, fragment_shader=FSR_EASU_FRAG_SRC)
        self._prog_fsr_rcas = ctx.program(vertex_shader=POST_VERT_SRC, fragment_shader=FSR_RCAS_FRAG_SRC)
        self._vao_fxaa      = ctx.simple_vertex_array(self._prog_fxaa,     quad_vbo, 'in_position')
        self._vao_taa       = ctx.simple_vertex_array(self._prog_taa,      quad_vbo, 'in_position')
        self._vao_tone      = ctx.simple_vertex_array(self._prog_tone,     quad_vbo, 'in_position')
        self._vao_sharp     = ctx.simple_vertex_array(self._prog_sharp,    quad_vbo, 'in_position')
        self._vao_fsr_easu  = ctx.simple_vertex_array(self._prog_fsr_easu, quad_vbo, 'in_position')
        self._vao_fsr_rcas  = ctx.simple_vertex_array(self._prog_fsr_rcas, quad_vbo, 'in_position')
        pf = self._prog_fxaa
        self._uf_scene       = pf['u_scene']
        self._uf_resolution  = pf['u_resolution']
        self._uf_gamma       = pf['u_gamma']
        self._uf_exposure    = pf['u_exposure']
        self._uf_saturation  = pf['u_saturation']
        self._uf_flip_y      = pf['u_flip_y']
        self._uf_fxaa_en     = pf['u_fxaa_enabled']
        self._uf_span_max    = pf['u_fxaa_span_max']
        self._uf_reduce_mul  = pf['u_fxaa_reduce_mul']
        self._uf_reduce_min  = pf['u_fxaa_reduce_min']
        self._uf_span_max.value   = FXAA_DEFAULTS['span_max']
        self._uf_reduce_mul.value = FXAA_DEFAULTS['reduce_mul']
        self._uf_reduce_min.value = FXAA_DEFAULTS['reduce_min']
        pt = self._prog_taa
        self._ut_current    = pt['u_current']
        self._ut_history    = pt['u_history']
        self._ut_resolution = pt['u_resolution']
        self._ut_jitter     = pt['u_jitter']
        self._ut_blend      = pt['u_blend_alpha']
        self._ut_first      = pt['u_taa_first']
        po = self._prog_tone
        self._uo_scene      = po['u_scene']
        self._uo_gamma      = po['u_gamma']
        self._uo_exposure   = po['u_exposure']
        self._uo_saturation = po['u_saturation']
        self._uo_flip_y     = po['u_flip_y']
        ps = self._prog_sharp
        self._us_scene      = ps['u_scene']
        self._us_resolution = ps['u_resolution']
        self._us_gamma      = ps['u_gamma']
        self._us_exposure   = ps['u_exposure']
        self._us_saturation = ps['u_saturation']
        self._us_flip_y     = ps['u_flip_y']
        self._us_strength   = ps['u_sharpen_strength']
        pe = self._prog_fsr_easu
        self._ue_scene      = pe['u_scene']
        self._ue_src_res    = pe['u_src_resolution']
        self._ue_flip_y     = pe['u_flip_y']
        self._ue_easu_sharp = pe['u_easu_sharpness']
        pr = self._prog_fsr_rcas
        self._ur_scene      = pr['u_scene']
        self._ur_resolution = pr['u_resolution']
        self._ur_gamma      = pr['u_gamma']
        self._ur_exposure   = pr['u_exposure']
        self._ur_saturation = pr['u_saturation']
        self._ur_flip_y     = pr['u_flip_y']
        self._ur_rcas_sharp = pr['u_rcas_sharpness']
        self._taa_size      = (0, 0)
        self._taa_bufs      = [None, None]
        self._taa_cur       = 0
        self._taa_first     = True
        self._taa_frame     = 0
        self._taau_scale    = TAAU_SCALE_DEFAULT
        self._taau_size     = (0, 0)
        self._taau_bufs     = [None, None]
        self._taau_cur      = 0
        self._taau_first    = True
        self._taau_frame    = 0
        self._taa_sharpen   = TAA_SHARPEN_DEFAULT
        self._taau_sharpen  = TAAU_SHARPEN_DEFAULT
        self._fsr_easu_sharpness = FSR_EASU_SHARPNESS_DEFAULT
        self._fsr_rcas_sharpness = FSR_RCAS_SHARPNESS_DEFAULT
        self._fsr_easu_buf       = None
        self._fsr_easu_buf_size  = (0, 0)

    def _ensure_taa_bufs(self, w: int, h: int):
        if self._taa_size == (w, h):
            return
        for b in self._taa_bufs:
            _free_fbo(b)
        self._taa_bufs  = [_make_fbo(self._ctx, w, h) for _ in range(2)]
        self._taa_size  = (w, h)
        self._taa_cur   = 0
        self._taa_first = True

    def _ensure_taau_bufs(self, sw: int, sh: int):
        if self._taau_size == (sw, sh):
            return
        for b in self._taau_bufs:
            _free_fbo(b)
        self._taau_bufs  = [_make_fbo(self._ctx, sw, sh) for _ in range(2)]
        self._taau_size  = (sw, sh)
        self._taau_cur   = 0
        self._taau_first = True

    def render(self,
               scene_tex: moderngl.Texture,
               target_fbo: moderngl.Framebuffer,
               gamma: float,
               exposure: float,
               saturation: float,
               resolution: tuple,
               flip_y: bool = False,
               fxaa: bool = False,
               taa: bool = False,
               taa_sharpen: bool = False,
               taau: bool = False):
        if taau:
            self._render_taau(scene_tex, target_fbo, gamma, exposure, saturation, resolution, flip_y)
        elif taa:
            self._render_taa(scene_tex, target_fbo, gamma, exposure, saturation, resolution, flip_y, taa_sharpen)
        else:
            self._render_fxaa(scene_tex, target_fbo, gamma, exposure, saturation, resolution, flip_y, fxaa)

    def _render_fxaa(self, scene_tex, target_fbo, gamma, exposure, saturation, resolution, flip_y, fxaa_on):
        rw, rh = resolution
        target_fbo.use()
        scene_tex.use(location=0)
        self._uf_scene.value      = 0
        self._uf_resolution.value = (float(rw), float(rh))
        self._uf_gamma.value      = gamma
        self._uf_exposure.value   = exposure
        self._uf_saturation.value = saturation
        self._uf_flip_y.value     = 1 if flip_y else 0
        self._uf_fxaa_en.value    = 1 if fxaa_on else 0
        self._vao_fxaa.render(moderngl.TRIANGLE_STRIP)

    def _render_taa(self, scene_tex, target_fbo, gamma, exposure, saturation, resolution, flip_y, sharpen):
        rw, rh   = resolution
        self._ensure_taa_bufs(rw, rh)
        cur_idx  = self._taa_cur
        hist_idx = 1 - cur_idx
        cur_buf  = self._taa_bufs[cur_idx]
        hist_buf = self._taa_bufs[hist_idx]
        jx, jy   = TAA_JITTER_SEQ[self._taa_frame % _TAA_SEQ_LEN]
        cur_buf['fbo'].use()
        scene_tex.use(location=0)
        hist_buf['tex'].use(location=1)
        self._ut_current.value    = 0
        self._ut_history.value    = 1
        self._ut_resolution.value = (float(rw), float(rh))
        self._ut_jitter.value     = (jx / rw, jy / rh)
        self._ut_blend.value      = self.TAA_BLEND_ALPHA
        self._ut_first.value      = 1 if self._taa_first else 0
        self._vao_taa.render(moderngl.TRIANGLE_STRIP)
        target_fbo.use()
        cur_buf['tex'].use(location=0)
        if sharpen:
            self._us_scene.value      = 0
            self._us_resolution.value = (float(rw), float(rh))
            self._us_gamma.value      = gamma
            self._us_exposure.value   = exposure
            self._us_saturation.value = saturation
            self._us_flip_y.value     = 1 if flip_y else 0
            self._us_strength.value   = self._taa_sharpen
            self._vao_sharp.render(moderngl.TRIANGLE_STRIP)
        else:
            self._uo_scene.value      = 0
            self._uo_gamma.value      = gamma
            self._uo_exposure.value   = exposure
            self._uo_saturation.value = saturation
            self._uo_flip_y.value     = 1 if flip_y else 0
            self._vao_tone.render(moderngl.TRIANGLE_STRIP)
        self._taa_cur   = hist_idx
        self._taa_first = False
        self._taa_frame += 1

    def _ensure_fsr_easu_buf(self, w: int, h: int):
        if self._fsr_easu_buf_size == (w, h):
            return
        _free_fbo(self._fsr_easu_buf)
        tex = self._ctx.texture((w, h), 3, dtype='f4')
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self._fsr_easu_buf = {'tex': tex, 'fbo': self._ctx.framebuffer(color_attachments=[tex])}
        self._fsr_easu_buf_size = (w, h)

    def _render_taau(self, scene_tex, target_fbo, gamma, exposure, saturation, resolution, flip_y):
        rw, rh = resolution
        sw     = max(1, int(rw * self._taau_scale))
        sh     = max(1, int(rh * self._taau_scale))
        self._ensure_taau_bufs(sw, sh)
        self._ensure_fsr_easu_buf(rw, rh)
        cur_idx  = self._taau_cur
        hist_idx = 1 - cur_idx
        cur_buf  = self._taau_bufs[cur_idx]
        hist_buf = self._taau_bufs[hist_idx]
        jx, jy   = TAA_JITTER_SEQ[self._taau_frame % _TAA_SEQ_LEN]
        cur_buf['fbo'].use()
        scene_tex.use(location=0)
        hist_buf['tex'].use(location=1)
        self._ut_current.value    = 0
        self._ut_history.value    = 1
        self._ut_resolution.value = (float(sw), float(sh))
        self._ut_jitter.value     = (jx / sw, jy / sh)
        self._ut_blend.value      = self.TAA_BLEND_ALPHA
        self._ut_first.value      = 1 if self._taau_first else 0
        self._vao_taa.render(moderngl.TRIANGLE_STRIP)
        easu_buf = self._fsr_easu_buf
        easu_buf['fbo'].use()
        cur_buf['tex'].use(location=0)
        self._ue_scene.value      = 0
        self._ue_src_res.value    = (float(sw), float(sh))
        self._ue_flip_y.value     = 0
        self._ue_easu_sharp.value = self._fsr_easu_sharpness
        self._vao_fsr_easu.render(moderngl.TRIANGLE_STRIP)
        target_fbo.use()
        easu_buf['tex'].use(location=0)
        self._ur_scene.value      = 0
        self._ur_resolution.value = (float(rw), float(rh))
        self._ur_gamma.value      = gamma
        self._ur_exposure.value   = exposure
        self._ur_saturation.value = saturation
        self._ur_flip_y.value     = 1 if flip_y else 0
        self._ur_rcas_sharp.value = self._fsr_rcas_sharpness
        self._vao_fsr_rcas.render(moderngl.TRIANGLE_STRIP)
        self._taau_cur   = hist_idx
        self._taau_first = False
        self._taau_frame += 1

    def get_taau_render_size(self, output_w: int, output_h: int) -> tuple:
        return (max(1, int(output_w * self._taau_scale)),
                max(1, int(output_h * self._taau_scale)))

    def set_fxaa_params(self, span_max: float, reduce_mul: float, reduce_min: float):
        self._uf_span_max.value   = span_max
        self._uf_reduce_mul.value = reduce_mul
        self._uf_reduce_min.value = reduce_min

    def set_taa_blend(self, alpha: float):
        self.TAA_BLEND_ALPHA = max(0.01, min(1.0, alpha))

    def set_taa_sharpen(self, strength: float):
        self._taa_sharpen = max(0.0, min(1.0, strength))

    def set_taau_sharpen(self, strength: float):
        self._taau_sharpen = max(0.0, min(1.0, strength))

    def set_taau_scale(self, scale: float):
        self._taau_scale = min(TAAU_VALID_SCALES, key=lambda x: abs(x - scale))
        self._taau_size  = (0, 0)
        self._taau_first = True
        self._taau_frame = 0
        self._fsr_easu_buf_size = (0, 0)

    def set_fsr_easu_sharpness(self, v: float):
        self._fsr_easu_sharpness = max(0.0, min(1.0, v))

    def set_fsr_rcas_sharpness(self, v: float):
        self._fsr_rcas_sharpness = max(0.0, min(1.0, v))

    def apply_fsr_preset(self, preset_name: str):
        if preset_name not in FSR_QUALITY_PRESETS:
            return
        p = FSR_QUALITY_PRESETS[preset_name]
        self.set_taau_scale(p['scale'])
        self.set_fsr_easu_sharpness(p['easu'])
        self.set_fsr_rcas_sharpness(p['rcas'])

    def get_fsr_info(self) -> dict:
        sw = max(1, int(100 * self._taau_scale))
        return {
            'render_pct':     int(round(self._taau_scale * 100)),
            'easu_sharpness': self._fsr_easu_sharpness,
            'rcas_sharpness': self._fsr_rcas_sharpness,
        }

    def reset_taa(self):
        self._taa_first  = True
        self._taa_frame  = 0
        self._taau_first = True
        self._taau_frame = 0

    def get_taa_jitter(self, scene_w: int, scene_h: int) -> tuple:
        jx, jy = TAA_JITTER_SEQ[self._taa_frame % _TAA_SEQ_LEN]
        return (jx / scene_w * 2.0, jy / scene_h * 2.0)

    def get_taau_jitter(self, output_w: int, output_h: int) -> tuple:
        sw = max(1, int(output_w * self._taau_scale))
        sh = max(1, int(output_h * self._taau_scale))
        jx, jy = TAA_JITTER_SEQ[self._taau_frame % _TAA_SEQ_LEN]
        return (jx / sw * 2.0, jy / sh * 2.0)

    def release(self):
        for vao in (self._vao_fxaa, self._vao_taa, self._vao_tone, self._vao_sharp, self._vao_fsr_easu, self._vao_fsr_rcas):
            try: vao.release()
            except Exception: pass
        for prog in (self._prog_fxaa, self._prog_taa, self._prog_tone, self._prog_sharp, self._prog_fsr_easu, self._prog_fsr_rcas):
            try: prog.release()
            except Exception: pass
        for b in self._taa_bufs + self._taau_bufs:
            _free_fbo(b)
        _free_fbo(self._fsr_easu_buf)



FSR2_ACCUMULATE_FRAG_SRC = (_SHADER_DIR / "shaders/postprocess/fsr2_accumulate_frag.glsl").read_text(encoding="utf-8")
FSR2_RECONSTRUCT_FRAG_SRC = (_SHADER_DIR / "shaders/postprocess/fsr2_reconstruct_frag.glsl").read_text(encoding="utf-8")
MOTION_VECTORS_FRAG_SRC = (_SHADER_DIR / "shaders/postprocess/motion_vectors_frag.glsl").read_text(encoding="utf-8")

MOTION_VEC_VERT_SRC = """
#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position * 0.5 + 0.5;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
"""

FSR2_DEPTH_THRESHOLD_DEFAULT   = 0.005
FSR2_REACTIVE_SCALE_DEFAULT    = 2.0
FSR2_BLEND_ALPHA_DEFAULT       = 0.1
FSR2_RCAS_SHARPNESS_DEFAULT    = 0.2
FSR2_VALID_SCALES              = (0.5, 0.585, 0.667, 0.77)
FSR2_QUALITY_PRESETS = {
    'Ultra Quality': {'scale': 0.77,  'rcas': 0.15, 'reactive': 1.5},
    'Quality':       {'scale': 0.667, 'rcas': 0.20, 'reactive': 2.0},
    'Balanced':      {'scale': 0.585, 'rcas': 0.25, 'reactive': 2.0},
    'Performance':   {'scale': 0.5,   'rcas': 0.30, 'reactive': 2.5},
}


def _make_fbo_rgba16f(ctx, w, h):
    tex = ctx.texture((w, h), 4, dtype='f2')
    tex.filter = (moderngl.LINEAR, moderngl.LINEAR)
    tex.repeat_x = False
    tex.repeat_y = False
    return {'tex': tex, 'fbo': ctx.framebuffer(color_attachments=[tex])}


def _make_depth_tex(ctx, w, h):
    tex = ctx.texture((w, h), 1, dtype='f4')
    tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
    tex.repeat_x = False
    tex.repeat_y = False
    return tex


class FSR2Renderer:
    def __init__(self, ctx: moderngl.Context, quad_vbo: moderngl.Buffer):
        self._ctx  = ctx
        self._vert = POST_VERT_SRC
        self._prog_accum = ctx.program(
            vertex_shader=self._vert,
            fragment_shader=FSR2_ACCUMULATE_FRAG_SRC,
        )
        self._prog_recon = ctx.program(
            vertex_shader=self._vert,
            fragment_shader=FSR2_RECONSTRUCT_FRAG_SRC,
        )
        self._prog_mv = ctx.program(
            vertex_shader=MOTION_VEC_VERT_SRC,
            fragment_shader=MOTION_VECTORS_FRAG_SRC,
        )
        self._vao_accum = ctx.simple_vertex_array(self._prog_accum, quad_vbo, 'in_position')
        self._vao_recon = ctx.simple_vertex_array(self._prog_recon, quad_vbo, 'in_position')
        self._vao_mv    = ctx.simple_vertex_array(self._prog_mv,    quad_vbo, 'in_position')
        self._scale         = 0.667
        self._rcas          = FSR2_RCAS_SHARPNESS_DEFAULT
        self._reactive      = FSR2_REACTIVE_SCALE_DEFAULT
        self._depth_thresh  = FSR2_DEPTH_THRESHOLD_DEFAULT
        self._blend_alpha   = FSR2_BLEND_ALPHA_DEFAULT
        self._render_size   = (0, 0)
        self._output_size   = (0, 0)
        self._accum_bufs    = [None, None]
        self._cur_idx       = 0
        self._first_frame   = True
        self._frame         = 0
        self._depth_cur     = None
        self._depth_prev    = None
        self._depth_size    = (0, 0)
        self._mv_buf        = None
        self._mv_size       = (0, 0)
        self._cam_pos       = (0.0, 0.0, 0.0)
        self._cam_fwd       = (0.0, 0.0, 1.0)
        self._cam_right     = (1.0, 0.0, 0.0)
        self._cam_up        = (0.0, 1.0, 0.0)
        self._cam_fov       = 1.5
        self._prev_cam_pos   = (0.0, 0.0, 0.0)
        self._prev_cam_fwd   = (0.0, 0.0, 1.0)
        self._prev_cam_right = (1.0, 0.0, 0.0)
        self._prev_cam_up    = (0.0, 1.0, 0.0)
        self._prev_cam_fov   = 1.5
        n = 16
        def halton(base, count):
            out = []
            for i in range(1, count + 1):
                f, r, j = 1.0, 0.0, i
                while j > 0:
                    f /= base; r += f * (j % base); j //= base
                out.append(r - 0.5)
            return out
        xs = halton(2, n)
        ys = halton(3, n)
        self._jitter_seq = tuple(zip(xs, ys))

    def get_render_size(self, out_w: int, out_h: int):
        return (max(1, int(out_w * self._scale)),
                max(1, int(out_h * self._scale)))

    def get_jitter(self, out_w: int, out_h: int):
        rw, rh = self.get_render_size(out_w, out_h)
        jx, jy = self._jitter_seq[self._frame % len(self._jitter_seq)]
        return (jx / rw * 2.0, jy / rh * 2.0)

    def _ensure_bufs(self, rw: int, rh: int, ow: int, oh: int):
        if self._render_size == (rw, rh) and self._output_size == (ow, oh):
            return
        for b in self._accum_bufs:
            _free_fbo(b)
        self._accum_bufs  = [_make_fbo_rgba16f(self._ctx, rw, rh) for _ in range(2)]
        self._render_size = (rw, rh)
        self._output_size = (ow, oh)
        self._first_frame = True
        if self._depth_cur is not None:
            self._depth_cur.release()
            self._depth_prev.release()
        self._depth_cur  = _make_depth_tex(self._ctx, rw, rh)
        self._depth_prev = _make_depth_tex(self._ctx, rw, rh)
        self._depth_size = (rw, rh)
        if self._mv_buf is not None:
            _free_fbo(self._mv_buf)
        mv_tex = self._ctx.texture((rw, rh), 2, dtype='f2')
        mv_tex.filter   = (moderngl.NEAREST, moderngl.NEAREST)
        mv_tex.repeat_x = False
        mv_tex.repeat_y = False
        self._mv_buf  = {'tex': mv_tex, 'fbo': self._ctx.framebuffer(color_attachments=[mv_tex])}
        self._mv_size = (rw, rh)

    def set_camera_matrices(self,
                            pos, fwd, right, up, fov: float):
        self._prev_cam_pos   = self._cam_pos
        self._prev_cam_fwd   = self._cam_fwd
        self._prev_cam_right = self._cam_right
        self._prev_cam_up    = self._cam_up
        self._prev_cam_fov   = self._cam_fov
        self._cam_pos   = tuple(pos)
        self._cam_fwd   = tuple(fwd)
        self._cam_right = tuple(right)
        self._cam_up    = tuple(up)
        self._cam_fov   = float(fov)

    def _render_motion_vectors(self, depth_tex: moderngl.Texture, rw: int, rh: int,
                               jitter_uv: tuple):
        self._mv_buf['fbo'].use()
        self._ctx.viewport = (0, 0, rw, rh)
        depth_tex.use(location=0)
        pm = self._prog_mv
        pm['u_depth'].value          = 0
        pm['u_render_size'].value    = (float(rw), float(rh))
        pm['u_jitter'].value         = jitter_uv
        pm['u_cam_pos'].value        = self._cam_pos
        pm['u_cam_fwd'].value        = self._cam_fwd
        pm['u_cam_right'].value      = self._cam_right
        pm['u_cam_up'].value         = self._cam_up
        pm['u_fov'].value            = self._cam_fov
        pm['u_prev_cam_pos'].value   = self._prev_cam_pos
        pm['u_prev_cam_fwd'].value   = self._prev_cam_fwd
        pm['u_prev_cam_right'].value = self._prev_cam_right
        pm['u_prev_cam_up'].value    = self._prev_cam_up
        pm['u_prev_fov'].value       = self._prev_cam_fov
        self._vao_mv.render(moderngl.TRIANGLE_STRIP)

    def render(self,
               scene_tex: moderngl.Texture,
               target_fbo: moderngl.Framebuffer,
               gamma: float,
               exposure: float,
               saturation: float,
               output_size: tuple,
               flip_y: bool = False,
               depth_tex: moderngl.Texture = None):
        ow, oh = output_size
        rw, rh = self.get_render_size(ow, oh)
        self._ensure_bufs(rw, rh, ow, oh)
        cur_idx  = self._cur_idx
        hist_idx = 1 - cur_idx
        cur_buf  = self._accum_bufs[cur_idx]
        hist_buf = self._accum_bufs[hist_idx]
        jx, jy    = self._jitter_seq[self._frame % len(self._jitter_seq)]
        jitter_uv = (jx / rw, jy / rh)
        active_depth = depth_tex if depth_tex is not None else self._depth_cur
        if not self._first_frame:
            self._render_motion_vectors(active_depth, rw, rh, jitter_uv)
        cur_buf['fbo'].use()
        self._ctx.viewport = (0, 0, rw, rh)
        scene_tex.use(location=0)
        hist_buf['tex'].use(location=1)
        active_depth.use(location=2)
        self._depth_prev.use(location=3)
        mv_tex = self._mv_buf['tex'] if not self._first_frame else active_depth
        mv_tex.use(location=4)
        pa = self._prog_accum
        pa['u_current'].value         = 0
        pa['u_history'].value         = 1
        pa['u_depth'].value           = 2
        pa['u_prev_depth'].value      = 3
        pa['u_motion'].value          = 4
        pa['u_render_size'].value     = (float(rw), float(rh))
        pa['u_jitter'].value          = jitter_uv
        pa['u_blend_alpha'].value     = self._blend_alpha
        pa['u_first_frame'].value     = 1 if self._first_frame else 0
        pa['u_depth_threshold'].value = self._depth_thresh
        self._vao_accum.render(moderngl.TRIANGLE_STRIP)
        if depth_tex is None:
            self._depth_cur, self._depth_prev = self._depth_prev, self._depth_cur
        target_fbo.use()
        self._ctx.viewport = (0, 0, ow, oh)
        cur_buf['tex'].use(location=0)
        pr = self._prog_recon
        pr['u_accumulated'].value    = 0
        pr['u_output_size'].value    = (float(ow), float(oh))
        pr['u_rcas_sharpness'].value = self._rcas
        pr['u_exposure'].value       = exposure
        pr['u_saturation'].value     = saturation
        pr['u_gamma'].value          = gamma
        pr['u_flip_y'].value         = 1 if flip_y else 0
        self._vao_recon.render(moderngl.TRIANGLE_STRIP)
        self._cur_idx     = hist_idx
        self._first_frame = False
        self._frame       += 1

    def reset(self):
        self._first_frame = True
        self._frame       = 0

    def set_scale(self, scale: float):
        self._scale       = min(FSR2_VALID_SCALES, key=lambda x: abs(x - scale))
        self._render_size = (0, 0)
        self._first_frame = True
        self._frame       = 0

    def set_rcas_sharpness(self, v: float):
        self._rcas = max(0.0, min(1.0, v))

    def set_reactive_scale(self, v: float):
        self._reactive = max(0.5, min(5.0, v))

    def set_depth_threshold(self, v: float):
        self._depth_thresh = max(1e-4, v)

    def set_blend_alpha(self, v: float):
        self._blend_alpha = max(0.05, min(0.5, v))

    def apply_preset(self, name: str):
        if name not in FSR2_QUALITY_PRESETS:
            return
        p = FSR2_QUALITY_PRESETS[name]
        self.set_scale(p['scale'])
        self.set_rcas_sharpness(p['rcas'])
        self.set_reactive_scale(p['reactive'])

    def get_info(self) -> dict:
        return {
            'render_pct':    int(round(self._scale * 100)),
            'rcas':          self._rcas,
            'reactive':      self._reactive,
            'depth_thresh':  self._depth_thresh,
            'blend_alpha':   self._blend_alpha,
        }

    def release(self):
        for b in self._accum_bufs:
            _free_fbo(b)
        if self._mv_buf is not None:
            _free_fbo(self._mv_buf)
        if self._depth_cur is not None:
            try: self._depth_cur.release()
            except Exception: pass
        if self._depth_prev is not None:
            try: self._depth_prev.release()
            except Exception: pass
        for prog in (self._prog_accum, self._prog_recon, self._prog_mv):
            try: prog.release()
            except Exception: pass