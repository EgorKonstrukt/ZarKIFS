import moderngl

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

vec3 sampleScene(vec2 uv) {
    return texture(u_scene, uv).rgb;
}

vec3 applyFXAA(vec2 uv, vec2 rcpFrame) {
    float spanMax    = max(u_fxaa_span_max, 1.0);
    float reduceMul  = u_fxaa_reduce_mul;
    float reduceMin  = u_fxaa_reduce_min;
    vec3 lumW = vec3(0.299, 0.587, 0.114);
    vec3 nw = sampleScene(uv + vec2(-1.0, -1.0) * rcpFrame);
    vec3 ne = sampleScene(uv + vec2( 1.0, -1.0) * rcpFrame);
    vec3 sw = sampleScene(uv + vec2(-1.0,  1.0) * rcpFrame);
    vec3 se = sampleScene(uv + vec2( 1.0,  1.0) * rcpFrame);
    vec3 m  = sampleScene(uv);
    float lumNW = dot(nw, lumW);
    float lumNE = dot(ne, lumW);
    float lumSW = dot(sw, lumW);
    float lumSE = dot(se, lumW);
    float lumM  = dot(m,  lumW);
    float lumMin = min(lumM, min(min(lumNW, lumNE), min(lumSW, lumSE)));
    float lumMax = max(lumM, max(max(lumNW, lumNE), max(lumSW, lumSE)));
    vec2 dir;
    dir.x = -((lumNW + lumNE) - (lumSW + lumSE));
    dir.y =  ((lumNW + lumSW) - (lumNE + lumSE));
    float dirReduce = max((lumNW + lumNE + lumSW + lumSE) * (0.25 * reduceMul), reduceMin);
    float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
    dir = clamp(dir * rcpDirMin, vec2(-spanMax), vec2(spanMax)) * rcpFrame;
    vec3 rgbA = 0.5 * (sampleScene(uv + dir * (1.0/3.0 - 0.5)) +
                       sampleScene(uv + dir * (2.0/3.0 - 0.5)));
    vec3 rgbB = rgbA * 0.5 + 0.25 * (sampleScene(uv + dir * -0.5) +
                                       sampleScene(uv + dir *  0.5));
    float lumB = dot(rgbB, lumW);
    return (lumB < lumMin || lumB > lumMax) ? rgbA : rgbB;
}

vec3 applyToneAndGrade(vec3 col) {
    col *= u_exposure;
    float lum = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = mix(vec3(lum), col, u_saturation);
    col = pow(clamp(col, 0.0, 1.0), vec3(1.0 / max(u_gamma, 0.1)));
    return col;
}

void main() {
    vec2 uv = u_flip_y != 0 ? vec2(v_uv.x, 1.0 - v_uv.y) : v_uv;
    vec3 col;
    if (u_fxaa_enabled != 0) {
        vec2 rcpFrame = 1.0 / u_resolution;
        col = applyFXAA(uv, rcpFrame);
    } else {
        col = texture(u_scene, uv).rgb;
    }
    col = applyToneAndGrade(col);
    fragColor = vec4(col, 1.0);
}
"""

FXAA_DEFAULTS = {
    'span_max':    8.0,
    'reduce_mul':  0.125,
    'reduce_min':  0.0078125,
}

class PostProcessor:
    def __init__(self, ctx: moderngl.Context, quad_vbo: moderngl.Buffer):
        self._ctx  = ctx
        self._prog = ctx.program(vertex_shader=POST_VERT_SRC,
                                  fragment_shader=POST_FRAG_SRC)
        self._uloc = {name: self._prog[name] for name in self._prog}
        self._vao  = ctx.simple_vertex_array(self._prog, quad_vbo, 'in_position')
        self._set_fxaa_defaults()

    def _set_fxaa_defaults(self):
        self._set('u_fxaa_span_max',   FXAA_DEFAULTS['span_max'])
        self._set('u_fxaa_reduce_mul', FXAA_DEFAULTS['reduce_mul'])
        self._set('u_fxaa_reduce_min', FXAA_DEFAULTS['reduce_min'])

    def _set(self, name, value):
        if name in self._uloc:
            self._uloc[name].value = value

    def render(self, scene_tex: moderngl.Texture, target_fbo: moderngl.Framebuffer,
               gamma: float, exposure: float, saturation: float,
               resolution: tuple, flip_y: bool = False, fxaa: bool = False):
        target_fbo.use()
        scene_tex.use(location=0)
        self._set('u_scene',        0)
        self._set('u_resolution',   (float(resolution[0]), float(resolution[1])))
        self._set('u_gamma',        gamma)
        self._set('u_exposure',     exposure)
        self._set('u_saturation',   saturation)
        self._set('u_flip_y',       1 if flip_y else 0)
        self._set('u_fxaa_enabled', 1 if fxaa else 0)
        self._vao.render(moderngl.TRIANGLE_STRIP)

    def set_fxaa_params(self, span_max: float, reduce_mul: float, reduce_min: float):
        self._set('u_fxaa_span_max',   span_max)
        self._set('u_fxaa_reduce_mul', reduce_mul)
        self._set('u_fxaa_reduce_min', reduce_min)

    def release(self):
        self._vao.release()
        self._prog.release()