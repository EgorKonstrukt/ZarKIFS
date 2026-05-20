#version 330 core
in vec2 in_position;
out vec2 v_uv;
void main() {
    v_uv = in_position;
    gl_Position = vec4(in_position, 0.0, 1.0);
}
