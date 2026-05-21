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

uniform vec3  u_cam_pos;
uniform float u_cam_yaw;
uniform float u_cam_pitch;
uniform float u_vel_x;
uniform float u_vel_y;
uniform float u_vel_z;
uniform float u_fps;
uniform int   u_fractal_type;
uniform int   u_iterations;
uniform float u_scale;
uniform int   u_render_scale;
uniform int   u_aa_samples;
uniform int   u_player_mode;
uniform int   u_frame_count;
uniform float u_elapsed;
uniform float u_dyn_res_scale;
uniform int   u_max_steps;
uniform float u_fov;
uniform float u_gravity_mode;

uniform sampler2D u_font_tex;
uniform vec2      u_font_atlas_size;
uniform vec2      u_font_char_size;
uniform int       u_font_first_char;
uniform int       u_font_cols;

float sdSeg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
}

float sdRing(vec2 p, vec2 c, float r) {
    return abs(length(p - c) - r);
}

vec4 blendOver(vec4 dst, vec3 rgb, float a) {
    return vec4(mix(dst.rgb, rgb, clamp(a, 0.0, 1.0)), max(dst.a, a));
}

float panelBg(vec2 px, vec2 tl, vec2 br) {
    vec2 inner = px - tl;
    vec2 size  = br - tl;
    if (inner.x < 0.0 || inner.y < 0.0 || inner.x > size.x || inner.y > size.y) return 0.0;
    float border = min(min(inner.x, inner.y), min(size.x - inner.x, size.y - inner.y));
    return border < 1.5 ? 0.55 : 0.35;
}

float barH(vec2 px, vec2 tl, float w, float h, float t) {
    vec2 local = px - tl;
    if (local.x < 0.0 || local.y < 0.0 || local.y > h) return 0.0;
    float fill = w * clamp(t, 0.0, 1.0);
    if (local.x < fill) return 0.75;
    if (local.x < w)    return 0.10;
    return 0.0;
}

float sampleChar(int code, vec2 cellUV) {
    if (cellUV.x < 0.0 || cellUV.x > 1.0 || cellUV.y < 0.0 || cellUV.y > 1.0) return 0.0;
    int idx = code - u_font_first_char;
    if (idx < 0) return 0.0;
    int col = idx % u_font_cols;
    int row = idx / u_font_cols;
    vec2 atlasCell = vec2(float(col), float(row));
    vec2 uv = (atlasCell + cellUV) * u_font_char_size / u_font_atlas_size;
    return texture(u_font_tex, uv).a;
}

float drawChar(vec2 px, vec2 pos, float cw, float ch, int code) {
    vec2 local = px - pos;
    vec2 cellUV = vec2(local.x / cw, local.y / ch);
    return sampleChar(code, cellUV);
}

float drawMinus(vec2 px, vec2 pos, float cw, float ch) {
    return drawChar(px, pos, cw, ch, 45);
}

float drawDot(vec2 px, vec2 pos, float cw, float ch) {
    return drawChar(px, pos, cw * 0.55, ch, 46);
}

float drawColon(vec2 px, vec2 pos, float cw, float ch) {
    return drawChar(px, pos, cw * 0.55, ch, 58);
}

float renderFloat(vec2 px, vec2 origin, float cw, float ch, float val,
                  int decimals, bool showSign, out float advance) {
    float result = 0.0;
    float x = origin.x;
    float gap = cw * 0.18;

    if (showSign) {
        if (val < 0.0) result += drawMinus(px, vec2(x, origin.y), cw * 0.65, ch);
        x += cw * 0.72 + gap;
    }

    float scale = 1.0;
    for (int i = 0; i < decimals; i++) scale *= 10.0;
    float absVal = abs(val);
    int intScaled = int(absVal * scale + 0.499);

    int digs[8];
    int count = 0;
    int tmp = intScaled;
    for (int i = 0; i < 8; i++) {
        digs[i] = tmp % 10;
        tmp /= 10;
        if (tmp == 0 && i >= decimals) { count = i + 1; break; }
        if (i == 7) { count = 8; break; }
    }
    if (count == 0) count = 1;

    for (int i = count - 1; i >= 0; i--) {
        int pos_from_right = i;
        result += drawChar(px, vec2(x, origin.y), cw, ch, 48 + digs[pos_from_right]);
        x += cw + gap;
        if (decimals > 0 && pos_from_right == decimals) {
            result += drawDot(px, vec2(x, origin.y), cw, ch);
            x += cw * 0.55 + gap;
        }
    }
    advance = x;
    return clamp(result, 0.0, 1.0);
}

float renderLabel(vec2 px, vec2 origin, float cw, float ch, int[8] codes, int len) {
    float result = 0.0;
    float x = origin.x;
    float gap = cw * 0.18;
    for (int i = 0; i < 8; i++) {
        if (i >= len) break;
        result += drawChar(px, vec2(x, origin.y), cw, ch, codes[i]);
        x += cw + gap;
    }
    return clamp(result, 0.0, 1.0);
}

void main() {
    if (u_enabled == 0) { discard; return; }

    vec2 px  = v_uv * u_res;
    vec2 ctr = u_res * 0.5;
    vec4 col = vec4(0.0);

    float crossR = 7.0, crossW = 1.8;
    vec2 dc = abs(px - ctr);
    if ((dc.x < crossR && dc.y < crossW) || (dc.y < crossR && dc.x < crossW)) {
        vec3 cc = u_on_ground == 1 ? vec3(0.15, 1.0, 0.25) : vec3(1.0, 0.55, 0.05);
        col = blendOver(col, cc, 0.95);
    }

    if (u_col_ring_px > 2.0) {
        float dr = sdRing(px, ctr, u_col_ring_px);
        vec3 rc = u_on_ground == 1 ? vec3(0.2, 1.0, 0.3) : vec3(1.0, 0.45, 0.05);
        if (dr < 3.5) col = blendOver(col, rc, clamp(1.0 - dr / 2.5, 0.0, 1.0) * 0.9);
    }

    if (u_gnd_ring_px > 2.0) {
        float dr = sdRing(px, ctr, u_gnd_ring_px);
        if (dr < 3.0) col = blendOver(col, vec3(0.25, 0.6, 1.0), clamp(1.0 - dr / 2.0, 0.0, 1.0) * 0.55);
    }

    vec2 normEnd = ctr + u_norm_dir_ss;
    float dn = sdSeg(px, ctr, normEnd);
    if (dn < 3.5) col = blendOver(col, vec3(0.2, 0.85, 1.0), clamp(1.0 - dn / 2.5, 0.0, 1.0) * 0.92);
    if (length(px - normEnd) < 5.0) col = blendOver(col, vec3(0.2, 0.85, 1.0), 0.95);

    vec2 gravEnd = ctr + u_grav_dir_ss;
    float dg = sdSeg(px, ctr, gravEnd);
    if (dg < 3.5) col = blendOver(col, vec3(1.0, 0.25, 0.25), clamp(1.0 - dg / 2.5, 0.0, 1.0) * 0.92);
    if (length(px - gravEnd) < 5.0) col = blendOver(col, vec3(1.0, 0.25, 0.25), 0.95);

    for (int i = 0; i < u_probe_count; i++) {
        float d = length(px - u_probe_ss[i]);
        if (d < 4.5) {
            vec3 pc = u_probe_sdf[i] < u_probe_radius
                ? vec3(1.0, 0.15, 0.15)
                : vec3(0.15, 0.95, 0.35);
            col = blendOver(col, pc, clamp(1.0 - d / 4.5, 0.0, 1.0) * 0.88);
        }
    }

    float cw = 8.0, ch = 14.0, lh = 16.0;
    float lbw = cw * 5.5;
    float panW = lbw + 130.0;
    float panH = 21.0 * lh + 12.0;
    float panX = 8.0, panY = 8.0;

    float bgA = panelBg(px, vec2(panX, panY), vec2(panX + panW, panY + panH));
    if (bgA > 0.0) col = blendOver(col, vec3(0.03, 0.05, 0.10), bgA);

    float lx = panX + 6.0;
    float vx = lx + lbw + 4.0;
    float row = panY + 7.0;
    float adv = 0.0;

    #define LABEL(px_, orig_, cw_, ch_, a,b,c,d,e,f,g,h, len_) \
        { int codes_[8]; codes_[0]=a; codes_[1]=b; codes_[2]=c; codes_[3]=d; \
          codes_[4]=e; codes_[5]=f; codes_[6]=g; codes_[7]=h; \
          renderLabel(px_, orig_, cw_, ch_, codes_, len_); }

    vec3 lblClr = vec3(0.55, 0.55, 0.65);

    // FPS
    {
        int c[8]; c[0]=70;c[1]=80;c[2]=83;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 3) * lblClr;
        vec3 fpsClr = u_fps >= 55.0 ? vec3(0.2,1.0,0.3) : (u_fps >= 25.0 ? vec3(1.0,0.8,0.1) : vec3(1.0,0.2,0.2));
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, clamp(u_fps,0.0,9999.0), 1, false, adv) * fpsClr;
    }
    row += lh;

    // TIME
    {
        int c[8]; c[0]=84;c[1]=73;c[2]=77;c[3]=69;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 4) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_elapsed, 1, false, adv) * vec3(0.75,0.75,0.3);
    }
    row += lh;

    // POS X
    {
        int c[8]; c[0]=80;c[1]=79;c[2]=83;c[3]=32;c[4]=88;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_cam_pos.x, 3, true, adv) * vec3(1.0,0.55,0.2);
    }
    row += lh;

    // POS Y
    {
        int c[8]; c[0]=80;c[1]=79;c[2]=83;c[3]=32;c[4]=89;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_cam_pos.y, 3, true, adv) * vec3(1.0,0.55,0.2);
    }
    row += lh;

    // POS Z
    {
        int c[8]; c[0]=80;c[1]=79;c[2]=83;c[3]=32;c[4]=90;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_cam_pos.z, 3, true, adv) * vec3(1.0,0.55,0.2);
    }
    row += lh;

    // SPD
    {
        int c[8]; c[0]=83;c[1]=80;c[2]=68;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 3) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_speed, 3, false, adv) * vec3(0.35,0.85,1.0);
    }
    row += lh;

    // VEL X
    {
        int c[8]; c[0]=86;c[1]=69;c[2]=76;c[3]=32;c[4]=88;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_vel_x, 3, true, adv) * vec3(0.5,0.8,1.0);
    }
    row += lh;

    // VEL Y
    {
        int c[8]; c[0]=86;c[1]=69;c[2]=76;c[3]=32;c[4]=89;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_vel_y, 3, true, adv) * vec3(0.5,0.8,1.0);
    }
    row += lh;

    // VEL Z
    {
        int c[8]; c[0]=86;c[1]=69;c[2]=76;c[3]=32;c[4]=90;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_vel_z, 3, true, adv) * vec3(0.5,0.8,1.0);
    }
    row += lh;

    // SDF
    {
        int c[8]; c[0]=83;c[1]=68;c[2]=70;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 3) * lblClr;
        vec3 sdfClr = u_sdf_val < u_radius ? vec3(1.0,0.25,0.25) : vec3(0.85,0.85,0.85);
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_sdf_val, 5, true, adv) * sdfClr;
    }
    row += lh;

    // RAD
    {
        int c[8]; c[0]=82;c[1]=65;c[2]=68;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 3) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_radius, 5, false, adv) * vec3(0.6,0.8,1.0);
    }
    row += lh;

    // GND
    {
        int c[8]; c[0]=71;c[1]=78;c[2]=68;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 3) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_gnd_thresh, 5, false, adv) * vec3(0.5,0.7,1.0);
    }
    row += lh;

    // TYPE
    {
        int c[8]; c[0]=84;c[1]=89;c[2]=80;c[3]=69;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 4) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, float(u_fractal_type), 0, false, adv) * vec3(0.85,0.7,1.0);
    }
    row += lh;

    // ITER
    {
        int c[8]; c[0]=73;c[1]=84;c[2]=69;c[3]=82;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 4) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, float(u_iterations), 0, false, adv) * vec3(0.75,0.75,1.0);
    }
    row += lh;

    // SCALE
    {
        int c[8]; c[0]=83;c[1]=67;c[2]=65;c[3]=76;c[4]=69;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_scale, 2, false, adv) * vec3(0.65,1.0,0.65);
    }
    row += lh;

    // DYRES
    {
        int c[8]; c[0]=68;c[1]=89;c[2]=82;c[3]=69;c[4]=83;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_dyn_res_scale, 1, false, adv) * vec3(0.95,0.95,0.5);
    }
    row += lh;

    // FOV
    {
        int c[8]; c[0]=70;c[1]=79;c[2]=86;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 3) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, u_fov, 2, false, adv) * vec3(0.6,0.95,0.95);
    }
    row += lh;

    // STEPS
    {
        int c[8]; c[0]=83;c[1]=84;c[2]=69;c[3]=80;c[4]=83;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 5) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, float(u_max_steps), 0, false, adv) * vec3(0.8,0.8,0.8);
    }
    row += lh;

    // AA
    {
        int c[8]; c[0]=65;c[1]=65;c[2]=0;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 2) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, float(u_aa_samples), 0, false, adv) * vec3(0.8,0.8,0.8);
    }
    row += lh;

    // GND? (on_ground flag)
    {
        int c[8]; c[0]=71;c[1]=82;c[2]=78;c[3]=68;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 4) * lblClr;
        vec3 groundColor = u_on_ground == 1 ? vec3(0.2,1.0,0.3) : vec3(1.0,0.3,0.3);
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, float(u_on_ground), 0, false, adv) * groundColor;
    }
    row += lh;

    // FRAME
    {
        int c[8]; c[0]=70;c[1]=82;c[2]=77;c[3]=0;c[4]=0;c[5]=0;c[6]=0;c[7]=0;
        col.rgb += renderLabel(px, vec2(lx,row), cw, ch, c, 3) * lblClr;
        col.rgb += renderFloat(px, vec2(vx,row), cw, ch, float(u_frame_count % 10000), 0, false, adv) * vec3(0.45,0.45,0.5);
    }

    // --- bars panel ---
    float bpanH  = 68.0;
    float bpanY  = panY + panH + 6.0;
    float bgA2   = panelBg(px, vec2(panX, bpanY), vec2(panX + panW, bpanY + bpanH));
    if (bgA2 > 0.0) col = blendOver(col, vec3(0.03, 0.04, 0.09), bgA2);

    float bx = panX + 6.0, bw = panW - 12.0, bh = 9.0;
    float by = bpanY + 10.0;

    float fpsNorm = clamp(u_fps / 120.0, 0.0, 1.0);
    vec3 fpsBCol  = mix(vec3(1.0,0.2,0.2), vec3(0.2,1.0,0.3), fpsNorm);
    float fBar = barH(px, vec2(bx,by), bw, bh, fpsNorm);
    if (fBar > 0.0) col = blendOver(col, fpsBCol, fBar);
    by += bh + 6.0;

    float spdNorm = clamp(u_speed / 5.0, 0.0, 1.0);
    float sBar = barH(px, vec2(bx,by), bw, bh, spdNorm);
    if (sBar > 0.0) col = blendOver(col, vec3(0.35,0.85,1.0), sBar);
    by += bh + 6.0;

    float sdfProximity = clamp(1.0 - abs(u_sdf_val) / max(u_gnd_thresh * 3.0, 0.0001), 0.0, 1.0);
    vec3 sdfBCol = mix(vec3(0.2,0.8,1.0), vec3(1.0,0.25,0.25), sdfProximity);
    float sBar2 = barH(px, vec2(bx,by), bw, bh, sdfProximity);
    if (sBar2 > 0.0) col = blendOver(col, sdfBCol, sBar2);

    // --- compass ---
    float cpanW = 70.0, cpanH = 70.0;
    float cpanX = u_res.x - cpanW - 8.0, cpanY = 8.0;
    float bgA3  = panelBg(px, vec2(cpanX, cpanY), vec2(cpanX + cpanW, cpanY + cpanH));
    if (bgA3 > 0.0) col = blendOver(col, vec3(0.03, 0.05, 0.10), bgA3);

    vec2 mc  = vec2(cpanX + cpanW * 0.5, cpanY + cpanH * 0.5);
    float mr = cpanW * 0.36;

    float dcirc = abs(length(px - mc) - mr);
    if (dcirc < 1.2) col = blendOver(col, vec3(0.3,0.3,0.55), 0.65);

    vec2 fwdDir  = normalize(vec2(sin(u_cam_yaw), -cos(u_cam_yaw)));
    vec2 fwdEnd  = mc + fwdDir * mr * 0.82;
    float dfwd   = sdSeg(px, mc, fwdEnd);
    if (dfwd < 2.5) col = blendOver(col, vec3(1.0,0.4,0.15), clamp(1.0 - dfwd / 2.5, 0.0, 1.0) * 0.95);
    if (length(px - fwdEnd) < 3.5) col = blendOver(col, vec3(1.0,0.6,0.3), 0.9);

    vec2 northEnd = mc + vec2(0.0, -mr * 0.75);
    float dnorth  = sdSeg(px, mc, northEnd);
    if (dnorth < 1.1) col = blendOver(col, vec3(0.4,0.5,1.0), clamp(1.0 - dnorth / 1.1, 0.0, 1.0) * 0.5);

    for (int ti = 0; ti < 4; ti++) {
        float angle  = float(ti) * 1.5707963;
        vec2  tdir   = vec2(sin(angle), -cos(angle));
        vec2  tstart = mc + tdir * (mr - 3.0);
        vec2  tend   = mc + tdir * mr;
        float dtick  = sdSeg(px, tstart, tend);
        if (dtick < 1.0) col = blendOver(col, vec3(0.5,0.5,0.6), 0.5);
    }

    float pitchBarH  = cpanH * 0.55;
    float pitchBarW  = 5.0;
    float pbX = cpanX + cpanW - 9.0 - pitchBarW;
    float pbY = cpanY + (cpanH - pitchBarH) * 0.5;
    float bgBar = panelBg(px, vec2(pbX - 1.0, pbY - 1.0), vec2(pbX + pitchBarW + 1.0, pbY + pitchBarH + 1.0));
    if (bgBar > 0.0) col = blendOver(col, vec3(0.05,0.05,0.10), bgBar * 0.8);

    float pitchNorm = clamp((u_cam_pitch + 1.5707963) / 3.1415927, 0.0, 1.0);
    float pFill = barH(px, vec2(pbX, pbY), pitchBarW, pitchBarH, pitchNorm);
    if (pFill > 0.0) col = blendOver(col, vec3(0.5,1.0,0.5), pFill * 0.75);
    float pitchDot = abs(length(px - vec2(pbX + pitchBarW * 0.5, pbY + pitchBarH * (1.0 - pitchNorm))) - 2.5);
    if (pitchDot < 1.5) col = blendOver(col, vec3(1.0,1.0,1.0), 0.9);

    fragColor = vec4(col.rgb, col.a > 0.0 ? 1.0 : 0.0);
}
