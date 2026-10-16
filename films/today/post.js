// The WebGL half of the film. The 2D canvases hold the painted shot(s); this pass puts glass in front of them when the
// shot looks through a window (running drops, fog you can wipe), joins two shots with a transition that behaves like
// wet paint, grades, blooms the lamps, lays the lyric on top and presses everything into one sheet of paper.
// Every output pixel is a function of the uniforms only (time comes in as a number), so frames can be drawn in any order.

const VS = `#version 300 es
in vec2 aPos; out vec2 vUv;
void main(){ vUv = aPos*0.5+0.5; gl_Position = vec4(aPos,0.,1.); }`;

const COMMON = `#version 300 es
precision highp float;
in vec2 vUv; out vec4 o;
uniform vec2 uRes; uniform float uT;
float h11(float p){ p = fract(p*.1031); p *= p+33.33; p *= p+p; return fract(p); }
float h21(vec2 p){ vec3 q = fract(vec3(p.xyx)*.1031); q += dot(q, q.yzx+33.33); return fract((q.x+q.y)*q.z); }
vec2 h22(vec2 p){ vec3 q = fract(vec3(p.xyx)*vec3(.1031,.1030,.0973)); q += dot(q, q.yzx+33.33); return fract((q.xx+q.yz)*q.zy); }
float vn(vec2 p){ vec2 i = floor(p), f = fract(p); f = f*f*(3.-2.*f);
  return mix(mix(h21(i), h21(i+vec2(1,0)), f.x), mix(h21(i+vec2(0,1)), h21(i+vec2(1,1)), f.x), f.y); }
float fbm(vec2 p){ float a = 0., w = .5; for(int i=0;i<5;i++){ a += vn(p)*w; p = p*2.03+vec2(17.3,9.1); w *= .5; } return a; }
`;

// ---------------------------------------------------------------- glass
// Height field of water on a pane: beads that sit and slowly swell, and runners that slide down a lane in stick-slip
// steps, leaving a wet track and a few beads behind. p: pane coords, x 0..aspect, y 0..1 downward.
const GLASS = COMMON + `
uniform sampler2D uTex, uWipe, uMask;
uniform float uRain, uBeads, uFog, uScale, uUseWipe, uUseMask, uRefr;
uniform vec3 uFogCol;

float dome(float d, float r){ float x = clamp(d/r, 0., 1.); return sqrt(1.-x*x); }

// x: water height, y: wetness of the track above a runner
vec2 runners(vec2 p, float t, float cols, float seed){
  float X = p.x*cols, Y = p.y*cols;
  float ci = floor(X), fx = fract(X)-.5;
  float r1 = h11(ci*17.31+seed), r2 = h11(ci*71.7+seed*3.), r3 = h11(ci*7.77+seed*5.);
  float P = mix(5., 11., r2);                               // lane period, in column widths
  float v = mix(.35, 1.1, r1);                              // lanes per second
  float ph = v*t/P*3. + .11*sin(t*(.9+r3)+r1*40.) + .05*sin(t*2.7*(.6+r2)+r3*20.) + r3*7.;   // stick, slip
  float q = Y/P - ph, k = floor(q), f = fract(q);
  float hk = h11(k*3.17+ci*9.7+seed);
  if(hk > uRain) return vec2(0.);                           // this cell's drop has not formed
  float mean = (r3-.5)*.45 + .16*sin(Y*.9+r1*30.) + .07*sin(Y*2.3+r2*11.);  // the path meanders with height on the pane
  float dx = fx-mean, f0 = .82, dy = (f-f0)*P;
  float R = mix(.16, .30, h11(hk*91.3));
  float drop = dome(length(vec2(dx, dy*.72+max(dy,0.)*.5)), R);
  float up = max(-dy, 0.);                                  // distance above the drop
  float wt = R*.55*exp(-up*.22);
  float track = smoothstep(wt, wt*.35, abs(dx))*exp(-up*.34)*step(dy, 0.);
  // beads the drop left on its way
  float g = .55, by = floor(Y/g), bh = h11(by*13.7+ci*5.1+seed);
  vec2 bc = vec2(mean+(bh-.5)*.10, (by+.5)*g);
  float bead = dome(length(vec2(fx, Y)-bc), .05+.07*bh)*step(.45, bh)*step(dy,0.)*exp(-up*.2);
  return vec2(max(drop, bead*.8), track);
}
float beads(vec2 p, float t, float n, float seed){
  vec2 g = p*n, id = floor(g), f = fract(g)-.5;
  vec2 hh = h22(id+seed);
  float life = fract(t*(.02+.03*hh.x)+hh.y*7.);             // swell slowly, then run away
  float sz = h21(id*1.7+seed); float r = (.07+.30*sz*sz)*(.35+.65*life)*smoothstep(1.,.96,life);
  if(h21(id+seed*2.3) > uBeads) return 0.;
  return dome(length(f-(hh-.5)*.55), r);
}
vec2 water(vec2 p, float t){
  vec2 a = runners(p, t, 9.*uScale, 1.), b = runners(p*1.0+vec2(.37,.0), t*1.13, 17.*uScale, 4.);
  float bd = max(beads(p, t, 26.*uScale, 2.), beads(p+3.1, t, 55.*uScale, 7.)*.7);
  return vec2(max(max(a.x, b.x*.8), bd*.75), max(a.y, b.y*.8));
}
void main(){
  vec2 uv = vUv, p = vec2(uv.x*uRes.x/uRes.y, 1.-uv.y);
  vec4 mk = texture(uMask, uv); float mask = uUseMask > .5 ? mk.r*mk.a : 1.;
  if(mask < .004){ o = vec4(texture(uTex, uv).rgb, 1.); return; }
  vec2 w = water(p, uT); float e = 1.5/uRes.y;
  float hx = water(p+vec2(e,0.), uT).x, hy = water(p+vec2(0.,e), uT).x;
  vec2 n = vec2(hx-w.x, hy-w.x)/e*.012;                     // slope of the water surface
  float wet = smoothstep(.02, .25, w.x);
  vec4 wp = texture(uWipe, uv); float clear = uUseWipe > .5 ? wp.r*wp.a : 0.;
  float fog = uFog*(1.-clear)*(1.-.85*clamp(w.y, 0., 1.))*(1.-wet);
  fog *= .82+.36*fbm(p*3.+7.);                              // breath is never even
  // a drop is a small lens: it shows the scene upside down and smaller
  vec2 ruv = uv + vec2(-n.x, n.y)*uRefr*mask + (vUv-.5)*(-.06)*wet*0.;
  float lod = fog*5.6;
  vec3 c = textureLod(uTex, ruv, lod).rgb;
  if(lod > 1.){                                             // soften the mip steps
    float s = exp2(lod)/uRes.y*.6;
    c = (c*2. + textureLod(uTex, ruv+vec2(s,s*.6), lod).rgb + textureLod(uTex, ruv+vec2(-s,s*.4), lod).rgb
       + textureLod(uTex, ruv+vec2(s*.3,-s), lod).rgb + textureLod(uTex, ruv+vec2(-s*.5,-s*.8), lod).rgb)/6.;
  }
  c = mix(c, uFogCol, fog*.42);
  // light on the water: a bright crescent where the surface faces the lamp, a dark rim opposite
  float lit = dot(n, vec2(-.6, -.8));
  c += vec3(1.,.96,.88)*smoothstep(.25, .9, lit)*.55*wet*mask;
  c *= 1.-smoothstep(.2, 1.1, -lit)*.38*wet*mask;
  c *= 1.-.10*clamp(w.y,0.,1.)*mask*(1.-wet);
  o = vec4(mix(texture(uTex, uv).rgb, c, mask), 1.);
}`;

// ---------------------------------------------------------------- join, grade, paper
const COMPOSE = COMMON + `
uniform sampler2D uA, uB, uText, uPaper;
uniform float uMix, uType, uFrame, uBoil, uGrain, uFade;
uniform vec2 uCtr; uniform vec3 uFadeCol;
uniform vec4 uGA[4], uGB[4];       // [sat, warm, gain, lift] [tint rgb, amt] [wave cx, cy, r, sat inside] [bloom, vignette, contrast, mono-tint]

vec3 look(sampler2D tex, vec2 uv, vec2 p, vec4 g[4]){
  vec3 c = texture(tex, uv).rgb;
  // lamp bloom: what is bright and warm spreads
  vec3 b = (textureLod(tex, uv, 4.).rgb*.5 + textureLod(tex, uv, 5.5).rgb*.35 + textureLod(tex, uv, 7.).rgb*.15);
  float bl = max(0., dot(b, vec3(.45,.4,.15))-.52);
  c += b*bl*g[3].x*vec3(1.25,1.0,.62);
  float l = dot(c, vec3(.299,.587,.114));
  float wv = smoothstep(g[2].z+.12, g[2].z-.12, distance(p, g[2].xy) + (fbm(p*4.+3.)-.5)*.22);
  float sat = mix(g[0].x, g[2].w, g[2].z > 0. ? wv : 0.);
  c = mix(vec3(l), c, sat);
  c = mix(c, vec3(l)*vec3(.80,.86,.96), g[3].w*(1.-(g[2].z > 0. ? wv : 0.)));       // the grey of the bridge is a cold grey
  c *= vec3(1.+.10*g[0].y, 1.+.02*g[0].y, 1.-.12*g[0].y);
  c = mix(c, c*g[1].rgb*1.6, g[1].a);
  c = (c-.5)*(1.+g[3].z)+.5;
  c = c*g[0].z + g[0].w;
  return c;
}
void main(){
  vec2 asp = vec2(uRes.x/uRes.y, 1.), p = vec2(vUv.x, 1.-vUv.y)*asp;
  // hand-made pictures never sit perfectly still: a breath of displacement that changes a few times a second
  float k = floor(uFrame/3.);
  vec2 bo = (vec2(vn(p*5.+k*7.13), vn(p*5.+31.7+k*3.71))-.5)*uBoil/uRes;
  vec2 uv = vUv+bo;
  vec3 c;
  if(uMix <= 0.){ c = look(uA, uv, p, uGA); }
  else if(uMix >= 1.){ c = look(uB, uv, p, uGB); }
  else {
    float m, edge = 0.;
    float n = fbm(p*2.6+11.) , n2 = fbm(p*9.+3.);
    if(uType < 1.5){ m = uMix; }                                                // 1 dissolve
    else if(uType < 2.5){                                                       // 2 bloom: the new picture spreads from a point like a wet stain
      float d = distance(p, uCtr*asp)/1.9;
      float v = (uMix*1.75-.32) - d - (n-.5)*.55 - (n2-.5)*.07;
      m = smoothstep(0., .07, v); edge = smoothstep(.0,.035,v)*smoothstep(.16,.035,v);
    } else if(uType < 3.5){                                                     // 3 brush: a loaded brush dragged across, left to right
      float lane = floor(p.y*5.+.3*sin(p.x*3.)), lo = h11(lane*7.7)*.22;
      float v = (uMix*1.6-.25-lo) - p.x/asp.x - (n-.5)*.16 - (n2-.5)*.05;
      m = smoothstep(0., .03, v); edge = smoothstep(.0,.02,v)*smoothstep(.10,.02,v);
    } else if(uType < 4.5){                                                     // 4 iris: a circle opens (the cup, the clock, the umbrella)
      float d = distance(p, uCtr*asp);
      float v = uMix*uMix*2.3 - d - (n2-.5)*.012;
      m = smoothstep(0., .008, v); edge = smoothstep(.0,.006,v)*smoothstep(.03,.006,v);
    } else if(uType < 5.5){                                                     // 5 run: the old picture runs down the sheet like too much water
      float col = fbm(vec2(p.x*7., 3.))*.9 + fbm(vec2(p.x*31., 9.))*.25;
      float v = uMix*2.2 - (1.-vUv.y)*.9 - col + .15;
      m = smoothstep(0., .10, v); edge = smoothstep(.0,.05,v)*smoothstep(.22,.05,v);
    } else if(uType < 6.5){                                                     // 6 iris closing onto a point (old picture shrinks into a circle)
      float d = distance(p, uCtr*asp);
      float v = d - (1.-uMix)*(1.-uMix)*2.3 - (n2-.5)*.012;
      m = smoothstep(0., .008, v); edge = smoothstep(.0,.006,v)*smoothstep(.03,.006,v);
    } else if(uType < 7.5){                                                     // 7 peel: the old picture comes off like a sticky note, from a corner
      // a flat fold whose crease sweeps from the corner uCtr (0..1) across the frame; the lifted part shows its back
      vec2 cn = uCtr*asp, dir = normalize(cn - asp*.5);
      float pm = dot(cn, dir), pn = dot(asp - cn, dir);                         // the corner, and the far corner
      float f = mix(pm + .02, pn - .02, uMix*uMix*(3.-2.*uMix));               // crease position along dir
      float q = dot(p, dir);
      vec2 pr = p + 2.*(f - q)*dir*.86;                                        // where this point would be on the flap (slightly short: paper curls)
      bool lifted = q > f;
      bool flap = !lifted && dot(pr, dir) > f && pr.x > 0. && pr.x < asp.x && pr.y > 0. && pr.y < 1.;
      vec3 a = look(uA, uv, p, uGA), b = look(uB, uv, p, uGB);
      if(lifted) c = b*(1.-.35*exp(-(q-f)*14.));                                 // the new picture, shadowed by the curl
      else c = a*(1.-.25*exp(-(f-q)*30.)*float(!flap));
      if(flap){
        vec2 fu = vec2(pr.x/asp.x, 1.-pr.y);
        vec3 back = mix(vec3(.985,.93,.70), texture(uA, fu).rgb, .12);          // the back of the note: paper, a little of the picture through it
        float sh = smoothstep(0., .25, (f-q));
        c = back*(.78+.22*sh)*(1.-.08*(fbm(p*30.)-.5));
      }
      m = -1.;
    } else if(uType > 8.5){                                                     // 9 square closing: the old picture shrinks into a small square (a window) at uCtr
      vec2 d = p - uCtr*asp; float an = .05*uMix;
      d = mat2(cos(an), -sin(an), sin(an), cos(an))*d;
      float v = max(abs(d.x)*.78, abs(d.y)) - (1.-uMix)*(1.-uMix)*1.9 - .0 - (n2-.5)*.01;
      m = smoothstep(0., .006, v); edge = smoothstep(.0,.005,v)*smoothstep(.022,.005,v);
    } else {                                                                    // 8 square: a small square of the new picture opens out, turned a little
      vec2 d = p - uCtr*asp; float an = .07*(1.-uMix);
      d = mat2(cos(an), -sin(an), sin(an), cos(an))*d;
      float v = uMix*uMix*1.9 - max(abs(d.x), abs(d.y)) - (n2-.5)*.01 - (n-.5)*.006;
      m = smoothstep(0., .006, v); edge = smoothstep(.0,.005,v)*smoothstep(.022,.005,v);
    }
    if(m >= 0.){
      vec3 a = look(uA, uv, p, uGA), b = look(uB, uv, p, uGB);
      c = mix(a, b, m);
      c *= 1.-edge*.42;                                                         // pigment gathers at the wet front
    }
  }
  c = mix(c, uFadeCol, uFade);
  vec4 tx = texture(uText, vUv); c = c*(1.-tx.a) + tx.rgb;                      // premultiplied lyric layer
  vec3 pp = texture(uPaper, vec2(vUv.x, 1.-vUv.y)*uRes/512.).rgb;
  c *= mix(vec3(1.), pp*pp*1.17, .9);                                        // the tooth of the sheet, pressed in a little harder than it was scanned
  float vg = distance(vUv, vec2(.5))*1.25; c *= 1.-uGA[3].y*vg*vg*mix(1., uGB[3].y/max(uGA[3].y,1e-3), clamp(uMix,0.,1.));
  c += (h21(gl_FragCoord.xy+floor(uFrame/2.)*17.)-.5)*uGrain;          // grain on twos: half the cost to encode, and it reads as paper, not video noise
  o = vec4(c, 1.);
}`;

function program(gl, fsSrc) {
  const mk = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s) + '\n' + src.split('\n').map((l, i) => (i + 1) + ': ' + l).join('\n')); return s; };
  const p = gl.createProgram(); gl.attachShader(p, mk(gl.VERTEX_SHADER, VS)); gl.attachShader(p, mk(gl.FRAGMENT_SHADER, fsSrc)); gl.bindAttribLocation(p, 0, 'aPos'); gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
  const u = {}, n = gl.getProgramParameter(p, gl.ACTIVE_UNIFORMS);
  for (let i = 0; i < n; i++) { const info = gl.getActiveUniform(p, i); u[info.name.replace(/\[0\]$/, '')] = gl.getUniformLocation(p, info.name); }
  return { p, u };
}

export const GRADE0 = { sat: 1, warm: 0, gain: 1, lift: 0, tint: [1, 1, 1], tintAmt: 0, wave: [0, 0, 0], satIn: 1, bloom: 0.5, vig: 0.42, contrast: 0.04, cold: 0 };
const gradeArr = (g0) => { const g = { ...GRADE0, ...(g0 || {}) }; return [g.sat, g.warm, g.gain, g.lift, ...g.tint, g.tintAmt, g.wave[0], g.wave[1], g.wave[2], g.satIn, g.bloom, g.vig, g.contrast, g.cold]; };

export function makePost(canvas, W, H, paperImg) {
  const gl = canvas.getContext('webgl2', { preserveDrawingBuffer: true, antialias: false, alpha: false });
  if (!gl) throw new Error('WebGL2 not available');
  const glass = program(gl, GLASS), comp = program(gl, COMPOSE);
  const vb = gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, vb); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  const tex = (mip, repeat) => {
    const t = gl.createTexture(); gl.bindTexture(gl.TEXTURE_2D, t);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, mip ? gl.LINEAR_MIPMAP_LINEAR : gl.LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, repeat ? gl.REPEAT : gl.CLAMP_TO_EDGE);
    return t;
  };
  const up = (t, src, mip, premult = false) => {
    gl.bindTexture(gl.TEXTURE_2D, t); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true); gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, premult);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, src); if (mip) gl.generateMipmap(gl.TEXTURE_2D);
  };
  const tA = tex(true), tB = tex(true), tText = tex(false), tWipe = tex(false), tMask = tex(false), tPaper = tex(false, true);
  gl.bindTexture(gl.TEXTURE_2D, tPaper); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, paperImg);
  const fbo = () => {
    const t = tex(true); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, W, H, 0, gl.RGBA, gl.UNSIGNED_BYTE, null); gl.generateMipmap(gl.TEXTURE_2D);
    const f = gl.createFramebuffer(); gl.bindFramebuffer(gl.FRAMEBUFFER, f); gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, t, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); return { f, t };
  };
  const fA = fbo(), fB = fbo();
  const bind = (unit, t, loc) => { gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, t); gl.uniform1i(loc, unit); };

  // one side of the join: upload the canvas, and if the shot looks through glass, run the glass pass into an FBO
  function side(cv, t, g, T, F) {
    up(T, cv, true);
    if (!g) return T;
    gl.useProgram(glass.p); gl.bindFramebuffer(gl.FRAMEBUFFER, F.f); gl.viewport(0, 0, W, H);
    if (g.wipe) up(tWipe, g.wipe, false);
    if (g.mask) up(tMask, g.mask, false);
    const u = glass.u; bind(0, T, u.uTex); bind(1, tWipe, u.uWipe); bind(2, tMask, u.uMask);
    gl.uniform2f(u.uRes, W, H); gl.uniform1f(u.uT, (g.time ?? t) + (g.seed || 0));      // g.time: the shot can slow the rain down
    gl.uniform1f(u.uRain, g.rain ?? 0.6); gl.uniform1f(u.uBeads, g.beads ?? 0.7); gl.uniform1f(u.uFog, g.fog ?? 0.5); gl.uniform1f(u.uScale, g.scale ?? 1);
    gl.uniform1f(u.uUseWipe, g.wipe ? 1 : 0); gl.uniform1f(u.uUseMask, g.mask ? 1 : 0); gl.uniform1f(u.uRefr, g.refr ?? 1);
    gl.uniform3fv(u.uFogCol, g.fogCol || [0.80, 0.84, 0.86]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindTexture(gl.TEXTURE_2D, F.t); gl.generateMipmap(gl.TEXTURE_2D);
    return F.t;
  }

  // frame({ t, frame, a: {cv, glass, grade}, b: {cv, glass, grade} | null, mix, type, ctr: [x, y] 0..1, text: canvas, fade, fadeCol, boil, grain })
  return function frame(f) {
    const a = side(f.a.cv, f.t, f.a.glass, tA, fA), b = f.b ? side(f.b.cv, f.t, f.b.glass, tB, fB) : a;
    up(tText, f.text, false, true);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null); gl.viewport(0, 0, W, H); gl.useProgram(comp.p);
    const u = comp.u; bind(0, a, u.uA); bind(1, b, u.uB); bind(2, tText, u.uText); bind(3, tPaper, u.uPaper);
    gl.uniform2f(u.uRes, W, H); gl.uniform1f(u.uT, f.t); gl.uniform1f(u.uFrame, f.frame);
    gl.uniform1f(u.uMix, f.b ? f.mix : 0); gl.uniform1f(u.uType, f.type || 1); gl.uniform2fv(u.uCtr, f.ctr || [0.5, 0.5]);
    gl.uniform4fv(u.uGA, gradeArr(f.a.grade)); gl.uniform4fv(u.uGB, gradeArr((f.b || f.a).grade));
    gl.uniform1f(u.uBoil, f.boil ?? 1.3); gl.uniform1f(u.uGrain, f.grain ?? 0.02); gl.uniform1f(u.uFade, f.fade || 0); gl.uniform3fv(u.uFadeCol, f.fadeCol || [0, 0, 0]);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  };
}
