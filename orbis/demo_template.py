"""HTML/CSS/JS template for the Orbis interactive console (body-only).

Kept as a Python string so :mod:`orbis.demo` can inject the embedded data at the
``/*DATA*/`` marker.  The document is intentionally body-only (a leading
``<style>`` then markup then ``<script>``) so it renders as a standalone local
file *and* as a wrapped Artifact.
"""

HTML_TEMPLATE = r"""<style>
:root{
  --bg:#0d0f14; --panel:#161a22; --panel2:#1e232e; --line:#2a303c;
  --ink:#e6e9ef; --muted:#828b9e; --accent:#ff4d4d; --accent-dim:#ff4d4d33;
  --good:#3ddc97; --hold:#f5c451;
  --mono:ui-monospace,"SF Mono",Menlo,Consolas,monospace;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
}
@media (prefers-color-scheme:light){
  :root{--bg:#f4f5f8;--panel:#ffffff;--panel2:#eef0f4;--line:#dcdfe7;
    --ink:#161a22;--muted:#5b6474;--accent:#e5322f;--accent-dim:#e5322f22;}
}
:root[data-theme="dark"]{--bg:#0d0f14;--panel:#161a22;--panel2:#1e232e;--line:#2a303c;
  --ink:#e6e9ef;--muted:#828b9e;--accent:#ff4d4d;--accent-dim:#ff4d4d33;}
:root[data-theme="light"]{--bg:#f4f5f8;--panel:#ffffff;--panel2:#eef0f4;--line:#dcdfe7;
  --ink:#161a22;--muted:#5b6474;--accent:#e5322f;--accent-dim:#e5322f22;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
  line-height:1.5;-webkit-font-smoothing:antialiased}
.wrap{max-width:1080px;margin:0 auto;padding:0 20px}
img.px{image-rendering:pixelated;display:block}
a{color:var(--accent)}
h1,h2,h3{text-wrap:balance;margin:0}
.mono{font-family:var(--mono)}
.eyebrow{font-family:var(--mono);font-size:11px;letter-spacing:.18em;
  text-transform:uppercase;color:var(--accent)}
.muted{color:var(--muted)}

/* header */
header{border-bottom:1px solid var(--line);padding:22px 0;position:sticky;top:0;
  background:color-mix(in srgb,var(--bg) 88%,transparent);backdrop-filter:blur(8px);z-index:9}
.brand{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap}
.brand .logo{font-weight:800;font-size:22px;letter-spacing:-.01em}
.brand .logo b{color:var(--accent)}
.brand .tag{color:var(--muted);font-size:13px;font-family:var(--mono)}
.chips{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
.chip{font-family:var(--mono);font-size:11px;color:var(--muted);border:1px solid var(--line);
  border-radius:999px;padding:4px 10px}
.chip b{color:var(--ink);font-weight:600}
.themebtn{position:absolute;right:20px;top:22px;background:var(--panel);border:1px solid var(--line);
  color:var(--muted);border-radius:8px;padding:6px 10px;font-family:var(--mono);font-size:12px;cursor:pointer}

section{padding:52px 0;border-bottom:1px solid var(--line)}
.sechead{display:flex;flex-direction:column;gap:8px;margin-bottom:26px}
.sechead h2{font-size:26px;letter-spacing:-.01em}
.sechead p{margin:0;color:var(--muted);max-width:62ch}

/* console */
.console{display:grid;grid-template-columns:1.15fr .85fr;gap:22px}
@media(max-width:820px){.console{grid-template-columns:1fr}}
.stage{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
.screen{position:relative;background:#000;border-radius:10px;overflow:hidden;
  aspect-ratio:1/1;display:flex;align-items:center;justify-content:center}
.screen img{width:100%;height:100%}
.badge{position:absolute;top:10px;left:10px;font-family:var(--mono);font-size:11px;
  padding:5px 9px;border-radius:6px;background:#0009;color:#fff;border:1px solid #fff2}
.badge.on{border-color:var(--accent);color:#fff;background:color-mix(in srgb,var(--accent) 30%,#000)}
.badge.ok{border-color:var(--good);color:var(--good);background:#0009}
.hud{display:flex;justify-content:space-between;gap:10px;margin-top:12px;
  font-family:var(--mono);font-size:12px;color:var(--muted)}
.hud b{color:var(--ink);font-weight:600}

/* timeline */
.timeline{position:relative;height:46px;margin-top:14px;user-select:none}
.track{position:absolute;top:16px;left:0;right:0;height:8px;background:var(--panel2);
  border-radius:5px;overflow:hidden}
.track .fill{position:absolute;left:0;top:0;bottom:0;background:var(--accent);width:0}
.ticks{position:absolute;top:12px;left:0;right:0;height:16px}
.tick{position:absolute;width:1px;height:16px;background:var(--line);top:0}
.marker{position:absolute;top:6px;height:28px;width:2px;background:var(--accent)}
.marker::after{content:"switch";position:absolute;top:-14px;left:-16px;font-family:var(--mono);
  font-size:9px;color:var(--accent);white-space:nowrap}
.scrub{position:absolute;inset:0;width:100%;margin:0;opacity:0;cursor:pointer}
.transport{display:flex;align-items:center;gap:10px;margin-top:8px}
.btn{background:var(--panel2);border:1px solid var(--line);color:var(--ink);border-radius:8px;
  padding:7px 14px;font-family:var(--mono);font-size:12px;cursor:pointer}
.btn:hover{border-color:var(--accent)}
.seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.seg button{background:transparent;border:0;color:var(--muted);padding:7px 12px;
  font-family:var(--mono);font-size:12px;cursor:pointer}
.seg button[aria-pressed="true"]{background:var(--accent);color:#fff}

.explain{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:18px}
.explain h3{font-size:15px;margin-bottom:8px}
.explain p{color:var(--muted);font-size:14px;margin:0 0 12px}
.kv{display:grid;grid-template-columns:auto 1fr;gap:6px 14px;font-family:var(--mono);font-size:12px}
.kv .k{color:var(--muted)} .kv .v{color:var(--ink);text-align:right}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:12px}
.dot{display:inline-block;width:9px;height:9px;border-radius:2px;vertical-align:middle;margin-right:5px}

/* gallery */
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:10px}
.card .screen{aspect-ratio:1/1}
.card .cap{font-family:var(--mono);font-size:11px;color:var(--muted);margin-top:8px;
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.card .cap b{color:var(--ink)}

/* sr */
.srrow{display:grid;grid-template-columns:1fr 1fr;gap:18px;align-items:start}
@media(max-width:640px){.srrow{grid-template-columns:1fr}}
.srrow figure{margin:0}
.srrow figcaption{font-family:var(--mono);font-size:12px;color:var(--muted);margin-top:8px}
.srrow .screen{aspect-ratio:1/1}

/* metrics */
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:14px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:16px}
.tile .lab{font-family:var(--mono);font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.1em}
.tile .num{font-size:30px;font-weight:700;margin-top:6px;font-variant-numeric:tabular-nums}
.tile .bar{height:6px;border-radius:4px;background:var(--panel2);margin-top:10px;overflow:hidden}
.tile .bar i{display:block;height:100%;background:var(--good)}
table.cases{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:12px;margin-top:18px}
table.cases th,table.cases td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--line)}
table.cases th:first-child,table.cases td:first-child{text-align:left;color:var(--ink)}
table.cases th{color:var(--muted);font-weight:500;text-transform:uppercase;font-size:10px;letter-spacing:.08em}
.scrollx{overflow-x:auto}

/* pipeline svg */
.diagram{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:20px}
.diagram svg{width:100%;height:auto;display:block}

footer{padding:40px 0;color:var(--muted);font-size:13px}
footer .mono{font-size:12px}
@media (prefers-reduced-motion:reduce){*{animation:none!important;transition:none!important}}
</style>

<header>
  <div class="wrap">
    <button class="themebtn" id="themeBtn">◐ theme</button>
    <div class="brand">
      <span class="logo">Visko <b>Orbis</b></span>
      <span class="tag">a live model · reference implementation</span>
    </div>
    <div class="chips" id="chips"></div>
  </div>
</header>

<main class="wrap">

<section id="switch">
  <div class="sechead">
    <span class="eyebrow">the live contract</span>
    <h2>Change the prompt while it runs</h2>
    <p>A prompt update is admitted at the next <em>uncommitted</em> chunk boundary
       and never revises delivered frames. Scrub past the marker: before it, the
       switched stream is byte-for-byte identical to the committed baseline; after
       it, the future is redirected. The badge is computed live from the embedded
       pixels — it is not a caption.</p>
  </div>
  <div class="console">
    <div class="stage">
      <div class="screen">
        <img class="px" id="swImg" alt="generated frame">
        <div class="badge" id="swBadge">delivered</div>
      </div>
      <div class="hud">
        <span>chunk <b id="swChunk">0</b> · frame <b id="swFrame">0</b></span>
        <span>prompt <b id="swPrompt">—</b> · <b id="swVer">v0</b></span>
      </div>
      <div class="timeline" id="swTimeline">
        <div class="ticks" id="swTicks"></div>
        <div class="track"><div class="fill" id="swFill"></div></div>
        <div class="marker" id="swMarker"></div>
        <input class="scrub" id="swScrub" type="range" min="0" max="0" value="0" aria-label="scrub">
      </div>
      <div class="transport">
        <button class="btn" id="swPlay">▶ play</button>
        <div class="seg" role="group" aria-label="stream">
          <button id="segSwitch" aria-pressed="true">live switch</button>
          <button id="segBase" aria-pressed="false">committed baseline</button>
        </div>
      </div>
    </div>
    <div class="explain">
      <h3>What you're looking at</h3>
      <p id="swExplain">A red circle is generated chunk by chunk. At the marked
        boundary a new instruction — <b>“a blue square moving up”</b> — is queued;
        it takes effect only from that chunk onward.</p>
      <div class="kv" id="swKv"></div>
      <div class="legend">
        <span><span class="dot" style="background:var(--muted)"></span>committed / delivered</span>
        <span><span class="dot" style="background:var(--accent)"></span>after switch (uncommitted)</span>
      </div>
    </div>
  </div>
</section>

<section id="gallery">
  <div class="sechead">
    <span class="eyebrow">text → video · one unified interface</span>
    <h2>Prompt-controlled rollouts</h2>
    <p>The same chunk-wise generator handles every prompt. Subject, color and
       motion are read from the instruction; state (position, velocity) is carried
       across chunks by the bounded history + memory.</p>
  </div>
  <div class="grid" id="gallery"></div>
</section>

<section id="modes">
  <div class="sechead">
    <span class="eyebrow">continuation interfaces</span>
    <h2>Image-to-video &amp; super-resolution</h2>
    <p>I2V seeds the first chunk from a reference frame with empty history, then
       continues. A separate streaming super-resolution stage refines each decoded
       chunk to a higher resolution — presentation detail, added without waiting
       for the full video.</p>
  </div>
  <div class="srrow">
    <figure>
      <div class="stage"><div class="screen">
        <img class="px" id="i2vImg" alt="i2v frame">
        <div class="badge ok" id="i2vBadge">I2V</div>
      </div></div>
      <figcaption>reference frame → continued rollout (I2V)</figcaption>
    </figure>
    <figure>
      <div class="srrow" style="gap:12px">
        <div><div class="stage"><div class="screen"><img class="px" id="srLo" alt="low res"></div></div>
          <figcaption id="srLoCap">native</figcaption></div>
        <div><div class="stage"><div class="screen"><img class="px" id="srHi" alt="high res"></div></div>
          <figcaption id="srHiCap">super-resolved</figcaption></div>
      </div>
    </figure>
  </div>
</section>

<section id="pipeline">
  <div class="sechead">
    <span class="eyebrow">how one chunk is made</span>
    <h2>The chunk-wise streaming loop</h2>
    <p>Each chunk reads the active instruction, a reference, a bounded history
       window and fixed-capacity persistent memory; a few-step rectified-flow
       student denoises the latent chunk, which is decoded, super-resolved and
       delivered. Older frames are consolidated into memory, so per-chunk cost is
       independent of how long the rollout has been running.</p>
  </div>
  <div class="diagram" id="diagram"></div>
</section>

<section id="metrics">
  <div class="sechead">
    <span class="eyebrow">evaluation on the toy benchmark</span>
    <h2>Prompt alignment &amp; stability</h2>
    <p>Because the world is known, we measure following directly and treat a whole
       rollout as the unit of analysis. Higher is better.</p>
  </div>
  <div class="tiles" id="tiles"></div>
  <div class="scrollx"><table class="cases" id="cases"></table></div>
</section>

</main>

<footer class="wrap">
  <div class="mono" id="cfgLine"></div>
  <p style="margin-top:10px;max-width:70ch">A faithful, CPU-runnable reference implementation of the mechanisms in
  <b>Visko Orbis 1.0</b> (chunk-wise latent rectified flow, bounded multi-scale memory, few-step distillation,
  live versioned prompt switching, streaming super-resolution), at a toy scale. Not the trained production system.</p>
</footer>

<script>
const DATA = /*DATA*/;

/* ---------- theme ---------- */
const root=document.documentElement;
document.getElementById('themeBtn').onclick=()=>{
  const cur=root.getAttribute('data-theme')||(matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light');
  root.setAttribute('data-theme',cur==='dark'?'light':'dark');
};

/* ---------- chips ---------- */
const cfg=DATA.config;
document.getElementById('chips').innerHTML=[
  ['native',cfg.native],['super-res',cfg.sr_native],
  ['chunk',cfg.chunk_frames+' frames'],['sampler',cfg.student_steps+'-step student'],
  ['memory',cfg.memory_tokens+' tokens'],['chunk latency','~'+cfg.latency_ms+' ms'],
  ['params',(cfg.params/1e6).toFixed(2)+'M'],
].map(([k,v])=>`<span class="chip">${k} <b>${v}</b></span>`).join('');
document.getElementById('cfgLine').textContent=
  `native ${cfg.native} · sr ${cfg.sr_native} · chunk ${cfg.chunk_frames}f · `+
  `${cfg.student_steps}-step distilled${cfg.distilled?'':' (undistilled)'} · `+
  `memory ${cfg.memory_tokens} tokens · history ${cfg.history_frames}f · `+
  `${(cfg.params/1e6).toFixed(2)}M params · ~${cfg.latency_ms} ms/chunk`;

/* ---------- generic looping player ---------- */
function loopPlayer(img,frames,fps){
  let i=0; const n=frames.length;
  img.src=frames[0];
  setInterval(()=>{i=(i+1)%n; img.src=frames[i];},1000/fps);
}

/* ---------- switch demo ---------- */
const SW=DATA.switch;
const swImg=document.getElementById('swImg'),swScrub=document.getElementById('swScrub');
const swFill=document.getElementById('swFill'),swBadge=document.getElementById('swBadge');
let swTrack='switched', swI=0, swPlaying=true;
const N=SW.switched.length;
swScrub.max=N-1;
// ticks per chunk
const ticks=document.getElementById('swTicks');
for(let f=0;f<N;f+=SW.chunk_frames){
  const t=document.createElement('div');t.className='tick';t.style.left=(f/(N-1)*100)+'%';ticks.appendChild(t);
}
document.getElementById('swMarker').style.left=(SW.boundary_frame/(N-1)*100)+'%';
document.getElementById('swKv').innerHTML=[
  ['base prompt',SW.prompt],['switch to',SW.switch],
  ['boundary','chunk '+SW.boundary_chunk],['streams','baseline vs live'],
].map(([k,v])=>`<span class="k">${k}</span><span class="v">${v}</span>`).join('');

function swRender(){
  const frames=SW[swTrack];
  swImg.src=frames[swI];
  swScrub.value=swI;
  swFill.style.width=(swI/(N-1)*100)+'%';
  const m=SW.meta[swI];
  document.getElementById('swChunk').textContent=m.chunk;
  document.getElementById('swFrame').textContent=swI;
  document.getElementById('swPrompt').textContent=(swTrack==='baseline')?SW.prompt:m.prompt;
  document.getElementById('swVer').textContent='v'+((swTrack==='baseline')?0:m.version);
  const identical=SW.baseline[swI]===SW.switched[swI];
  if(swTrack==='switched' && !identical){
    swBadge.textContent='switched · diverged from baseline';
    swBadge.className='badge on';
  }else{
    swBadge.textContent=identical?'identical to baseline ✓':'delivered';
    swBadge.className='badge ok';
  }
}
function swTick(){ if(swPlaying){swI=(swI+1)%N; swRender();} }
swScrub.addEventListener('input',()=>{swPlaying=false;swI=+swScrub.value;setPlay(false);swRender();});
function setPlay(p){swPlaying=p;document.getElementById('swPlay').textContent=p?'❚❚ pause':'▶ play';}
document.getElementById('swPlay').onclick=()=>setPlay(!swPlaying);
function pick(track){swTrack=track;
  document.getElementById('segSwitch').setAttribute('aria-pressed',track==='switched');
  document.getElementById('segBase').setAttribute('aria-pressed',track==='baseline');swRender();}
document.getElementById('segSwitch').onclick=()=>pick('switched');
document.getElementById('segBase').onclick=()=>pick('baseline');
setInterval(swTick,1000/SW.fps); swRender();

/* ---------- gallery ---------- */
const gal=document.getElementById('gallery');
DATA.scenes.forEach(s=>{
  const el=document.createElement('div');el.className='card';
  el.innerHTML=`<div class="screen"><img class="px" alt="${s.prompt}"></div>
    <div class="cap"><b>${s.mode}</b> · ${s.prompt}</div>`;
  gal.appendChild(el);
  loopPlayer(el.querySelector('img'),s.frames,s.fps);
});

/* ---------- i2v + sr ---------- */
loopPlayer(document.getElementById('i2vImg'),DATA.i2v.frames,DATA.i2v.fps);
loopPlayer(document.getElementById('srLo'),DATA.sr.lr,DATA.sr.fps);
loopPlayer(document.getElementById('srHi'),DATA.sr.hr,DATA.sr.fps);
document.getElementById('srLoCap').textContent=`native ${DATA.sr.lw}×${DATA.sr.lh}`;
document.getElementById('srHiCap').textContent=`super-resolved ${DATA.sr.hw}×${DATA.sr.hh} (${DATA.sr.scale}×)`;

/* ---------- metrics ---------- */
const agg=DATA.metrics.aggregate;
const tileDefs=[['color_acc','color match'],['shape_acc','shape match'],
  ['direction_ok','motion dir'],['stability','temporal stability']];
document.getElementById('tiles').innerHTML=tileDefs.map(([k,lab])=>{
  const v=agg[k]; const pct=Math.round(v*100);
  return `<div class="tile"><div class="lab">${lab}</div>
    <div class="num">${pct}%</div><div class="bar"><i style="width:${pct}%"></i></div></div>`;
}).join('');
const rows=DATA.metrics.per_case;
document.getElementById('cases').innerHTML=
  `<tr><th>prompt</th><th>color</th><th>shape</th><th>dir</th><th>stability</th><th>motion</th></tr>`+
  rows.map(r=>`<tr><td>${r.prompt}</td>
    <td>${r.color_acc.toFixed(2)}</td><td>${r.shape_acc.toFixed(2)}</td>
    <td>${r.direction_ok.toFixed(2)}</td><td>${r.stability.toFixed(2)}</td>
    <td>${r.motion.toFixed(3)}</td></tr>`).join('');

/* ---------- pipeline diagram ---------- */
document.getElementById('diagram').innerHTML=pipelineSVG();
function pipelineSVG(){
  const A=getComputedStyle(root).getPropertyValue('--accent').trim()||'#ff4d4d';
  const ink=getComputedStyle(root).getPropertyValue('--ink').trim()||'#e6e9ef';
  const mut=getComputedStyle(root).getPropertyValue('--muted').trim()||'#828b9e';
  const ln=getComputedStyle(root).getPropertyValue('--line').trim()||'#2a303c';
  const p2=getComputedStyle(root).getPropertyValue('--panel2').trim()||'#1e232e';
  function box(x,y,w,h,label,sub,stroke){return `<rect x="${x}" y="${y}" rx="9" width="${w}" height="${h}" fill="${p2}" stroke="${stroke||ln}"/>
    <text x="${x+w/2}" y="${y+ (sub?h/2-3:h/2+4)}" fill="${ink}" font-family="ui-monospace,monospace" font-size="12" text-anchor="middle">${label}</text>
    ${sub?`<text x="${x+w/2}" y="${y+h/2+13}" fill="${mut}" font-family="ui-monospace,monospace" font-size="9.5" text-anchor="middle">${sub}</text>`:''}`;}
  function arrow(x1,y1,x2,y2,c){return `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="${c||mut}" stroke-width="1.6" marker-end="url(#ah)"/>`;}
  const conds=[['c_k','instruction'],['r_k','reference'],['H_k','history'],['M_k','memory']];
  let inb=conds.map((c,i)=>box(20,20+i*54,120,42,c[0],c[1],i===0?A:ln)).join('');
  let arrows=conds.map((c,i)=>arrow(140,41+i*54,205,150,ln)).join('');
  return `<svg viewBox="0 0 900 250" role="img" aria-label="chunk pipeline">
    <defs><marker id="ah" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="${mut}"/></marker></defs>
    ${inb}${arrows}
    ${box(205,120,150,60,'few-step flow','rectified-flow student',A)}
    ${arrow(355,150,395,150)}
    ${box(395,120,120,60,'decode','latent → pixels')}
    ${arrow(515,150,555,150)}
    ${box(555,120,140,60,'super-res','windowed refine')}
    ${arrow(695,150,735,150,A)}
    ${box(735,120,140,60,'deliver','progressive FIFO',A)}
    <line x1="465" y1="180" x2="465" y2="215" stroke="${mut}" stroke-width="1.4" stroke-dasharray="4 3"/>
    <path d="M465,215 L120,215 L120,190" fill="none" stroke="${mut}" stroke-width="1.4" stroke-dasharray="4 3" marker-end="url(#ah)"/>
    <text x="300" y="232" fill="${mut}" font-family="ui-monospace,monospace" font-size="10">committed frames → consolidated into bounded memory (M_k)</text>
  </svg>`;
}
</script>
"""
