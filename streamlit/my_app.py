#!/usr/bin/env python3
"""
══════════════════════════════════════════════════════════════
 Superlet Explorer — Vizualizare EEG cu FASLT real
══════════════════════════════════════════════════════════════

 Upload complet:  EEG .bin + Event-Codes .bin + Event-Timestamps .bin
                  → extragere trial-uri → Superlet Transform → vizualizare

 Algoritmul Superlet:  Moca et al. 2021, Nature Communications
                       Implementare Bârzan & Ardelean (TINS)

 Utilizare:
     pip install streamlit numpy pandas scipy
     streamlit run app_superlet_streamlit.py
"""

import streamlit as st
import numpy as np
import json
import os
import io
from scipy.signal import fftconvolve

# ══════════════════════════════════════════════════════════════
# SUPERLET TRANSFORM  (Moca et al. 2021 — implementare TINS)
# ══════════════════════════════════════════════════════════════

MORLET_SD_SPREAD = 6
MORLET_SD_FACTOR = 2.5


def _morlet(fc, nc, fs):
    sd = (nc / 2) * (1 / np.abs(fc)) / MORLET_SD_FACTOR
    size = int(2 * np.floor(np.round(sd * fs * MORLET_SD_SPREAD) / 2) + 1)
    half = int(np.floor(size / 2))
    alpha = MORLET_SD_SPREAD / 2
    t_idx = np.arange(size, dtype=np.float64) - half
    gauss = np.exp(-((t_idx * (alpha / half)) ** 2) * 0.5)
    igsum = 1 / gauss.sum()
    t_sec = t_idx / fs
    return gauss * np.exp(2 * np.pi * fc * t_sec * 1j) * igsum


def _fractional(x):
    return x - int(x)


def superlet_transform(signal, fs, foi, c1, orders):
    """
    Fractional Adaptive Superlet Transform (FASLT).

    Args:
        signal:  1D array (n_samples,)
        fs:      sampling rate (Hz)
        foi:     array of frequencies of interest
        c1:      base cycles
        orders:  tuple (o_min, o_max) — ordine adaptive pe foi

    Returns:
        spectrogram: (n_freq, n_samples) — putere
    """
    n = len(signal)
    n_freq = len(foi)
    ord_vec = np.linspace(orders[0], orders[1], n_freq)
    result = np.zeros((n_freq, n), dtype=np.float64)

    for i_f in range(n_freq):
        fc = foi[i_f]
        order = ord_vec[i_f]
        n_wavelets = int(np.floor(order))

        pool = np.ones(n, dtype=np.float64)

        if n_wavelets > 1:
            for i_w in range(n_wavelets):
                wav = _morlet(fc, (i_w + 1) * c1, fs)
                resp = fftconvolve(signal, wav, "same")
                pool *= 2 * np.abs(resp) ** 2

            # Fractional part
            frac = _fractional(order)
            if frac > 0:
                wav = _morlet(fc, (n_wavelets + 1) * c1, fs)
                resp = fftconvolve(signal, wav, "same")
                pool *= (2 * np.abs(resp) ** 2) ** frac
                rfactor = 1 / (n_wavelets + frac)
            else:
                rfactor = 1.0 / n_wavelets

            result[i_f, :] = pool ** rfactor
        else:
            # Order 1 = simplu CWT
            wav = _morlet(fc, c1, fs)
            result[i_f, :] = 2 * np.abs(fftconvolve(signal, wav, "same")) ** 2

    return result


# ══════════════════════════════════════════════════════════════
# CONSTANTE
# ══════════════════════════════════════════════════════════════

FS = 1024
DTYPE_F32 = np.float32
DTYPE_I32 = np.int32
CLASS_NAMES = {1: "Seen", 2: "Uncertain", 3: "Nothing"}
FOI = np.arange(4, 101, dtype=float)  # 4–100 Hz


# ══════════════════════════════════════════════════════════════
# FUNCȚII ÎNCĂRCARE & EXTRACȚIE
# ══════════════════════════════════════════════════════════════

def load_bin(uploaded_file, dtype=np.float32):
    raw = uploaded_file.read()
    uploaded_file.seek(0)
    return np.frombuffer(raw, dtype=dtype)


def extract_trials(signal, ev_codes, ev_ts, t_pre=0.2, t_post=0.8, quality_mask=None):
    """Extrage epoci stimulus-locked."""
    n_samp = len(signal)
    stim_idx = np.where(ev_codes == 129)[0]
    pre_s = int(round(t_pre * FS))
    post_s = int(round(t_post * FS))
    ep_len = pre_s + post_s + 1
    times = (np.arange(ep_len) - pre_s) / FS

    epochs, labels, rts = [], [], []
    csv_ctr = 0

    for k, si in enumerate(stim_idx):
        nxt = stim_idx[k + 1] if k < len(stim_idx) - 1 else len(ev_codes)
        resp_j = None
        for j in range(si + 1, nxt):
            if ev_codes[j] in {1, 2, 3}:
                resp_j = j
                break

        q_ok = True
        if quality_mask is not None and csv_ctr < len(quality_mask):
            q_ok = quality_mask[csv_ctr] == 1
        csv_ctr += 1

        if resp_j is None or not q_ok:
            continue
        s0 = ev_ts[si] - pre_s
        s1 = ev_ts[si] + post_s
        if s0 < 0 or s1 >= n_samp:
            continue

        epoch = signal[s0: s1 + 1].copy().astype(np.float64)
        b_idx = np.where((times >= -t_pre) & (times <= 0.0))[0]
        epoch -= epoch[b_idx].mean()

        epochs.append(epoch)
        labels.append(int(ev_codes[resp_j]))
        rts.append(int(ev_ts[resp_j] - ev_ts[si]))

    if not epochs:
        return None, None, times, None
    return np.stack(epochs), np.array(labels), times, np.array(rts)


def extract_windows(signal, win_sec=1.0, step_sec=0.5, max_win=20):
    win_s = int(win_sec * FS)
    step_s = int(step_sec * FS)
    times = np.arange(win_s) / FS
    windows, starts = [], []
    for s0 in range(0, len(signal) - win_s, step_s):
        if len(windows) >= max_win:
            break
        ch = signal[s0: s0 + win_s].copy().astype(np.float64)
        ch -= ch[: win_s // 5].mean()
        windows.append(ch)
        starts.append(s0)
    if not windows:
        return None, None, times, None
    return np.stack(windows), np.zeros(len(windows), int), times, np.array(starts)


def generate_demo(n_trials=10, n_samples=1025):
    np.random.seed(42)
    times = (np.arange(n_samples) - 205) / FS
    epochs = np.random.randn(n_trials, n_samples) * 5
    for i in range(n_trials):
        epochs[i] += 3 * np.sin(2 * np.pi * 10 * times + np.random.rand() * 6.28)
        epochs[i] += 1.5 * np.sin(2 * np.pi * 25 * times + np.random.rand() * 6.28)
        t0 = int(0.2 * FS)
        if t0 + 130 < n_samples:
            epochs[i, t0+80:t0+130] += -8 * np.exp(-0.5 * ((np.arange(50)-25)/10)**2)
        if t0 + 400 < n_samples:
            epochs[i, t0+250:t0+400] += 5 * np.exp(-0.5 * ((np.arange(150)-75)/30)**2)
    y = np.array([1,1,1,3,3,3,1,3,2,2])[:n_trials]
    rt = np.random.randint(500, 5000, n_trials)
    return epochs.astype(np.float64), y, times, rt


# ══════════════════════════════════════════════════════════════
# COMPUTE SPECTROGRAMS (cached)
# ══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Se calculează spectrogramele Superlet...")
def compute_all_spectrograms(_epochs, c1, o_min, o_max):
    """
    Calculează FASLT + CWT pentru toate trial-urile.
    Returnează spectrograme downsampled (uint8 quantized) pentru vizualizare.
    """
    n_trials, n_samples = _epochs.shape
    ds_factor = max(1, n_samples // 256)  # ~256 time bins

    specs_faslt = []
    specs_cwt = []

    for i in range(n_trials):
        sig = _epochs[i]

        # FASLT (ordine adaptive)
        sp_faslt = superlet_transform(sig, FS, FOI, c1, (o_min, o_max))
        specs_faslt.append(sp_faslt[:, ::ds_factor])

        # CWT (ordin 1, pt comparație)
        sp_cwt = superlet_transform(sig, FS, FOI, c1, (1, 1))
        specs_cwt.append(sp_cwt[:, ::ds_factor])

    return specs_faslt, specs_cwt, ds_factor


def quantize_spec(spec, global_max=None):
    """Cuantizează spectrograma la uint8 pentru JSON compact."""
    if global_max is None or global_max == 0:
        global_max = spec.max() or 1.0
    normed = np.sqrt(np.clip(spec / global_max, 0, 1))  # sqrt for perceptual scaling
    return (normed * 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════
# SERIALIZARE JSON
# ══════════════════════════════════════════════════════════════

def prepare_json(epochs, labels, times, rts, specs_faslt, specs_cwt,
                 ds_factor, source, is_continuous, max_per_class=10):
    trials = []
    global_max_faslt = max(s.max() for s in specs_faslt) or 1.0
    global_max_cwt = max(s.max() for s in specs_cwt) or 1.0

    # Selectăm trial-urile
    if is_continuous:
        sel_idx = list(range(min(len(epochs), max_per_class * 3)))
    else:
        sel_idx = []
        for cls in [1, 3, 2]:
            idx = np.where(labels == cls)[0][:max_per_class]
            sel_idx.extend(idx.tolist())

    for i in sel_idx:
        sig = epochs[i]
        q_faslt = quantize_spec(specs_faslt[i], global_max_faslt)
        q_cwt = quantize_spec(specs_cwt[i], global_max_cwt)

        trial = {
            "idx": int(i),
            "cls": int(labels[i]) if labels is not None else 0,
            "cls_name": CLASS_NAMES.get(int(labels[i]), "Window") if not is_continuous else "Window",
            "rt_ms": int(rts[i] / FS * 1000) if rts is not None and rts[i] > 100 else int(rts[i]) if rts is not None else 0,
            "sig": [round(float(v), 2) for v in sig],
            # Spectrograms as flat arrays of uint8
            "sp_faslt": q_faslt.flatten().tolist(),
            "sp_cwt": q_cwt.flatten().tolist(),
            "sp_rows": int(q_faslt.shape[0]),
            "sp_cols": int(q_faslt.shape[1]),
        }
        trials.append(trial)

    return json.dumps({
        "src": source,
        "fs": FS,
        "N": int(epochs.shape[1]),
        "cont": is_continuous,
        "t_ms": [round(t * 1000, 1) for t in times.tolist()],
        "foi_min": int(FOI[0]),
        "foi_max": int(FOI[-1]),
        "ds": ds_factor,
        "trials": trials,
    }, separators=(",", ":"))


# ══════════════════════════════════════════════════════════════
# HTML COMPONENT
# ══════════════════════════════════════════════════════════════

def build_html(data_json, c1, o_min, o_max):
    return f'''
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

.S *{{box-sizing:border-box;margin:0;padding:0;}}
.S{{font-family:'DM Sans',sans-serif;background:transparent;color:#c8cad0;padding:6px 0;}}

/* Controls */
.S .row{{display:flex;align-items:center;gap:10px;margin:0 0 6px;}}
.S .row label{{font-size:10px;color:#5a6070;min-width:56px;text-transform:uppercase;letter-spacing:0.06em;font-weight:500;}}
.S .row input[type="range"]{{flex:1;height:3px;-webkit-appearance:none;appearance:none;background:#1a1f2a;border-radius:2px;outline:none;}}
.S .row input[type="range"]::-webkit-slider-thumb{{-webkit-appearance:none;width:14px;height:14px;border-radius:50%;background:#0e1117;border:2px solid #5b8df9;cursor:pointer;}}
.S .row span{{font-family:'JetBrains Mono',monospace;font-size:10px;min-width:50px;text-align:right;color:#7a8090;}}
.S select{{font-size:11px;padding:3px 8px;border:1px solid #222838;border-radius:4px;background:#0e1117;color:#c8cad0;cursor:pointer;font-family:inherit;}}

/* Trial buttons */
.S .btns{{max-height:70px;overflow-y:auto;margin:0 0 4px;}}
.S .b{{display:inline-block;padding:3px 9px;margin:1px;font-size:9px;border-radius:4px;cursor:pointer;border:1px solid #222838;background:#0e1117;color:#7a8090;transition:all 0.1s;font-family:inherit;}}
.S .b:hover{{background:#151a24;border-color:#303848;}}
.S .b.on{{background:#5b8df9;color:#fff;border-color:#5b8df9;}}
.S .b.c1{{border-left:3px solid #5b8df9;}}
.S .b.c3{{border-left:3px solid #f9605b;}}
.S .b.c2{{border-left:3px solid #f9c55b;}}
.S .b.c0{{border-left:3px solid #666;}}

/* Labels */
.S .L{{font-size:9px;font-weight:500;color:#3e4455;margin:8px 0 3px;text-transform:uppercase;letter-spacing:0.08em;}}
.S .nfo{{font-family:'JetBrains Mono',monospace;font-size:9px;color:#3e4455;margin:1px 0 4px;}}

/* Canvas */
.S canvas{{width:100%;border-radius:5px;border:1px solid #161b25;margin-bottom:2px;background:#0a0d14;}}

/* Legend */
.S .lg{{display:flex;gap:10px;margin:0 0 6px;font-size:8px;color:#3e4455;flex-wrap:wrap;}}
.S .lg i{{display:inline-block;width:10px;height:2px;border-radius:1px;vertical-align:middle;margin-right:2px;}}

/* Spectrogram toggle */
.S .tog{{display:flex;gap:0;margin:4px 0;}}
.S .tog button{{font-size:9px;padding:4px 12px;border:1px solid #222838;background:#0e1117;color:#5a6070;cursor:pointer;font-family:inherit;transition:all 0.1s;}}
.S .tog button:first-child{{border-radius:4px 0 0 4px;}}
.S .tog button:last-child{{border-radius:0 4px 4px 0;border-left:0;}}
.S .tog button.on{{background:#5b8df9;color:#fff;border-color:#5b8df9;}}

/* Stats */
.S .st{{display:flex;gap:5px;margin:4px 0 8px;flex-wrap:wrap;}}
.S .st div{{background:#0d1018;border:1px solid #1a1f2a;border-radius:4px;padding:4px 8px;font-size:9px;color:#5a6070;font-family:'JetBrains Mono',monospace;}}
.S .st div b{{color:#c8cad0;font-weight:500;}}
</style>

<div class="S">

<div class="st" id="stats"></div>

<p class="L">Trial-uri</p>
<div class="btns" id="btns"></div>
<p class="nfo" id="nfo">&nbsp;</p>

<div class="row">
  <label>Frecvență</label>
  <input type="range" id="freq" min="{int(FOI[0])}" max="{int(FOI[-1])}" value="30" step="1" oninput="upd()">
  <span id="fO">30 Hz</span>
</div>
<div class="row">
  <label>Poziție</label>
  <input type="range" id="pos" min="0" max="1023" value="512" step="1" oninput="upd()">
  <span id="pO">300 ms</span>
</div>

<p class="L">Semnal EEG + wavelet Morlet</p>
<canvas id="c1" height="130"></canvas>
<div class="lg">
  <span><i style="background:#5b8df9"></i>Semnal EEG</span>
  <span><i style="background:#f97a5b"></i>Wavelet Morlet (c₁={c1})</span>
  <span>Linia punctată = stimulus ON</span>
</div>

<p class="L">Putere la frecvența selectată</p>
<canvas id="c2" height="85"></canvas>
<div class="lg">
  <span><i style="background:#3dd9a0"></i>Putere (|convoluție|²)</span>
  <span><i style="background:#f97a5b"></i>Cursor</span>
</div>

<div style="display:flex;align-items:center;gap:10px;margin:8px 0 3px;">
  <p class="L" style="margin:0;">Spectrogramă</p>
  <div class="tog" id="tog">
    <button class="on" onclick="setMode('faslt')">FASLT (o={o_min}→{o_max})</button>
    <button onclick="setMode('cwt')">CWT (o=1)</button>
  </div>
</div>
<canvas id="c3" height="190"></canvas>
<div class="lg">
  <span>Pre-calculat server-side cu algoritmul Moca et al. 2021 · c₁={c1} · Chenar = freq selectată · Linie = cursor temporal</span>
</div>

</div>

<script>
const D={data_json};
const c1=D.c1,c2=D.c2,c3=D.c3;
const cv1=document.getElementById('c1'),cv2=document.getElementById('c2'),cv3=document.getElementById('c3');
const x1=cv1.getContext('2d'),x2=cv2.getContext('2d'),x3=cv3.getContext('2d');

function rsz(c){{const d=devicePixelRatio||1;const r=c.getBoundingClientRect();c.width=r.width*d;c.height=r.height*d;c.getContext('2d').scale(d,d);}}
[cv1,cv2,cv3].forEach(rsz);
let W=cv1.getBoundingClientRect().width;
let H1=cv1.getBoundingClientRect().height;
let H2=cv2.getBoundingClientRect().height;
let H3=cv3.getBoundingClientRect().height;
const N=D.N;
document.getElementById('pos').max=N-1;

const M='#3a4050',GR='rgba(255,255,255,0.025)',ST='rgba(255,255,255,0.08)';
let cur=0, sig=new Float64Array(N), spMode='faslt';

// Stats
const sb=document.getElementById('stats');
if(!D.cont){{
  const cc={{}};D.trials.forEach(t=>{{cc[t.cls_name]=(cc[t.cls_name]||0)+1;}});
  let h='<div>Trials: <b>'+D.trials.length+'</b></div>';
  for(const[k,v] of Object.entries(cc)) h+='<div>'+k+': <b>'+v+'</b></div>';
  h+='<div>FASLT: <b>c₁={c1}, o={o_min}→{o_max}</b></div><div>Fs: <b>'+D.fs+'Hz</b></div>';
  h+='<div>Epocă: <b>'+D.t_ms[0]+'→'+D.t_ms[D.t_ms.length-1]+' ms</b></div>';
  sb.innerHTML=h;
}}else{{
  sb.innerHTML='<div>Ferestre: <b>'+D.trials.length+'</b></div><div>Samples: <b>'+N+'</b></div><div>Fs: <b>'+D.fs+'Hz</b></div>';
}}

// Buttons
const bd=document.getElementById('btns');
D.trials.forEach((tr,i)=>{{
  const b=document.createElement('button');
  b.className='b c'+tr.cls;
  b.textContent=D.cont?'W'+i:'T'+tr.idx+' ('+tr.cls_name+')';
  b.onclick=()=>sel(i);
  bd.appendChild(b);
}});

function sel(i){{
  cur=i;
  document.querySelectorAll('.b').forEach((b,j)=>b.classList.toggle('on',j===i));
  const tr=D.trials[i];
  document.getElementById('nfo').textContent=
    D.cont?'Fereastră '+i+' · '+N+' samples':
    'Trial '+tr.idx+' · '+tr.cls_name+' · RT: '+tr.rt_ms+' ms';
  const raw=tr.sig;for(let j=0;j<N;j++)sig[j]=raw[j];
  upd();
}}

function setMode(m){{
  spMode=m;
  document.querySelectorAll('.tog button').forEach(b=>b.classList.remove('on'));
  document.querySelectorAll('.tog button').forEach(b=>{{if((m==='faslt'&&b.textContent.includes('FASLT'))||(m==='cwt'&&b.textContent.includes('CWT')))b.classList.add('on');}});
  upd();
}}

// Simple CWT wavelet for interactive overlay (JS side)
function mkW(fc,nc,fs){{
  const sd=(nc/2)*(1/fc)/2.5;
  const sz=Math.floor(Math.round(sd*fs*6)/2)*2+1;
  const h=Math.floor(sz/2);
  const w=new Float64Array(sz);
  let gs=0;for(let i=0;i<sz;i++){{const t=(i-h)*(3/(h||1));gs+=Math.exp(-t*t*0.5);}}
  for(let i=0;i<sz;i++){{const tg=(i-h)*(3/(h||1));const g=Math.exp(-tg*tg*0.5)/gs;const ts=(i-h)/fs;w[i]=g*Math.cos(2*Math.PI*fc*ts);}}
  return w;
}}
function conv(s,w){{
  const o=new Float64Array(s.length);const h=Math.floor(w.length/2);
  for(let i=0;i<s.length;i++){{let v=0;for(let j=0;j<w.length;j++){{const si=i-h+j;if(si>=0&&si<s.length)v+=s[si]*w[j];}}o[i]=v*v;}}
  return o;
}}

// Inferno colormap (perceptual)
function inferno(v){{
  // v in 0..255
  const t=v/255;
  const r=Math.round(Math.min(255, t<0.35? t/0.35*80 : t<0.65? 80+(t-0.35)/0.3*170 : 250-(t-0.65)/0.35*10));
  const g=Math.round(Math.min(255, t<0.4? t/0.4*15 : t<0.75? 15+(t-0.4)/0.35*120 : 135+(t-0.75)/0.25*120));
  const b=Math.round(t<0.25? 20+t/0.25*100 : t<0.55? 120-(t-0.25)/0.3*70 : t<0.8? 50-(t-0.55)/0.25*40 : 10+(t-0.8)/0.2*80);
  return `rgb(${{r}},${{g}},${{b}})`;
}}

function upd(){{
  const freq=+document.getElementById('freq').value;
  const pos=+document.getElementById('pos').value;
  const tMs=D.t_ms;
  const posLabel=D.cont?Math.round(pos/D.fs*1000)+' ms':tMs[Math.min(pos,tMs.length-1)]+' ms';
  document.getElementById('fO').textContent=freq+' Hz';
  document.getElementById('pO').textContent=posLabel;

  const wav=mkW(freq,3,1024);
  const half=Math.floor(wav.length/2);
  const pwr=conv(sig,wav);
  const px=i=>i/N*W;
  let t0px=D.cont?-10:px(tMs.findIndex(t=>t>=0));
  if(t0px<0&&!D.cont)t0px=px(Math.round(0.2*N));

  // ═══ G1: Signal + Wavelet ═══
  let sn=Infinity,sx=-Infinity;
  for(let i=0;i<N;i++){{if(sig[i]<sn)sn=sig[i];if(sig[i]>sx)sx=sig[i];}}
  if(sx===sn)sx=sn+1;
  const sY=v=>12+(1-(v-sn)/(sx-sn))*(H1-24);

  x1.clearRect(0,0,W,H1);
  x1.fillStyle=GR;for(let g=0;g<=4;g++)x1.fillRect(0,12+g*(H1-24)/4,W,0.5);

  if(t0px>0){{x1.strokeStyle=ST;x1.lineWidth=1;x1.setLineDash([4,4]);x1.beginPath();x1.moveTo(t0px,0);x1.lineTo(t0px,H1);x1.stroke();x1.setLineDash([]);}}

  x1.strokeStyle='#5b8df9';x1.lineWidth=1;x1.beginPath();
  for(let i=0;i<N;i++){{const xp=px(i),y=sY(sig[i]);i?x1.lineTo(xp,y):x1.moveTo(xp,y);}}x1.stroke();

  const ws=pos-half,we=pos+half;
  x1.fillStyle='rgba(249,122,91,0.06)';
  x1.fillRect(px(Math.max(0,ws)),0,px(Math.min(N,we)-Math.max(0,ws)),H1);

  const wM=Math.max(...wav.map(Math.abs))*0.7;
  x1.strokeStyle='#f97a5b';x1.lineWidth=1.3;x1.beginPath();let st=false;
  for(let j=0;j<wav.length;j++){{const si=ws+j;if(si<0||si>=N)continue;const xp=px(si),wN=wav[j]/wM,y=H1/2-wN*(H1/2-12);st?x1.lineTo(xp,y):x1.moveTo(xp,y);st=true;}}
  x1.stroke();

  x1.fillStyle=M;x1.font='9px JetBrains Mono,monospace';
  if(!D.cont){{x1.fillText(tMs[0]+'ms',2,H1-2);if(t0px>0)x1.fillText('0',t0px+3,H1-2);x1.fillText(tMs[tMs.length-1]+'ms',W-46,H1-2);}}
  else{{x1.fillText('0',2,H1-2);x1.fillText(Math.round(N/1024*1000)+'ms',W-40,H1-2);}}
  x1.fillText(wav.length+'smp · '+Math.round(wav.length/1024*1000)+'ms',Math.min(px(pos)-40,W-150),10);

  // ═══ G2: Power ═══
  let pM=0;for(let i=0;i<N;i++)if(pwr[i]>pM)pM=pwr[i];pM=pM*1.1||1;
  const pY=v=>6+(1-v/pM)*(H2-14);

  x2.clearRect(0,0,W,H2);x2.fillStyle=GR;x2.fillRect(0,pY(0),W,0.5);
  if(t0px>0){{x2.strokeStyle=ST;x2.lineWidth=1;x2.setLineDash([4,4]);x2.beginPath();x2.moveTo(t0px,0);x2.lineTo(t0px,H2);x2.stroke();x2.setLineDash([]);}}

  x2.fillStyle='rgba(61,217,160,0.06)';x2.beginPath();x2.moveTo(0,pY(0));
  for(let i=0;i<N;i++)x2.lineTo(px(i),pY(pwr[i]));x2.lineTo(px(N-1),pY(0));x2.fill();
  x2.strokeStyle='#3dd9a0';x2.lineWidth=1;x2.beginPath();
  for(let i=0;i<N;i++){{const xp=px(i),y=pY(pwr[i]);i?x2.lineTo(xp,y):x2.moveTo(xp,y);}}x2.stroke();

  x2.strokeStyle='#f97a5b';x2.lineWidth=1;x2.setLineDash([4,3]);x2.beginPath();x2.moveTo(px(pos),0);x2.lineTo(px(pos),H2);x2.stroke();x2.setLineDash([]);
  x2.fillStyle='#f97a5b';x2.beginPath();x2.arc(px(pos),pY(pwr[Math.min(pos,N-1)]),3,0,Math.PI*2);x2.fill();
  x2.fillStyle=M;x2.font='9px JetBrains Mono,monospace';x2.fillText('power @'+freq+'Hz',4,H2-2);

  // ═══ G3: Pre-computed Spectrogram ═══
  const tr=D.trials[cur];
  const sp=spMode==='faslt'?tr.sp_faslt:tr.sp_cwt;
  const nR=tr.sp_rows, nC=tr.sp_cols;
  const cH=H3/nR;

  x3.clearRect(0,0,W,H3);
  for(let fi=0;fi<nR;fi++){{
    const y=H3-(fi+1)*cH;
    for(let ti=0;ti<nC;ti++){{
      const v=sp[fi*nC+ti];
      x3.fillStyle=inferno(v);
      const xp=ti/nC*W;
      const cW=W/nC+1;
      x3.fillRect(xp,y,cW,Math.ceil(cH)+1);
    }}
  }}

  if(t0px>0){{x3.strokeStyle='rgba(255,255,255,0.15)';x3.lineWidth=1;x3.setLineDash([4,4]);x3.beginPath();x3.moveTo(t0px,0);x3.lineTo(t0px,H3);x3.stroke();x3.setLineDash([]);}}

  const fIdx=freq-D.foi_min;
  const rY=H3-(fIdx+1)*cH;
  x3.strokeStyle='rgba(255,255,255,0.45)';x3.lineWidth=1;x3.strokeRect(0,rY,W,cH);

  x3.strokeStyle='#f97a5b';x3.lineWidth=1;x3.setLineDash([4,3]);x3.beginPath();x3.moveTo(px(pos),0);x3.lineTo(px(pos),H3);x3.stroke();x3.setLineDash([]);

  x3.fillStyle='rgba(255,255,255,0.55)';x3.font='9px JetBrains Mono,monospace';
  x3.fillText(D.foi_min+'Hz',3,H3-3);x3.fillText(D.foi_max+'Hz',3,10);
  x3.fillText(freq+'Hz ▸',3,rY+cH/2+3);
  if(t0px>0)x3.fillText('stim',t0px+3,H3-3);
  x3.fillText(spMode==='faslt'?'FASLT':'CWT',W-32,10);
}}

addEventListener('resize',()=>{{[cv1,cv2,cv3].forEach(rsz);W=cv1.getBoundingClientRect().width;H1=cv1.getBoundingClientRect().height;H2=cv2.getBoundingClientRect().height;H3=cv3.getBoundingClientRect().height;upd();}});

sel(0);
</script>
'''


# ══════════════════════════════════════════════════════════════
# STREAMLIT APP
# ══════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="Superlet Explorer — EEG",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with st.sidebar:
        st.markdown("## 🧠 Superlet Explorer")
        st.caption(
            "Vizualizare interactivă cu Fractional Adaptive Superlet Transform "
            "(Moca et al. 2021)"
        )
        st.divider()

        mode = st.radio(
            "Mod",
            ["Upload complet (.bin)", "Upload un canal", "Demo"],
            index=0,
        )

        # Defaults
        eeg_file = ev_codes_file = ev_ts_file = csv_file = None
        t_pre = 0.2; t_post = 0.8; max_trials = 8
        win_sec = 1.0; max_windows = 15
        c1 = 3; o_min = 1; o_max = 10

        if mode == "Upload complet (.bin)":
            st.markdown("#### 1. Fișier EEG (un canal, float32)")
            eeg_file = st.file_uploader("EEG .bin", type=["bin"], key="eeg")

            st.markdown("#### 2. Events (int32)")
            ev_codes_file = st.file_uploader("Event-Codes .bin", type=["bin"], key="ec")
            ev_ts_file = st.file_uploader("Event-Timestamps .bin", type=["bin"], key="et")

            st.markdown("#### 3. Trialinfo CSV (opțional)")
            csv_file = st.file_uploader("CSV", type=["csv"], key="csv")

            st.divider()
            st.markdown("#### Parametri epoching")
            t_pre = st.slider("Pre-stimulus (s)", 0.1, 0.5, 0.2, 0.05)
            t_post = st.slider("Post-stimulus (s)", 0.5, 2.0, 0.8, 0.1)
            max_trials = st.slider("Max trials/clasă", 3, 30, 8)

        elif mode == "Upload un canal":
            st.markdown("#### Fișier EEG (un canal, float32)")
            eeg_file = st.file_uploader("EEG .bin", type=["bin"], key="eeg_s")
            st.divider()
            win_sec = st.slider("Fereastră (s)", 0.5, 3.0, 1.0, 0.1)
            max_windows = st.slider("Nr. ferestre", 5, 40, 15)

        st.divider()
        st.markdown("#### Parametri Superlet")
        c1 = st.slider("Base cycles (c₁)", 1, 7, 3, help="Nr. cicluri wavelet de bază")
        o_col1, o_col2 = st.columns(2)
        with o_col1:
            o_min = st.number_input("Ordin min", 1, 20, 1)
        with o_col2:
            o_max = st.number_input("Ordin max", 1, 20, 10)

        st.divider()
        st.markdown(
            '<div style="font-size:9px;color:#555;line-height:1.4;">'
            '<b>Algoritm:</b> FASLT — Moca, Bârzan et al. 2021<br>'
            '<b>Implementare:</b> TINS<br>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ═══ Main area ═══
    st.markdown("### Superlet Transform Explorer")

    # ── Load data ──
    epochs = labels = times = rts = None
    source_name = ""
    is_continuous = False

    if mode == "Upload complet (.bin)":
        if not eeg_file:
            st.info(
                " **Încarcă cele 3 fișiere `.bin` din sidebar:**\n\n"
                "| Fișier | Tip | Conținut |\n"
                "|--------|-----|--------|\n"
                "| `Dots_30_0XX-YY.bin` | float32 | Un canal EEG continuu |\n"
                "| `Dots_30_0XX-Event-Codes.bin` | int32 | 129=stim, 1=Seen, 2=Unc, 3=Nothing |\n"
                "| `Dots_30_0XX-Event-Timestamps.bin` | int32 | Index sample per event |\n"
                "| `*trialinfo.csv` | CSV | GoodTrialsManual (opțional) |"
            )
            return

        signal = load_bin(eeg_file, DTYPE_F32)
        source_name = eeg_file.name

        if not ev_codes_file or not ev_ts_file:
            st.warning("Încarcă și fișierele Event-Codes + Event-Timestamps pentru modul complet.")
            return

        ev_codes = load_bin(ev_codes_file, DTYPE_I32)
        ev_ts = load_bin(ev_ts_file, DTYPE_I32)

        # Quality mask
        quality_mask = None
        if csv_file:
            import pandas as pd
            csv_file.seek(0)
            txt = csv_file.read().decode("utf-8"); csv_file.seek(0)
            skip = 0
            for i, ln in enumerate(txt.split("\n")):
                if ln.startswith("Trial,"):
                    skip = i; break
            df = pd.read_csv(io.StringIO(txt), skiprows=skip)
            if "GoodTrialsManual" in df.columns:
                quality_mask = df["GoodTrialsManual"].values
                st.success(f"CSV: {int(quality_mask.sum())}/{len(quality_mask)} good trials")

        epochs, labels, times, rts = extract_trials(
            signal, ev_codes, ev_ts, t_pre, t_post, quality_mask
        )
        is_continuous = False

    elif mode == "Upload un canal":
        if not eeg_file:
            st.info("Încarcă un fișier `.bin` EEG (float32, un canal)")
            return
        signal = load_bin(eeg_file, DTYPE_F32)
        source_name = eeg_file.name
        epochs, labels, times, rts = extract_windows(signal, win_sec, max_windows=max_windows)
        is_continuous = True

    else:
        epochs, labels, times, rts = generate_demo()
        source_name = "Demo (sintetic)"
        is_continuous = False

    if epochs is None or len(epochs) == 0:
        st.error("Nu s-au extras segmente. Verifică fișierele.")
        return

    # ── Stats ──
    c1_m, c2_m, c3_m, c4_m = st.columns(4)
    c1_m.metric("Segmente", epochs.shape[0])
    c2_m.metric("Samples", epochs.shape[1])
    if not is_continuous:
        u, cts = np.unique(labels, return_counts=True)
        d = dict(zip(u, cts))
        c3_m.metric("Seen / Nothing", f"{d.get(1,0)} / {d.get(3,0)}")
        c4_m.metric("Uncertain", d.get(2, 0))
    else:
        c3_m.metric("Durată fereastră", f"{win_sec}s")
        c4_m.metric("Fișier", source_name)

    # ── Compute spectrograms ──
    specs_faslt, specs_cwt, ds_factor = compute_all_spectrograms(
        epochs, c1, o_min, o_max
    )

    st.caption(
        f"✅ Spectrograme calculate: **FASLT** (c₁={c1}, o={o_min}→{o_max}) "
        f"vs **CWT** (o=1) · {len(FOI)} frecvențe ({int(FOI[0])}–{int(FOI[-1])} Hz) · "
        f"downsample ×{ds_factor}"
    )

    # ── Build & render ──
    data_json = prepare_json(
        epochs, labels, times, rts,
        specs_faslt, specs_cwt, ds_factor,
        source_name, is_continuous,
        max_per_class=max_trials,
    )

    html = build_html(data_json, c1, o_min, o_max)
    st.components.v1.html(html, height=680, scrolling=False)

    # ── Info panel ──
    with st.expander("ℹ️ Despre pipeline & algoritm"):
        st.markdown(f"""
**Fișier:** `{source_name}`  
**Parametri Superlet:** c₁ = {c1}, ordine = {o_min} → {o_max}  
**Frecvențe:** {int(FOI[0])}–{int(FOI[-1])} Hz ({len(FOI)} bins)

---

**Ce e Superlet Transform?**

Algoritmul FASLT (Moca et al., Nature Communications, 2021) combină wavelet-uri Morlet 
cu numere crescătoare de cicluri (c₁, 2·c₁, ..., o·c₁) prin **media geometrică** 
a răspunsurilor individuale. Astfel se obține simultan rezoluție temporală bună 
(de la wavelet-urile scurte) și rezoluție frecvențială bună (de la cele lungi) — 
ceea ce un singur wavelet nu poate face.

**Ordinile adaptive** cresc liniar de la frecvențele joase la cele înalte, 
reducând redundanța la frecvențele mari unde CWT-ul standard are blur temporal.

**Toggle FASLT ↔ CWT:** Compară spectrograma cu ordine adaptive (FASLT) vs. 
un singur wavelet (CWT, ordin 1). Diferența e vizibilă mai ales la frecvențele 
înalte, unde FASLT are precizie temporală mult mai bună.

---

**Cele 3 grafice:**

1. **Semnal + Wavelet** — semnalul EEG cu waveletul Morlet suprapus la frecvența/poziția selectate
2. **Putere** — |convoluție|² la frecvența curentă (un rând din spectrogramă, calculat interactiv în JS)
3. **Spectrogramă** — pre-calculată server-side cu algoritmul complet FASLT/CWT, cuantizată și redată cu colormap inferno
        """)


if __name__ == "__main__":
    main()
