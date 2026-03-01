#!/usr/bin/env python3
"""capstone2 Phase 1+2 data export to 7 Google Sheets CSVs.

Phase 2 expansion:
  - system_power: Solo 10 + Concurrent 40
  - workload_resource_usage: 9 Solo + 10 combo Concurrent
  - verification: 10 combo
  - scaling_pattern: 9 workloads
  - cross-combo: Phase 1+2 all
  - IPC/cache: 9 workloads
  - (new) anchor_spectrum: anchor-spectrum classification

Usage:
    cd <project-root>   # directory containing data_merged/ and data/
    python analysis/export_gsheet.py
"""

import os
import pandas as pd
import numpy as np

# ── paths (relative to project root = parent of analysis/) ──
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SUMMARY = os.path.join(BASE, 'data_merged', 'summary.csv')
df = pd.read_csv(SUMMARY)

solo = df[df['combo'] == '-'].copy()
conc = df[df['combo'] != '-'].copy()

# ═══════════════════════════════════════════════════════════════════
# Sheet 1: System Power (Solo 10 + Concurrent 40)
# ═══════════════════════════════════════════════════════════════════
rows = []

for _, r in solo.iterrows():
    rows.append({
        'experiment': r['exp_id'],
        'type': r['type'],
        'workload_A': r['vm_a'],
        'workload_B': '-',
        'ratio': '-',
        'cores_A': r['vm_a_cores'],
        'cores_B': '-',
        'wall_power_W': round(r['wall_W'], 2) if pd.notna(r['wall_W']) else '',
        'cpu_power_W': round(r['rapl_W'], 2),
        'gpu0_power_W': round(r['gpu0_W'], 2),
        'gpu1_power_W': round(r['gpu1_W'], 2),
        'dram_power_W': round(r['dram_W'], 2),
        'others_power_W': round(r['others_W'], 2) if pd.notna(r['others_W']) else '',
        'sys_cpu_pct': round(r['sys_cpu_pct'], 1),
    })

rows.append({k: '' for k in rows[0]})

for _, r in conc.sort_values(['combo', 'ratio']).iterrows():
    rows.append({
        'experiment': r['exp_id'],
        'type': r['type'],
        'workload_A': r['vm_a'],
        'workload_B': r['vm_b'],
        'ratio': r['ratio'],
        'cores_A': r['vm_a_cores'],
        'cores_B': r['vm_b_cores'],
        'wall_power_W': round(r['wall_W'], 2) if pd.notna(r['wall_W']) else '',
        'cpu_power_W': round(r['rapl_W'], 2),
        'gpu0_power_W': round(r['gpu0_W'], 2),
        'gpu1_power_W': round(r['gpu1_W'], 2),
        'dram_power_W': round(r['dram_W'], 2),
        'others_power_W': round(r['others_W'], 2) if pd.notna(r['others_W']) else '',
        'sys_cpu_pct': round(r['sys_cpu_pct'], 1),
    })

notes = [
    {k: '' for k in rows[0]},
    {'experiment': '[Note]'},
    {'experiment': 'CPU', 'type': 'Intel Xeon E5-2630 v4 (10C x 2HT = 20 logical cores, 1 socket). cpu_power=RAPL package-0.'},
    {'experiment': 'GPU', 'type': 'NVIDIA Titan V x2. GeForce 210 removed (0221). gpu0/gpu1=per-GPU nvidia-smi.'},
    {'experiment': 'Memory', 'type': 'RAPL DRAM. DDR4 2133MHz 32GiB (1-DIMM).'},
    {'experiment': 'Others', 'type': 'Wall - CPU - GPU0 - GPU1 - DRAM = 63~70W (constant). PSU loss, fans, chipset, etc.'},
    {'experiment': 'Turbo', 'type': 'OFF (no_turbo=1). Governor=ondemand. systemd persistent config.'},
    {'experiment': 'VM simulation', 'type': 'cgroup v2: cpuset (core isolation) + memory.max (memory cap). GPU assigned via --device cuda:N.'},
    {'experiment': 'Protocol', 'type': '30s idle -> 120s workload -> 30s cooldown. 1s sampling. drop_caches=3 before each experiment.'},
    {'experiment': 'Phase 1', 'type': '0223~0224. Solo 7 + Conc 24 (A,B,C,D,E,F x 4 ratios).'},
    {'experiment': 'Phase 2', 'type': '0225. Solo 3 + Conc 16 (G,H,I,J x 4 ratios). Workloads: Training, ffmpeg, GPU-LLM.'},
]
rows.extend(notes)

out1 = pd.DataFrame(rows)
out1.to_csv(os.path.join(BASE, 'gsheet_system_power.csv'), index=False)
print('[1/7] gsheet_system_power.csv')


# ═══════════════════════════════════════════════════════════════════
# Sheet 2: Per-workload resource usage (per-PID)
# ═══════════════════════════════════════════════════════════════════
rows2 = []

for _, r in solo[solo['vm_a'] != 'idle'].iterrows():
    rows2.append({
        'experiment': r['exp_id'],
        'type': 'solo',
        'workload': r['vm_a'],
        'cpu_pct': round(r['a_cpu_pct'], 1),
        'gpu_sm_pct': round(r['a_sm_pct'], 1) if pd.notna(r['a_sm_pct']) else 0,
        'gpu_power_W': round(r['gpu0_W'], 1),
        'mem_rss_MiB': round(r['a_rss_mib'], 0) if pd.notna(r['a_rss_mib']) else '',
        'dram_power_W': round(r['dram_W'], 2),
        'io_read_mbs': round(r['a_io_r_mbs'], 2) if pd.notna(r['a_io_r_mbs']) else 0,
        'io_write_mbs': round(r['a_io_w_mbs'], 2) if pd.notna(r['a_io_w_mbs']) else 0,
    })

rows2.append({k: '' for k in rows2[0]})

conc_1_1 = conc[conc['ratio'] == '1:1'].sort_values('combo')
for _, r in conc_1_1.iterrows():
    rows2.append({
        'experiment': r['exp_id'],
        'type': f"conc ({r['combo']})",
        'workload': r['vm_a'],
        'cpu_pct': round(r['a_cpu_pct'], 1),
        'gpu_sm_pct': round(r['a_sm_pct'], 1) if pd.notna(r['a_sm_pct']) else 0,
        'gpu_power_W': round(r['gpu0_W'], 1),
        'mem_rss_MiB': round(r['a_rss_mib'], 0) if pd.notna(r['a_rss_mib']) else '',
        'dram_power_W': round(r['dram_W'], 2),
        'io_read_mbs': round(r['a_io_r_mbs'], 2) if pd.notna(r['a_io_r_mbs']) else 0,
        'io_write_mbs': round(r['a_io_w_mbs'], 2) if pd.notna(r['a_io_w_mbs']) else 0,
    })
    rows2.append({
        'experiment': r['exp_id'],
        'type': f"conc ({r['combo']})",
        'workload': r['vm_b'],
        'cpu_pct': round(r['b_cpu_pct'], 1),
        'gpu_sm_pct': round(r['b_sm_pct'], 1) if pd.notna(r['b_sm_pct']) else 0,
        'gpu_power_W': round(r['gpu1_W'], 1),
        'mem_rss_MiB': round(r['b_rss_mib'], 0) if pd.notna(r['b_rss_mib']) else '',
        'dram_power_W': '',
        'io_read_mbs': round(r['b_io_r_mbs'], 2) if pd.notna(r['b_io_r_mbs']) else 0,
        'io_write_mbs': round(r['b_io_w_mbs'], 2) if pd.notna(r['b_io_w_mbs']) else 0,
    })

notes2 = [
    {k: '' for k in rows2[0]},
    {'experiment': '[Note]'},
    {'experiment': 'Workloads (9)', 'type': 'Phase1: ResNet-CPU, ResNet-GPU, GEMM, LLM, YOLO, NodeJS. Phase2: Training, ffmpeg, GPU-LLM.'},
    {'experiment': 'cpu_pct', 'type': 'per-PID CPU% (/proc/[pid]/stat). Distinct from sys_cpu_pct (system-wide).'},
    {'experiment': 'gpu_sm_pct', 'type': 'per-PID GPU SM% (nvidia-smi pmon).'},
    {'experiment': 'gpu_power_W', 'type': 'Solo: GPU0 power. Conc vm_a: GPU0, vm_b: GPU1.'},
    {'experiment': 'mem_rss_MiB', 'type': 'RSS (physical memory usage). /proc/[pid]/status VmRSS.'},
    {'experiment': 'Resource allocation', 'type': 'Solo/Conc 1:1: 10 cores each, 14GB. vm_a->GPU0(cuda:0), vm_b->GPU1(cuda:1).'},
]
rows2.extend(notes2)

out2 = pd.DataFrame(rows2)
out2.to_csv(os.path.join(BASE, 'gsheet_workload_resource_usage.csv'), index=False)
print('[2/7] gsheet_workload_resource_usage.csv')


# ═══════════════════════════════════════════════════════════════════
# Sheet 3: Solo vs Concurrent verification (1:1 ratio, 10 combos)
# ═══════════════════════════════════════════════════════════════════
rows3 = []

combos = {
    'A': ('ResNet-CPU', 'GEMM'), 'B': ('ResNet-CPU', 'LLM'),
    'C': ('GEMM', 'YOLO'), 'D': ('LLM', 'NodeJS'),
    'E': ('ResNet-GPU', 'ResNet-CPU'), 'F': ('ResNet-GPU', 'LLM'),
    'G': ('ffmpeg', 'LLM'), 'H': ('Training', 'ffmpeg'),
    'I': ('GPU-LLM', 'GEMM'), 'J': ('Training', 'NodeJS'),
}

gpu_map = {
    'A': {'GEMM': 'gpu1_W'},
    'C': {'GEMM': 'gpu0_W', 'YOLO': 'gpu1_W'},
    'E': {'ResNet-GPU': 'gpu0_W'},
    'F': {'ResNet-GPU': 'gpu0_W'},
    'H': {'Training': 'gpu0_W'},
    'I': {'GPU-LLM': 'gpu0_W', 'GEMM': 'gpu1_W'},
    'J': {'Training': 'gpu0_W'},
}

for combo_id in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']:
    if combo_id not in conc_1_1['combo'].values:
        continue
    vm_a, vm_b = combos[combo_id]
    c_row = conc_1_1[conc_1_1['combo'] == combo_id].iloc[0]

    for wl, side in [(vm_a, 'a'), (vm_b, 'b')]:
        s = solo[solo['vm_a'] == wl]
        if len(s) == 0:
            continue
        s = s.iloc[0]

        solo_cpu = round(s['a_cpu_pct'], 1)
        conc_cpu = round(c_row[f'{side}_cpu_pct'], 1)
        d_cpu = round(conc_cpu - solo_cpu, 1)

        gpu_col = gpu_map.get(combo_id, {}).get(wl, None)
        if gpu_col:
            solo_gpu = s['gpu0_W']
            conc_gpu = c_row[gpu_col]
            d_gpu = round(conc_gpu - solo_gpu, 1)
            d_gpu_pct = round((conc_gpu - solo_gpu) / solo_gpu * 100, 1)
        else:
            solo_gpu = conc_gpu = d_gpu = d_gpu_pct = ''

        solo_rss = round(s['a_rss_mib'], 0) if pd.notna(s['a_rss_mib']) else ''
        conc_rss = round(c_row[f'{side}_rss_mib'], 0) if pd.notna(c_row[f'{side}_rss_mib']) else ''

        rows3.append({
            'combo': combo_id,
            'workload': wl,
            'solo_cpu_pct': solo_cpu,
            'conc_cpu_pct': conc_cpu,
            'Δcpu_pct': d_cpu,
            'solo_gpu_power_W': solo_gpu if solo_gpu != '' else '',
            'conc_gpu_power_W': conc_gpu if conc_gpu != '' else '',
            'Δgpu_power_W': d_gpu,
            'Δgpu_pct': d_gpu_pct,
            'solo_rss_MiB': solo_rss,
            'conc_rss_MiB': conc_rss,
        })

notes3 = [
    {k: '' for k in rows3[0]} if rows3 else {},
    {'combo': '[Note]'},
    {'combo': 'Verification goal', 'workload': 'Confirm that a single workload energy profile is preserved under concurrent execution.'},
    {'combo': 'Delta', 'workload': 'Conc - Solo. CPU%: within +/-3%p, GPU Power: within +/-4%.'},
    {'combo': 'Phase 1', 'workload': 'A=ResNet-CPU+GEMM, B=ResNet-CPU+LLM, C=GEMM+YOLO, D=LLM+NodeJS, E=ResNet-GPU+ResNet-CPU, F=ResNet-GPU+LLM'},
    {'combo': 'Phase 2', 'workload': 'G=ffmpeg+LLM, H=Training+ffmpeg, I=GPU-LLM+GEMM, J=Training+NodeJS'},
]
rows3.extend(notes3)

out3 = pd.DataFrame(rows3)
out3.to_csv(os.path.join(BASE, 'gsheet_verification.csv'), index=False)
print('[3/7] gsheet_verification.csv')


# ═══════════════════════════════════════════════════════════════════
# Sheet 4: CPU Scaling Pattern
# ═══════════════════════════════════════════════════════════════════
rows4 = []

for col in ['vm_a_cores', 'vm_b_cores']:
    conc[col] = pd.to_numeric(conc[col], errors='coerce').astype('Int64')

wl_data = {}
for _, r in conc.iterrows():
    combo = r['combo']
    if pd.notna(r['vm_a_cores']):
        wl_data.setdefault(r['vm_a'], []).append({
            'cores': int(r['vm_a_cores']), 'cpu_pct': r['a_cpu_pct'],
            'gpu_W': r['gpu0_W'], 'sm_pct': r['a_sm_pct'],
            'combo': combo, 'side': 'vm_a',
        })
    if pd.notna(r['vm_b_cores']):
        wl_data.setdefault(r['vm_b'], []).append({
            'cores': int(r['vm_b_cores']), 'cpu_pct': r['b_cpu_pct'],
            'gpu_W': r['gpu1_W'], 'sm_pct': r['b_sm_pct'],
            'combo': combo, 'side': 'vm_b',
        })

solo_bl = {}
for _, r in solo[solo['vm_a'] != 'idle'].iterrows():
    solo_bl[r['vm_a']] = {
        'cpu_pct': r['a_cpu_pct'], 'gpu_W': r['gpu0_W'], 'sm_pct': r['a_sm_pct'],
    }

# All 9 workloads
wl_order = ['LLM', 'ResNet-CPU', 'ffmpeg', 'GEMM', 'YOLO', 'ResNet-GPU', 'Training', 'GPU-LLM', 'NodeJS']
pattern_map = {
    'LLM': 'Memory-bound', 'ResNet-CPU': 'Compute-bound', 'ffmpeg': 'Compute-bound',
    'GEMM': 'Flat', 'YOLO': 'Flat', 'ResNet-GPU': 'Flat', 'NodeJS': 'Flat',
    'Training': 'GPU-mixed', 'GPU-LLM': 'GPU-memory',
}

for wl in wl_order:
    if wl not in wl_data:
        continue
    sb = solo_bl.get(wl, {})
    rows4.append({
        'workload': wl, 'pattern': pattern_map.get(wl, ''),
        'source': 'Solo', 'combo': '-', 'cores': 10,
        'cpu_pct': round(sb.get('cpu_pct', 0), 1),
        'gpu_power_W': round(sb.get('gpu_W', 0), 1),
        'gpu_sm_pct': round(sb.get('sm_pct', 0), 1),
    })
    for e in sorted(wl_data[wl], key=lambda x: (x['combo'], x['cores'])):
        rows4.append({
            'workload': '', 'pattern': '',
            'source': f"Conc {e['side']}", 'combo': e['combo'],
            'cores': e['cores'],
            'cpu_pct': round(e['cpu_pct'], 1),
            'gpu_power_W': round(e['gpu_W'], 1),
            'gpu_sm_pct': round(e['sm_pct'], 1),
        })
    rows4.append({k: '' for k in rows4[0]})

notes4 = [
    {'workload': '[Note]'},
    {'workload': 'Phase 2 workloads', 'pattern': 'Training: GPU fine-tuning (IPC 1.68), ffmpeg: CPU x264 encoding (IPC 1.32), GPU-LLM: OPT-1.3B fp16 inference (IPC 0.66).'},
    {'workload': 'IPC (0225)', 'pattern': 'ResNet-CPU=1.75, ffmpeg=1.32, LLM=0.48, NodeJS=1.24, GEMM=2.29, YOLO=1.39, ResNet-GPU=1.01, Training=1.68, GPU-LLM=0.66.'},
]
rows4.extend(notes4)

out4 = pd.DataFrame(rows4)
out4.to_csv(os.path.join(BASE, 'gsheet_scaling_pattern.csv'), index=False)
print('[4/7] gsheet_scaling_pattern.csv')


# ═══════════════════════════════════════════════════════════════════
# Sheet 5: Cross-Combo Verification (Phase 1 + Phase 2)
# ═══════════════════════════════════════════════════════════════════
rows5 = []
cross_data = []

# LLM (CPU%): B vm_b, D vm_a, F vm_b, G vm_b
s_llm = solo[solo['vm_a'] == 'LLM']
if len(s_llm) > 0:
    llm_entries = [{'cores': 10, 'source': 'Solo', 'combo': '-', 'partner': '-', 'value': round(s_llm.iloc[0]['a_cpu_pct'], 1)}]
    for combo, side, cores_col, cpu_col in [
        ('B', 'vm_b', 'vm_b_cores', 'b_cpu_pct'),
        ('D', 'vm_a', 'vm_a_cores', 'a_cpu_pct'),
        ('F', 'vm_b', 'vm_b_cores', 'b_cpu_pct'),
        ('G', 'vm_b', 'vm_b_cores', 'b_cpu_pct'),
    ]:
        subset = conc[conc['combo'] == combo]
        if len(subset) == 0:
            continue
        partner = combos[combo][0] if side == 'vm_b' else combos[combo][1]
        for _, r in subset.sort_values(cores_col).iterrows():
            if pd.notna(r[cores_col]):
                llm_entries.append({'cores': int(r[cores_col]), 'source': f'Conc {side}', 'combo': combo, 'partner': partner, 'value': round(r[cpu_col], 1)})
    cross_data.append(('LLM', 'CPU%', 'Memory-bound', llm_entries))

# ffmpeg (CPU%): G vm_a, H vm_b
s_ff = solo[solo['vm_a'] == 'ffmpeg']
if len(s_ff) > 0:
    ff_entries = [{'cores': 10, 'source': 'Solo', 'combo': '-', 'partner': '-', 'value': round(s_ff.iloc[0]['a_cpu_pct'], 1)}]
    for combo, side, cores_col, cpu_col in [
        ('G', 'vm_a', 'vm_a_cores', 'a_cpu_pct'),
        ('H', 'vm_b', 'vm_b_cores', 'b_cpu_pct'),
    ]:
        subset = conc[conc['combo'] == combo]
        if len(subset) == 0:
            continue
        partner = combos[combo][0] if side == 'vm_b' else combos[combo][1]
        for _, r in subset.sort_values(cores_col).iterrows():
            if pd.notna(r[cores_col]):
                ff_entries.append({'cores': int(r[cores_col]), 'source': f'Conc {side}', 'combo': combo, 'partner': partner, 'value': round(r[cpu_col], 1)})
    cross_data.append(('ffmpeg', 'CPU%', 'Compute-bound', ff_entries))

# GEMM (GPU Power): A vm_b(GPU1), C vm_a(GPU0), I vm_b(GPU1)
s_gemm = solo[solo['vm_a'] == 'GEMM']
if len(s_gemm) > 0:
    gemm_entries = [{'cores': 10, 'source': 'Solo', 'combo': '-', 'partner': '-', 'value': round(s_gemm.iloc[0]['gpu0_W'], 1)}]
    for combo, gpu_col, cores_col in [
        ('A', 'gpu1_W', 'vm_b_cores'),
        ('C', 'gpu0_W', 'vm_a_cores'),
        ('I', 'gpu1_W', 'vm_b_cores'),
    ]:
        subset = conc[conc['combo'] == combo]
        if len(subset) == 0:
            continue
        partner = combos[combo][0] if cores_col == 'vm_b_cores' else combos[combo][1]
        for _, r in subset.sort_values(cores_col).iterrows():
            if pd.notna(r[cores_col]):
                gemm_entries.append({'cores': int(r[cores_col]), 'source': f'Conc ({gpu_col})', 'combo': combo, 'partner': partner, 'value': round(r[gpu_col], 1)})
    cross_data.append(('GEMM', 'GPU Power (W)', 'Flat', gemm_entries))

# Training (GPU Power): H vm_a(GPU0), J vm_a(GPU0)
s_tr = solo[solo['vm_a'] == 'Training']
if len(s_tr) > 0:
    tr_entries = [{'cores': 10, 'source': 'Solo', 'combo': '-', 'partner': '-', 'value': round(s_tr.iloc[0]['gpu0_W'], 1)}]
    for combo in ['H', 'J']:
        subset = conc[conc['combo'] == combo]
        if len(subset) == 0:
            continue
        for _, r in subset.sort_values('vm_a_cores').iterrows():
            if pd.notna(r['vm_a_cores']):
                tr_entries.append({'cores': int(r['vm_a_cores']), 'source': 'Conc vm_a (GPU0)', 'combo': combo, 'partner': combos[combo][1], 'value': round(r['gpu0_W'], 1)})
    cross_data.append(('Training', 'GPU Power (W)', 'GPU-mixed', tr_entries))

# Build rows
for wl, metric, pattern, entries in cross_data:
    first = True
    for e in sorted(entries, key=lambda x: (x['cores'], x['combo'])):
        rows5.append({
            'workload': wl if first else '',
            'metric': metric if first else '',
            'pattern': pattern if first else '',
            'cores': e['cores'],
            'source': e['source'],
            'combo': e['combo'],
            'partner': e['partner'],
            'value': e['value'],
        })
        first = False
    rows5.append({k: '' for k in rows5[0]})

notes5 = [
    {'workload': '[Note]'},
    {'workload': 'Verification goal', 'metric': 'Verify that the same workload at the same core count has identical energy profiles regardless of partner.'},
    {'workload': 'Phase 2 additions', 'metric': 'LLM: +G combo (vs ffmpeg), ffmpeg: G+H combo, GEMM: +I combo (vs GPU-LLM), Training: H+J combo.'},
]
rows5.extend(notes5)

out5 = pd.DataFrame(rows5)
out5.to_csv(os.path.join(BASE, 'gsheet_cross_combo.csv'), index=False)
print('[5/7] gsheet_cross_combo.csv')


# ═══════════════════════════════════════════════════════════════════
# Sheet 6: IPC & Cache Characteristics (9 workloads)
# ═══════════════════════════════════════════════════════════════════
perf_csv = os.path.join(BASE, 'data', 'perf_stat_summary.csv')
if os.path.exists(perf_csv):
    perf = pd.read_csv(perf_csv)
    name_map = {
        'resnet': 'ResNet-CPU', 'llm': 'LLM', 'nodejs': 'NodeJS',
        'resnet_gpu': 'ResNet-GPU', 'gemm': 'GEMM', 'yolo': 'YOLO',
        'training': 'Training', 'ffmpeg': 'ffmpeg', 'gpu_llm': 'GPU-LLM',
    }
    cat_map = {
        'GEMM': 'GPU-dominant', 'ResNet-GPU': 'GPU-dominant', 'YOLO': 'GPU-dominant',
        'Training': 'GPU-mixed', 'GPU-LLM': 'GPU-memory',
        'ResNet-CPU': 'Compute-bound', 'ffmpeg': 'Compute-bound',
        'LLM': 'Memory-bound', 'NodeJS': 'Light',
    }
    perf['workload_name'] = perf['workload'].map(name_map)
    perf['category'] = perf['workload_name'].map(cat_map)

    rows6 = []
    wl_order6 = ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO', 'ResNet-CPU', 'ffmpeg', 'LLM', 'NodeJS']
    for wl in wl_order6:
        subset = perf[perf['workload_name'] == wl]
        if len(subset) == 0:
            continue
        r = subset.iloc[0]
        llc_loads = r['llc_loads'] if r['llc_loads'] > 0 else 1
        cache_refs = r['cache_references'] if r['cache_references'] > 0 else 1
        rows6.append({
            'workload': wl,
            'category': cat_map.get(wl, ''),
            'IPC': round(r['ipc'], 2),
            'instructions': int(r['instructions']),
            'cycles': int(r['cycles']),
            'LLC_load_misses': int(r['llc_load_misses']),
            'LLC_loads': int(r['llc_loads']),
            'LLC_miss_rate_pct': round(r['llc_load_misses'] / llc_loads * 100, 1),
            'cache_references': int(r['cache_references']),
            'cache_misses': int(r['cache_misses']),
            'cache_miss_rate_pct': round(r['cache_misses'] / cache_refs * 100, 1),
        })

    notes6 = [
        {k: '' for k in rows6[0]},
        {'workload': '[Note]'},
        {'workload': 'Measurement', 'category': 'perf stat -e instructions,cycles,LLC-load-misses,LLC-loads,cache-references,cache-misses. Solo 10 cores/14GB, 60s.'},
        {'workload': 'IPC', 'category': 'Instructions Per Cycle. Compute-bound: IPC>1.0, Memory-bound: IPC<0.7.'},
        {'workload': 'Phase 2', 'category': 'Training=1.68 (GPU mixed, high CPU-side IPC), ffmpeg=1.32 (Compute-bound confirmed), GPU-LLM=0.66 (GPU memory-heavy, low CPU).'},
        {'workload': 'IPC reproducibility', 'category': 'CPU compute-bound: ResNet-CPU(1.75) ~ Training(1.68) ~ ffmpeg(1.32) > 1.0. Memory-bound: LLM(0.48) ~ GPU-LLM(0.66) < 0.7.'},
    ]
    rows6.extend(notes6)

    out6 = pd.DataFrame(rows6)
    out6.to_csv(os.path.join(BASE, 'gsheet_IPC_cache.csv'), index=False)
    print('[6/7] gsheet_IPC_cache.csv')
else:
    print('[6/7] SKIP: perf_stat_summary.csv not found')


# ═══════════════════════════════════════════════════════════════════
# Sheet 7: Anchor-Spectrum Classification (new)
# ═══════════════════════════════════════════════════════════════════
rows7 = []

idle_row = solo[solo['vm_a'] == 'idle']
if len(idle_row) > 0:
    idle = idle_row.iloc[0]
    idle_total = idle['rapl_W'] + idle['gpu0_W'] + idle['gpu1_W'] + idle['dram_W']

    wl_all = ['GEMM', 'GPU-LLM', 'Training', 'ResNet-GPU', 'YOLO', 'ResNet-CPU', 'ffmpeg', 'LLM', 'NodeJS']
    anchor_set = {'GEMM', 'ResNet-CPU', 'NodeJS'}
    category_map = {
        'GEMM': 'GPU-dominant (Anchor, upper bound)',
        'GPU-LLM': 'GPU-dominant (Memory-heavy)',
        'Training': 'GPU-dominant (Mixed)',
        'ResNet-GPU': 'GPU-dominant (Inference)',
        'YOLO': 'GPU-dominant (Lightweight)',
        'ResNet-CPU': 'CPU Compute-bound (Anchor)',
        'ffmpeg': 'CPU Compute-bound (Realistic)',
        'LLM': 'CPU Memory-bound',
        'NodeJS': 'CPU Light (Anchor, lower bound)',
    }

    for wl in wl_all:
        s = solo[solo['vm_a'] == wl]
        if len(s) == 0:
            continue
        r = s.iloc[0]
        comp_total = r['rapl_W'] + r['gpu0_W'] + r['gpu1_W'] + r['dram_W']
        active = comp_total - idle_total
        gpu_active = max(0, r['gpu0_W'] - idle['gpu0_W'])
        gpu_frac = gpu_active / max(active, 1) * 100

        rows7.append({
            'workload': wl,
            'role': 'Anchor' if wl in anchor_set else 'Spectrum',
            'category': category_map.get(wl, ''),
            'active_W': round(active, 1),
            'gpu_fraction_pct': round(gpu_frac, 1),
            'cpu_pct': round(r['a_cpu_pct'], 1),
            'gpu_sm_pct': round(r['a_sm_pct'], 1) if pd.notna(r['a_sm_pct']) else 0,
            'rapl_W': round(r['rapl_W'], 1),
            'gpu0_W': round(r['gpu0_W'], 1),
            'dram_W': round(r['dram_W'], 2),
        })

    notes7 = [
        {k: '' for k in rows7[0]},
        {'workload': '[Anchor-Spectrum Framing]'},
        {'workload': 'Anchor', 'role': 'GEMM (GPU upper bound), ResNet-CPU (CPU compute anchor), NodeJS (lower bound). Intentional extreme points to establish classification boundaries.'},
        {'workload': 'Spectrum', 'role': 'Remaining 6 workloads placed on a realistic spectrum between anchors. Phase 2 added 3 to fill gaps.'},
        {'workload': 'GPU fraction', 'role': 'GPU active energy / total active energy. 50% threshold for GPU-dominant vs CPU-dominant.'},
    ]
    rows7.extend(notes7)

    out7 = pd.DataFrame(rows7)
    out7.to_csv(os.path.join(BASE, 'gsheet_anchor_spectrum.csv'), index=False)
    print('[7/7] gsheet_anchor_spectrum.csv')
else:
    print('[7/7] SKIP: idle solo data not found')

print()
print('Done! 7 gsheet CSVs saved.')
