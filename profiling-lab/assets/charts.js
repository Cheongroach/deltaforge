// SPDX-License-Identifier: Apache-2.0
(function () {
  "use strict";
  const data = window.DELTAFORGE_DATA;
  const $ = id => document.getElementById(id);
  const esc = text => String(text).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const style = getComputedStyle(document.documentElement);
  const color = Object.fromEntries(["bg", "bg2", "ink", "muted", "rule", "accent", "accent2", "soft"].map(name => [name, style.getPropertyValue("--" + name).trim()]));
  const charts = new Map();
  const storageKey = "deltaforge-profiling-lab-v2";
  let notes = { answers: {}, checks: {} };
  try { notes = JSON.parse(localStorage.getItem(storageKey)) || notes; } catch (_) { /* file:// may deny storage */ }
  notes.answers ||= {}; notes.checks ||= {};
  const status = message => { $("global-status").textContent = message; };
  function save() {
    try { localStorage.setItem(storageKey, JSON.stringify(notes)); } catch (_) { status("当前浏览器限制本地保存；你仍可以导出练习 JSON。"); }
    $("progress-state").textContent = `已回答 ${Object.keys(notes.answers).length}/3，复跑清单 ${Object.values(notes.checks).filter(Boolean).length}/5。练习保存在当前浏览器，可随时导出。`;
  }
  function download(name, object) {
    const blob = new Blob([JSON.stringify(object, null, 2)], { type: "application/json" });
    const link = document.createElement("a"); link.href = URL.createObjectURL(blob); link.download = name;
    link.click(); setTimeout(() => URL.revokeObjectURL(link.href), 2000);
  }
  function chart(id, option) {
    if (!window.echarts) { $(id).textContent = "图表库未加载，请确认 _shared/js 随页面完整解压。表格数据仍可阅读。"; return; }
    let instance = charts.get(id);
    if (!instance) { instance = echarts.init($(id), null, { renderer: "svg" }); charts.set(id, instance); }
    instance.setOption({ animation: false, color: [color.accent, color.accent2, color.muted], textStyle: { color: color.ink, fontFamily: style.getPropertyValue("--font-ui").trim() }, aria: { enabled: true }, ...option }, true);
    return instance;
  }
  window.addEventListener("resize", () => charts.forEach(instance => instance.resize()));
  function tab(name) {
    document.querySelectorAll("[data-tab]").forEach(button => button.setAttribute("aria-selected", String(button.dataset.tab === name)));
    document.querySelectorAll(".tab-section").forEach(section => { section.hidden = section.id !== "section-" + name; });
    requestAnimationFrame(() => {
      if (name === "measured") renderCase();
      if (name === "trace") renderTrace();
      if (name === "sim") renderSim();
      charts.forEach(instance => instance.resize());
    });
  }
  document.querySelectorAll("[data-tab]").forEach(button => button.addEventListener("click", () => tab(button.dataset.tab)));
  document.querySelectorAll("[data-goto]").forEach(button => button.addEventListener("click", () => { tab(button.dataset.goto); window.scrollTo({ top: 0, behavior: "smooth" }); }));
  document.querySelector(".tabs").addEventListener("keydown", event => {
    if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return;
    const buttons = [...document.querySelectorAll("[data-tab]")], current = buttons.indexOf(document.activeElement);
    if (current < 0) return;
    event.preventDefault();
    let next = event.key === "Home" ? 0 : event.key === "End" ? buttons.length - 1 : (current + (event.key === "ArrowRight" ? 1 : -1) + buttons.length) % buttons.length;
    buttons[next].focus(); tab(buttons[next].dataset.tab);
  });

  const lessons = [
    ["01 固定问题", "先说清“哪一次执行”", "先选固定 shape、dtype、batch 和非零 state。只改一个变量；不要把 BF16 对 FP32、冷编译对热执行、低并发对高并发混为同一个比值。", "本项目第 02 页是固定种子随机夹具；它不是任务准确率或 vLLM 端到端测试。"],
    ["02 数值门禁", "快之前，先确认没算错", "比较 output 与 final state。FP32 state 必须交接，不能每一步重新清零。短序列数值通过，还需要检查 padding、GQA、强 gate 和连续 state handoff。", "前一轮的 8 项回归与设备数值检查是门禁；profile 并不替代这些检查。"],
    ["03 预热与编译", "首次编译移出采集窗口", "先执行导出／编译和四次预热，再开始 profile。首次使用 CUDA profiler 本身也可能增加开销，因此 profile 时间不能直接替代稳定延迟。", "在第 02 页查看 910B4 / T=128：约 5% 的节省，要很多次调用才能摊销编译。"],
    ["04 采小窗口", "标记范围，再打开 profiler", "torch.profiler 用 CPU/CUDA activities 选择采集层，record_function 给阶段命名。这里每种路径只采两次，并把同步留在阶段内，让你容易对应 CPU 标签与 GPU 执行。", "同步会改变流水与重叠，适合教学定位，不适合证明在线性能。"],
    ["05 读三种证据", "调用、执行、空洞分开看", "CPU total 含嵌套子调用；GPU kernel 时间来自设备。多 stream 的 kernel 时间可能重叠，要看时间区间并集。GPU 覆盖率不等于 SM occupancy。", "在第 03 页对比 decode_torch 与 decode_triton：先看 kernel 数，再看 CPU 提交与设备事件。"],
    ["06 验证假设", "改一项，再做无采集计时", "如果怀疑发射过多，就融合或改变 batch；如果怀疑带宽，先量访存与 DRAM 指标；如果 tile 变大却更慢，检查寄存器与 spill。最后重新跑相同协议的多组计时。", "没有 ncu 指标时，不要写“HBM 打满”或“occupancy 已优化”。把缺失证据列为下一步。"],
  ];
  function lesson(index) {
    [...$("lesson-menu").children].forEach((button, i) => button.classList.toggle("active", i === index));
    $("lesson-body").innerHTML = `<span class="pill">${esc(lessons[index][0])}</span><h3 style="margin-top:14px">${esc(lessons[index][1])}</h3><p>${esc(lessons[index][2])}</p><div class="notice"><p>${esc(lessons[index][3])}</p></div>`;
  }
  lessons.forEach((entry, index) => { const button = document.createElement("button"); button.textContent = entry[0]; button.addEventListener("click", () => lesson(index)); $("lesson-menu").append(button); });
  lesson(0);
  if (!data) { status("缺少 assets/data.js，请完整解压离线包。"); return; }
  $("event-count").textContent = data.measurements.length;
  $("measurement-count").textContent = data.measurements.filter(row => row.event === "measurement").length;

  const cases = data.summary;
  const shapeText = shape => `B=${shape.B}, T=${shape.T}, Hk/Hv=${shape.Hk}/${shape.Hv}, K/V=${shape.K}/${shape.V}, ${shape.dtype}`;
  const sameShape = (a, b) => ["B", "T", "Hk", "Hv", "K", "V", "dtype"].every(key => a[key] === b[key]);
  const caseLabel = row => `${row.run.startsWith("h20") ? "H20" : "910B4"} / ${row.kind} / T=${row.shape.T}, B=${row.shape.B}, H=${row.shape.Hv}`;
  cases.forEach((row, index) => { $("case-select").add(new Option(caseLabel(row), String(index))); });
  const initialCase = cases.findIndex(row => row.run === "npu-constants" && row.shape.T === 512);
  $("case-select").value = String(Math.max(0, initialCase));
  function lookup(row, candidate) {
    return data.measurements.find(event => event.event === "measurement" && event.run === row.run && sameShape(event.shape, row.shape) &&
      (row.kind === "prefill" ? event.implementation === (candidate ? "fx_lowered_compiled" : "portable_chunk_eager") && event.chunk_size === row.chunk_size
        : event.implementation === (candidate ? "triton_fused_decode" : "torch_token_recurrence") &&
          (!candidate || event.value_tile === row.value_tile && event.num_warps === row.num_warps)));
  }
  function renderCase() {
    const row = cases[Number($("case-select").value)], baseline = lookup(row, false), candidate = lookup(row, true);
    $("case-contract").textContent = `${shapeText(row.shape)}；state=FP32；${row.kind === "prefill" ? "chunk=" + row.chunk_size : "tile=" + row.value_tile + ", warps=" + row.num_warps}。4 次预热，7 组重复，每组 5 次调用。`;
    $("base-ms").textContent = row.baseline_ms.toFixed(4) + " ms";
    $("opt-ms").textContent = row.optimized_ms.toFixed(4) + " ms";
    $("case-speed").textContent = row.speedup.toFixed(3) + "×";
    $("baseline-label").textContent = row.kind === "prefill" ? "同 chunk eager 中位数" : "自实现 PyTorch 中位数";
    const slower = row.speedup < 1;
    $("case-conclusion").innerHTML = `<strong>${slower ? "这组编译候选更慢，应保留 eager。" : "这组候选更快，但仍只是本轮扫描结果。"}</strong><p>${slower ? "数学等价与编译成功，不保证后端 schedule 更好。下一步应对比 kernel、提交空洞与图执行开销，不能凭时间数字确定根因。" : "独立复跑确认之后再选择；这个比值没有使用 FLA 或 vLLM 作为基线。"}</p>`;
    $("compile-detail").textContent = row.kind === "decode" ? "这里没有把 Triton 首次 JIT 成本纳入稳定调用延迟；这组数据不能回答首次请求耗时。" :
      `本轮导出 + 编译 + 首次执行 ${row.compile_seconds.toFixed(3)} s；` + (row.break_even_calls === null ? "候选每次更慢，没有性能回本次数。" : `估算约 ${row.break_even_calls.toLocaleString("zh-CN")} 次调用回本。`);
    chart("sample-chart", {
      tooltip: { trigger: "axis", appendToBody: true },
      legend: { textStyle: { color: color.muted }, bottom: 0 },
      grid: { left: 65, right: 24, top: 24, bottom: 65 },
      xAxis: { type: "category", data: baseline.wall_ms_samples.map((_, i) => `组 ${i + 1}`), axisLabel: { color: color.muted }, axisLine: { lineStyle: { color: color.rule } } },
      yAxis: { type: "value", name: "wall ms", nameTextStyle: { color: color.muted }, axisLabel: { color: color.muted }, splitLine: { lineStyle: { color: color.rule } } },
      series: [
        { type: "line", name: "基线", data: baseline.wall_ms_samples, symbolSize: 8, lineStyle: { color: color.accent }, itemStyle: { color: color.accent } },
        { type: "line", name: "候选", data: candidate.wall_ms_samples, symbolSize: 8, lineStyle: { color: color.accent2 }, itemStyle: { color: color.accent2 } },
      ],
    });
    $("case-table").innerHTML = cases.map((entry, i) => `<tr class="${Number($("case-select").value) === i ? "picked" : ""}"><td>${esc(caseLabel(entry))}</td><td>${esc(shapeText(entry.shape))}</td><td class="numeric">${entry.baseline_ms.toFixed(4)}</td><td class="numeric">${entry.optimized_ms.toFixed(4)}</td><td class="numeric">${entry.speedup.toFixed(3)}×</td><td>${entry.recommendation === "eager" ? "保留 eager" : "独立确认候选"}</td></tr>`).join("");
  }
  $("case-select").addEventListener("change", renderCase);

  const profiles = { ...data.profiles };
  const titleForProfile = profile => profile.device === "cuda" ? `${profile.device_name} / CPU + CUDA` : "CPU / 仅主机执行";
  Object.entries(profiles).forEach(([key, profile]) => $("profile-select").add(new Option(titleForProfile(profile), key)));
  $("profile-select").value = profiles.cuda ? "cuda" : Object.keys(profiles)[0];
  function currentProfile() { return profiles[$("profile-select").value]; }
  function phases() {
    const profile = currentProfile(); $("phase-select").replaceChildren();
    profile.phases.forEach((phase, i) => $("phase-select").add(new Option(phase.name.replace("deltaforge/", ""), String(i))));
    const tritonIndex = profile.phases.findIndex(phase => phase.name.includes("decode_triton"));
    $("phase-select").value = String(Math.max(0, tritonIndex));
  }
  function union(events) {
    if (!events.length) return 0;
    const sorted = events.map(event => [event.start_us, event.start_us + event.duration_us]).sort((a, b) => a[0] - b[0]);
    let [start, end] = sorted[0], total = 0;
    sorted.slice(1).forEach(([left, right]) => { if (left <= end) end = Math.max(end, right); else { total += end - start; start = left; end = right; } });
    return total + end - start;
  }
  function renderTrace() {
    const profile = currentProfile(), phase = profile.phases[Number($("phase-select").value)];
    if (!phase) return;
    const allInPhase = profile.timeline.filter(event => event.start_us >= phase.start_us - .001 && event.start_us < phase.start_us + phase.duration_us);
    const kernelEvents = allInPhase.filter(event => event.category === "kernel");
    const coverage = union(kernelEvents.map(event => ({ ...event, duration_us: Math.min(event.duration_us, phase.start_us + phase.duration_us - event.start_us) })));
    $("phase-ms").textContent = (phase.duration_us / 1000).toFixed(3) + " ms";
    $("kernel-count").textContent = kernelEvents.length;
    $("gpu-coverage").textContent = profile.gpu_trace_available ? (coverage / 1000).toFixed(3) + " ms" : "无 GPU 数据";
    $("profile-contract").textContent = `profile 形状：prefill B=1,T=128,H=4,K/V=32,chunk=32；decode B=4,T=1,H=16,K/V=128。${titleForProfile(profile)}；PyTorch ${profile.versions?.torch || "未记录"}；同步在 CUDA 阶段内。与第 02 页的不同 shape 不能直接混用。`;
    if (profile.imported) $("profile-contract").textContent = "用户导入的 Chrome trace；未校验形状、版本、数值或计时协议。内容只在本页本地处理。";
    $("trace-note").innerHTML = profile.gpu_trace_available ? `<strong>GPU 时间线覆盖不等于 GPU 硬件利用率。</strong><p>这里没有 ncu 的 SM、Tensor Core、DRAM 指标。CPU total 还会嵌套；阶段内同步会改变重叠和时间。</p>` : `<strong>这份采集只有 CPU 事件。</strong><p>选择“只看 GPU kernel”会显示空图，不会补造 GPU 时间线。CPU 算子耗时无法证明 H20 或 910B4 的性能。</p>`;
    const filter = $("trace-filter").value.toLowerCase(), category = $("category-select").value;
    const selected = allInPhase.filter(event => (category === "all" || event.category === category || category === "cuda_runtime" && event.category === "cuda_driver") && event.name.toLowerCase().includes(filter));
    const groups = new Map();
    selected.forEach(event => { if (!groups.has(event.lane)) groups.set(event.lane, []); groups.get(event.lane).push(event); });
    const rows = [], laneNames = [], laneMap = new Map();
    groups.forEach((group, lane) => {
      const stack = [];
      group.sort((a, b) => a.start_us - b.start_us || b.duration_us - a.duration_us).forEach(event => {
        while (stack.length && stack[stack.length - 1] <= event.start_us + .0001) stack.pop();
        const depth = event.category === "cpu_op" ? stack.length : 0;
        if (event.category === "cpu_op") stack.push(event.start_us + event.duration_us);
        if (depth > 5) return;
        const label = event.category === "cpu_op" ? `CPU 算子 · 嵌套 ${depth}` : event.category === "kernel" ? "CUDA kernel" : event.category === "cuda_runtime" ? "CUDA runtime" : event.category === "cuda_driver" ? "CUDA driver" : "阶段标签";
        const unique = lane + "/" + depth;
        if (!laneMap.has(unique)) { laneMap.set(unique, laneNames.length); laneNames.push(label); }
        rows.push([laneMap.get(unique), (event.start_us - phase.start_us) / 1000,
          Math.min(event.start_us + event.duration_us - phase.start_us, phase.duration_us) / 1000,
          event.name, event.category, event.duration_us]);
      });
    });
    const plotted = rows.slice(0, 1800);
    $("trace-chart").style.height = Math.max(310, laneNames.length * 35 + 95) + "px";
    const instance = chart("trace-chart", {
      tooltip: { appendToBody: true, formatter: item => `${esc(item.value[3])}<br>${esc(item.value[4])}<br>持续 ${Number(item.value[5]).toFixed(3)} µs` },
      grid: { top: 25, left: 126, right: 45, bottom: 70 },
      xAxis: { min: 0, max: phase.duration_us / 1000, type: "value", name: "ms", axisLabel: { color: color.muted }, splitLine: { lineStyle: { color: color.rule } } },
      yAxis: { inverse: true, type: "category", data: laneNames, axisLabel: { color: color.muted, fontSize: 11 }, axisLine: { lineStyle: { color: color.rule } } },
      dataZoom: [{ type: "slider", xAxisIndex: 0, height: 20, bottom: 12, borderColor: color.rule, fillerColor: color.accent + "22", textStyle: { color: color.muted } }, { type: "inside", xAxisIndex: 0 }],
      series: [{
        type: "custom", data: plotted, encode: { x: [1, 2], y: 0 },
        renderItem: (params, api) => {
          const left = api.coord([api.value(1), api.value(0)]), right = api.coord([api.value(2), api.value(0)]), height = Math.min(21, api.size([0, 1])[1] * .65);
          const fill = api.value(4) === "kernel" ? color.accent : ["cuda_runtime", "cuda_driver"].includes(api.value(4)) ? color.accent2 : api.value(4) === "user_annotation" ? color.soft : color.muted + "66";
          const shape = echarts.graphic.clipRectByRect({ x: left[0], y: left[1] - height / 2, width: Math.max(.8, right[0] - left[0]), height }, { x: params.coordSys.x, y: params.coordSys.y, width: params.coordSys.width, height: params.coordSys.height });
          return shape && { type: "rect", shape, style: { fill, stroke: api.value(4) === "user_annotation" ? color.accent : fill, lineWidth: .5 } };
        },
      }],
    });
    if (instance) { instance.off("click"); instance.on("click", item => { if (item.value) $("event-detail").textContent = `${item.value[3]} | ${item.value[4]} | 开始 ${item.value[1].toFixed(4)} ms | 持续 ${Number(item.value[5]).toFixed(3)} µs。此处属于带 profiler 干预的执行。`; }); }
    $("trace-count").textContent = `当前阶段 ${allInPhase.length} 个原始事件，筛选后 ${selected.length} 个；图中绘制 ${plotted.length} 个，CPU 嵌套显示到第 5 层。缩放用于观察间隙，不修改原始时间。`;
    const operators = [...(profile.operator_table || [])].sort((a, b) => b.self_cpu_us - a.self_cpu_us).slice(0, 40);
    $("op-table").innerHTML = operators.map(entry => `<tr><td>${esc(entry.name)}</td><td class="numeric">${entry.calls}</td><td class="numeric">${entry.self_cpu_us.toFixed(2)}</td><td class="numeric">${entry.total_cpu_us.toFixed(2)}</td><td class="numeric">${Number(entry.self_device_us).toFixed(2)}</td><td class="numeric">${entry.self_cpu_memory_bytes}</td></tr>`).join("");
  }
  $("profile-select").addEventListener("change", () => { phases(); renderTrace(); });
  ["phase-select", "category-select"].forEach(id => $(id).addEventListener("change", renderTrace));
  $("trace-filter").addEventListener("input", renderTrace);
  $("download-profile").addEventListener("click", () => download("deltaforge-profile-summary.json", currentProfile()));
  $("trace-import").addEventListener("change", async () => {
    const file = $("trace-import").files[0];
    if (!file) return;
    if (file.size > 25 * 1024 * 1024) { status("教学导入上限 25 MiB；更大 trace 请使用 Perfetto。"); return; }
    try {
      const raw = JSON.parse(await file.text());
      if (!Array.isArray(raw.traceEvents)) throw new Error("没有 traceEvents 数组");
      const events = raw.traceEvents.filter(event => event.ph === "X" && Number.isFinite(event.ts) && Number.isFinite(event.dur) && event.dur > 0 && typeof event.name === "string");
      if (!events.length) throw new Error("没有可显示的完整 duration 事件");
      const low = events.reduce((value, event) => Math.min(value, event.ts), Infinity), high = events.reduce((value, event) => Math.max(value, event.ts + event.dur), -Infinity);
      const lanes = new Map();
      const timeline = events.map(event => {
        let category = event.cat || "cpu_op";
        if (category.toLowerCase().includes("kernel")) category = "kernel";
        if (!["cpu_op", "kernel", "cuda_runtime", "cuda_driver", "user_annotation", "gpu_memcpy"].includes(category)) category = "cpu_op";
        const key = `${category}/${event.pid}/${event.tid}`;
        if (!lanes.has(key)) lanes.set(key, category + " / lane " + (lanes.size + 1));
        return { name: event.name, category, lane: lanes.get(key), start_us: event.ts - low, duration_us: event.dur };
      });
      const annotations = events.filter(event => event.cat === "user_annotation" && event.name.startsWith("deltaforge/"));
      profiles.imported = { imported: true, device: "imported", phases: annotations.length ? annotations.map(event => ({ name: event.name, start_us: event.ts - low, duration_us: event.dur })) : [{ name: "导入的完整窗口", start_us: 0, duration_us: high - low }], timeline, gpu_trace_available: timeline.some(event => event.category === "kernel"), operator_table: [] };
      if (![...$("profile-select").options].some(option => option.value === "imported")) $("profile-select").add(new Option("用户导入 / 本地处理", "imported"));
      $("profile-select").value = "imported"; phases(); renderTrace(); status("已导入；内容未上传到服务器。");
    } catch (error) { status("无法导入：" + error.message); }
  });
  phases();

  const presets = {
    launch: { kernels: 16, launch: 8, bytes: 8, bw: 800, flops: 1, throughput: 20, fused: 1, reuse: 25 },
    bandwidth: { kernels: 16, launch: 2, bytes: 1024, bw: 400, flops: 1, throughput: 100, fused: 1, reuse: 50 },
    compute: { kernels: 16, launch: 2, bytes: 8, bw: 1200, flops: 100, throughput: 20, fused: 1, reuse: 50 },
  };
  function setPreset() { Object.entries(presets[$("sim-preset").value]).forEach(([key, value]) => { $("sim-" + key).value = value; }); renderSim(); }
  function renderSim() {
    const params = Object.fromEntries(Object.keys(presets.launch).map(key => [key, Number($("sim-" + key).value)]));
    const fused = Math.min(params.fused, params.kernels);
    $("sim-fused").max = params.kernels;
    if (params.fused !== fused) $("sim-fused").value = fused;
    Object.keys(params).forEach(key => { $("sim-" + key + "-out").textContent = key === "fused" ? fused : params[key]; });
    const memory = params.bytes * 1024 * 1024 / (params.bw * 1e9) * 1e6;
    const compute = params.flops / params.throughput * 1000;
    const beforeCompute = Math.max(memory, compute), afterCompute = Math.max(memory * (1 - params.reuse / 100), compute);
    const originalLaunch = params.kernels * params.launch, fusedLaunch = fused * params.launch;
    const before = originalLaunch + beforeCompute, after = fusedLaunch + afterCompute;
    $("sim-before").textContent = before.toFixed(1) + " µs"; $("sim-after").textContent = after.toFixed(1) + " µs";
    $("sim-explain").textContent = `教学预测 ${ (before / after).toFixed(2) }×。原关键设备时间由${memory > compute ? "访存" : "计算"}决定：访存 ${memory.toFixed(1)} µs，计算 ${compute.toFixed(1)} µs。总计算量保持不变。`;
    chart("sim-chart", {
      tooltip: { trigger: "axis", axisPointer: { type: "shadow" }, appendToBody: true },
      legend: { bottom: 0, textStyle: { color: color.muted } },
      grid: { top: 24, left: 65, right: 25, bottom: 60 },
      xAxis: { type: "value", name: "µs", axisLabel: { color: color.muted }, splitLine: { lineStyle: { color: color.rule } } },
      yAxis: { type: "category", data: ["原路径（模拟）", "融合后（模拟）"], axisLabel: { color: color.muted }, axisLine: { lineStyle: { color: color.rule } } },
      series: [{ name: "发射", type: "bar", stack: "time", data: [originalLaunch, fusedLaunch], itemStyle: { color: color.accent2 }, barMaxWidth: 35 }, { name: "设备关键路径", type: "bar", stack: "time", data: [beforeCompute, afterCompute], itemStyle: { color: color.accent }, barMaxWidth: 35 }],
    });
    const f = Number($("amdahl-f").value) / 100, s = Number($("amdahl-s").value), whole = 1 / (1 - f + f / s);
    $("amdahl-f-out").textContent = (f * 100).toFixed(0) + "%"; $("amdahl-s-out").textContent = s.toFixed(1) + "×";
    $("amdahl-result").textContent = whole.toFixed(2) + "×";
    $("amdahl-explain").textContent = `只加速原来 ${ (f * 100).toFixed(0) }% 的耗时，整模型计算示例为 ${whole.toFixed(2)}×；不是 vLLM 实测。`;
  }
  Object.keys(presets.launch).forEach(key => $("sim-" + key).addEventListener("input", renderSim));
  ["amdahl-f", "amdahl-s"].forEach(id => $(id).addEventListener("input", renderSim));
  $("sim-preset").addEventListener("change", setPreset); $("sim-reset").addEventListener("click", setPreset);

  const questions = [
    { id: "regression", question: "910B4 / T=512 的 compiled 性能比为 0.811×。你会怎么做？", options: ["因为编译成功，立即采用 compiled", "保留 eager，再采 profile 检查这个形状", "把其他形状的加速比写到这里"], correct: 1, explanation: "基线 / 候选小于 1，表示候选延迟更高。保留 eager；时间数据只证明回归，不证明具体原因。到第 02 页对照七组样本。", next: "measured" },
    { id: "cpu_total", question: "CPU total 表里 aten::matmul、aten::bmm 都很大。能直接相加当作整段耗时吗？", options: ["可以，每行都是独立耗时", "不可以，调用有嵌套，先看 Self CPU 与时间线", "只要名称不同就可以"], correct: 1, explanation: "CPU total 会包含子调用。直接相加可能重复计数。GPU 多 stream 也可能重叠；时间区间并集只是覆盖时间，仍不是 SM 利用率。", next: "trace" },
    { id: "whole_model", question: "某算子占整模型 20% 的时间，算子快 5×。现在能宣称整模型也快 5× 吗？", options: ["可以，算子就是模型", "只能作为局部算子结果；计算示例约 1.19×", "肯定超过 5×"], correct: 1, explanation: "Amdahl 示例：1 / (0.8 + 0.2 / 5) ≈ 1.19。真实占比还要采样；与 vLLM 的比较需要整模型、相同配置和请求负载。", next: "sim" },
  ];
  function renderQuestions() {
    $("quiz-questions").innerHTML = questions.map((question, index) => {
      const answer = notes.answers[question.id];
      return `<div class="question"><h3>${index + 1}. ${esc(question.question)}</h3><div class="tools">${question.options.map((option, choice) => `<button data-answer="${question.id}" data-choice="${choice}" aria-pressed="${answer === choice}">${esc(option)}</button>`).join("")}</div>${answer === undefined ? "" : `<div class="feedback"><strong>${answer === question.correct ? "判断正确。" : "回到证据再看看。"}</strong> ${esc(question.explanation)}</div><button data-review="${question.next}">查看对应证据 →</button>`}</div>`;
    }).join("");
  }
  $("quiz-questions").addEventListener("click", event => {
    const answer = event.target.closest("[data-answer]"), review = event.target.closest("[data-review]");
    if (answer) { notes.answers[answer.dataset.answer] = Number(answer.dataset.choice); save(); renderQuestions(); }
    if (review) tab(review.dataset.review);
  });
  document.querySelectorAll("[data-check]").forEach(input => {
    input.checked = Boolean(notes.checks[input.dataset.check]);
    input.addEventListener("change", () => { notes.checks[input.dataset.check] = input.checked; save(); });
  });
  $("export-notes").addEventListener("click", () => download("deltaforge-my-profiling-practice.json", {
    version: "0.2", exported_at: new Date().toISOString(), evidence: "DeltaForge operator fixtures",
    simulation_is_not_measurement: true, notes,
    simulation: Object.fromEntries(Object.keys(presets.launch).map(key => [key, Number($("sim-" + key).value)])),
    questions: questions.map(question => ({ id: question.id, selected: notes.answers[question.id] ?? null, correct: notes.answers[question.id] === question.correct })),
  }));
  $("reset-notes").addEventListener("click", () => { notes = { answers: {}, checks: {} }; document.querySelectorAll("[data-check]").forEach(input => { input.checked = false; }); save(); renderQuestions(); });
  document.querySelectorAll("[data-copy]").forEach(button => button.addEventListener("click", async () => {
    const text = $(button.dataset.copy).textContent;
    try { await navigator.clipboard.writeText(text); status("命令已复制。"); }
    catch (_) { const selection = getSelection(), range = document.createRange(); range.selectNodeContents($(button.dataset.copy)); selection.removeAllRanges(); selection.addRange(range); status("浏览器限制剪贴板；命令已选中，请手动复制。"); }
  }));
  renderQuestions(); save(); renderCase(); renderTrace(); setPreset();
  status("全部图表与数据从本地文件加载；模拟结果不会自动写入真实测量。");
})();
