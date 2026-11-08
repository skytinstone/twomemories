"""Detailed live monitor for runs on this machine (the Mac 0.8B runs, D-30). Read-only: it never writes results.

  python scripts/watch.py                    # results/mac, Qwen3.5-0.8B, phases 1 2 3 5, refresh every 3 s
  python scripts/watch.py --interval 5 --out results/mac --model Qwen/Qwen3.5-0.8B

Shows: whether the run process is alive and which phase it runs, memory (process footprint incl. MPS, system, swap),
chip temperature (IOKit HID sensors, no sudo) and throttling, an overall bar with remaining time and finish
clock, one line per phase (done / running / waiting), the cell being written now with its own bar, the next
cells, a per-task cell map of the running phase, unaltered accuracy per task, and the last log lines.

Remaining time = sum over unfinished samples of (seconds per sample of that cell, else of that task, else overall)
x (wall clock / row seconds over recent rows, gaps over 60 s such as sleep excluded).
"""
from __future__ import annotations

import argparse
import datetime as dt
import re
import statistics as st
import subprocess
import sys
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_phase import load_conditions  # noqa: E402
from status import _Tail  # noqa: E402

from twomem import data  # noqa: E402
from twomem.runner import REPO, cell_hash, cell_path, subset_id_of  # noqa: E402

TZ = ZoneInfo("Asia/Seoul")
PHASES = ["phase1_probe", "phase2_bits", "phase3_tokens", "phase5_channels"]
PHASE_KO = {"phase1_probe": "개입 조건 비교", "phase2_bits": "state 비트 수", "phase3_tokens": "토큰 수 x KV 비트",
            "phase5_channels": "채널 제거 (drop / conv)"}
TASK_KO = {"docvqa": "DocVQA", "chartqa": "ChartQA", "textvqa": "TextVQA", "pope": "POPE"}
GAP = 60          # seconds; a longer gap between rows (sleep, model load) is left out of the overhead factor
RECENT = 400      # rows used for the overhead factor


def bar(frac: float, width: int) -> str:
    frac = min(max(frac, 0.0), 1.0)
    n = int(frac * width * 8)
    full, part = divmod(n, 8)
    done = "█" * full + ("▏▎▍▌▋▊▉"[part - 1] if part else "")
    return f"\033[32m{done}\033[0m\033[90m{'─' * (width - full - (1 if part else 0))}\033[0m"   # green done, grey rest


def dur(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}h{m:02d}m" if h else (f"{m}m{s:02d}s" if m else f"{s}s")


def clock(t: float, fmt: str = "%m-%d %H:%M") -> str:
    return dt.datetime.fromtimestamp(t, TZ).strftime(fmt)


def cond_ko(c: dict | None) -> str:
    if c is None:
        return "blind (이미지 없음)"
    lv = {"full": "원본", "4": "4bit", "8": "8bit", "6": "6bit", "8ef": "8bit+EF", "6ef": "6bit+EF",
          "noise": "노이즈", "zero": "0으로", "drop": "삭제", "zeroconv": "0으로+conv"}
    s = f"KV {lv.get(c['kv'], c['kv'])} / State {lv.get(c['state'], c['state'])}"
    if c.get("token_keep", 1.0) != 1.0:
        s += f" / 시각 토큰 {int(c['token_keep'] * 100)}% ({'균등' if c['token_method'] == 'uniform' else '무작위'})"
    s += " / 전 토큰" if c.get("scope") == "whole" else ""
    return s


def pad(s: str, width: int) -> str:
    """ljust by terminal columns (Hangul is two columns wide)."""
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    return s + " " * max(1, width - w)


def sh(*cmd: str) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=3).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def _mb(s: str) -> float:
    """top's MEM column ("7499M", "1.2G", "812K") in MB."""
    m = re.match(r"([\d.]+)([KMG])", s)
    return float(m.group(1)) * {"K": 1 / 1024, "M": 1, "G": 1024}[m.group(2)] if m else 0.0


def footprint_mb(pid: int) -> float:
    """Physical footprint (includes MPS/Metal buffers, which RSS leaves out), as Activity Monitor shows."""
    out = sh("top", "-l", "1", "-pid", str(pid), "-stats", "pid,mem").splitlines()
    return _mb(out[-1].split()[1]) if out and out[-1].split()[0] == str(pid) else 0.0


def system_memory() -> tuple[float, float, float, float]:
    """(total GB, used GB, swap used GB, swap total GB). used = total x (1 - free %) from memory_pressure."""
    total = int(sh("sysctl", "-n", "hw.memsize") or 0) / 2**30
    m = re.search(r"(\d+)%", sh("memory_pressure", "-Q"))
    used = total * (1 - int(m.group(1)) / 100) if m else 0.0
    sw = re.findall(r"(total|used) = ([\d.]+)M", sh("sysctl", "-n", "vm.swapusage"))
    swd = {k: float(v) / 1024 for k, v in sw}
    return total, used, swd.get("used", 0.0), swd.get("total", 0.0)


def running_process() -> tuple[int, str, str] | None:
    """(pid, phase, cpu) of a live run_phase.py, if any."""
    for line in sh("pgrep", "-fl", "python.*run_phase.py").splitlines():
        m = re.match(r"(\d+)\s+\S*python[\d.]*\s+\S*run_phase\.py\s.*--phase\s+(\S+)", line)   # not the shell or caffeinate
        if m and "--dry-run" not in line:
            pid = int(m.group(1))
            cpu = sh("ps", "-o", "%cpu=", "-p", str(pid))
            return pid, m.group(2), f"CPU {cpu}%" if cpu else ""
    return None


class Thermal:
    """Apple Silicon temperatures from the IOKit HID sensor services, no sudo (same source as the Stats app).

    chip = max over "PMU tdie*" (SoC die sensors); also battery ("gas gauge battery") and SSD ("NAND CH0 temp").
    Throttling comes from ``pmset -g therm`` (a recorded thermal or performance warning level).
    """

    def __init__(self):
        import ctypes
        import ctypes.util
        from ctypes import c_bool, c_char_p, c_double, c_int32, c_int64, c_long, c_uint32, c_void_p

        self.ct = ctypes
        io = ctypes.cdll.LoadLibrary(ctypes.util.find_library("IOKit"))
        cf = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreFoundation"))
        for f, res, args in [
            (cf.CFStringCreateWithCString, c_void_p, [c_void_p, c_char_p, c_uint32]),
            (cf.CFNumberCreate, c_void_p, [c_void_p, c_long, c_void_p]),
            (cf.CFDictionaryCreate, c_void_p, [c_void_p, c_void_p, c_void_p, c_long, c_void_p, c_void_p]),
            (cf.CFArrayGetCount, c_long, [c_void_p]),
            (cf.CFArrayGetValueAtIndex, c_void_p, [c_void_p, c_long]),
            (cf.CFStringGetCString, c_bool, [c_void_p, c_char_p, c_long, c_uint32]),
            (cf.CFRelease, None, [c_void_p]),
            (io.IOHIDEventSystemClientCreate, c_void_p, [c_void_p]),
            (io.IOHIDEventSystemClientSetMatching, None, [c_void_p, c_void_p]),
            (io.IOHIDEventSystemClientCopyServices, c_void_p, [c_void_p]),
            (io.IOHIDServiceClientCopyProperty, c_void_p, [c_void_p, c_void_p]),
            (io.IOHIDServiceClientCopyEvent, c_void_p, [c_void_p, c_int64, c_int32, c_int64]),
            (io.IOHIDEventGetFloatValue, c_double, [c_void_p, c_int32]),
        ]:
            f.restype, f.argtypes = res, args
        self.io, self.cf, self.utf8 = io, cf, 0x08000100

        def cfstr(s):
            return cf.CFStringCreateWithCString(None, s.encode(), self.utf8)

        def cfnum(v):
            x = ctypes.c_int32(v)
            return cf.CFNumberCreate(None, 3, ctypes.byref(x))   # kCFNumberSInt32Type

        keys = (c_void_p * 2)(cfstr("PrimaryUsagePage"), cfstr("PrimaryUsage"))
        vals = (c_void_p * 2)(cfnum(0xFF00), cfnum(5))            # Apple vendor page, temperature sensor usage
        kcb = c_void_p.in_dll(cf, "kCFTypeDictionaryKeyCallBacks")
        vcb = c_void_p.in_dll(cf, "kCFTypeDictionaryValueCallBacks")
        match = cf.CFDictionaryCreate(None, keys, vals, 2, ctypes.addressof(kcb), ctypes.addressof(vcb))
        self.client = io.IOHIDEventSystemClientCreate(None)
        io.IOHIDEventSystemClientSetMatching(self.client, match)
        self.product = cfstr("Product")
        self.buf = ctypes.create_string_buffer(256)
        self.peak = 0.0

    def read(self) -> dict[str, list[float]]:
        out = defaultdict(list)
        svcs = self.io.IOHIDEventSystemClientCopyServices(self.client)
        if not svcs:
            return out
        for i in range(self.cf.CFArrayGetCount(svcs)):
            s = self.cf.CFArrayGetValueAtIndex(svcs, i)
            name = self.io.IOHIDServiceClientCopyProperty(s, self.product)
            if not name:
                continue
            ok = self.cf.CFStringGetCString(name, self.buf, 256, self.utf8)
            self.cf.CFRelease(name)
            ev = self.io.IOHIDServiceClientCopyEvent(s, 15, 0, 0)   # kIOHIDEventTypeTemperature
            if ok and ev:
                v = self.io.IOHIDEventGetFloatValue(ev, 15 << 16)
                if 0 < v < 150:
                    out[self.buf.value.decode()].append(v)
            if ev:
                self.cf.CFRelease(ev)
        self.cf.CFRelease(svcs)
        return out

    def line(self) -> str:
        r = self.read()
        die = [v for k, vs in r.items() if k.startswith("PMU tdie") for v in vs]
        if not die:
            return "온도: 센서를 읽지 못함"
        hot = max(die)
        self.peak = max(self.peak, hot)
        color = "32" if hot < 80 else "33" if hot < 95 else "31"
        width = 20
        n = round(min(hot, 110) / 110 * width)
        gauge = f"\033[{color}m{'█' * n}\033[0m\033[90m{'─' * (width - n)}\033[0m"
        extra = []
        for key, label in (("gas gauge battery", "배터리"), ("NAND CH0 temp", "SSD")):
            if r.get(key):
                extra.append(f"{label} {max(r[key]):.0f}°C")
        therm = sh("pmset", "-g", "therm")
        throttle = "없음" if "No thermal warning" in therm and "No performance warning" in therm else "\033[31m있음\033[0m"
        state = "정상" if hot < 80 else "높음" if hot < 95 else "매우 높음"
        return (f"온도:   칩 [{gauge}] \033[{color}m{hot:.0f}°C\033[0m ({state}, 평균 {st.mean(die):.0f}, 최대 {self.peak:.0f})"
                f"   {'   '.join(extra)}   쓰로틀링 {throttle}")


class Watch:
    def __init__(self, out: Path, model: str, phases: list[str], log_dir: Path):
        self.out, self.model, self.phases, self.log_dir = out, model, phases, log_dir
        self.cells = {}        # hash -> dict(task, cond, n, path)
        self.plan = {}         # phase -> [hash] in run order (task-major, blind first, like run_phase.py)
        subs = {}
        for ph in phases:
            cfg, conds = load_conditions(ph, None)
            order = []
            for task in cfg["tasks"]:
                sub = subs.setdefault(task, subset_id_of(task))
                for c in [None] + conds:
                    h = cell_hash(model, task, c, sub, cfg.get("max_pixels"))
                    self.cells.setdefault(h, {"task": task, "cond": None if c is None else c.to_dict(),
                                              "n": data.SUBSET_SIZES[task], "path": cell_path(out, model, task, c)})
                    order.append(h)
            self.plan[ph] = order
        self.tails = {p: _Tail(p) for p in {c["path"] for c in self.cells.values()}}
        self.rows = defaultdict(dict)   # hash -> sample_id -> row
        self.stream = []                # (timestamp, seconds) of every row, for the overhead factor
        self.peak_mb = 0.0              # largest footprint seen by this monitor
        try:
            self.thermal = Thermal() if sys.platform == "darwin" else None
        except (OSError, AttributeError, ValueError):
            self.thermal = None

    def refresh(self) -> None:
        for tail in self.tails.values():
            for r in tail.new_rows():
                h = r.get("cell_hash")
                if h in self.cells and r.get("sample_id") not in self.rows[h]:
                    r["_t"] = dt.datetime.fromisoformat(r["timestamp"]).timestamp()
                    self.rows[h][r["sample_id"]] = r
                    self.stream.append((r["_t"], r["seconds"]))
        self.stream.sort()

    def overhead(self) -> float:
        s = self.stream[-RECENT:]
        wall = sum(b[0] - a[0] for a, b in zip(s, s[1:]) if b[0] - a[0] <= GAP)
        rowsec = sum(b[1] for a, b in zip(s, s[1:]) if b[0] - a[0] <= GAP)
        return max(1.0, wall / rowsec) if rowsec > 0 and len(s) > 20 else 1.2

    def sps(self, h: str) -> float:
        own = [r["seconds"] for r in self.rows[h].values()]
        if len(own) >= 5:
            return st.mean(own)
        task = self.cells[h]["task"]
        same = [r["seconds"] for k, rs in self.rows.items() if self.cells[k]["task"] == task for r in rs.values()]
        if same:
            return st.mean(same)
        alls = [s for _, s in self.stream]
        return st.mean(alls) if alls else 1.0

    def done(self, h: str) -> int:
        return min(len(self.rows[h]), self.cells[h]["n"])

    def render(self, interval: float) -> str:
        self.refresh()
        now, oh = time.time(), self.overhead()
        proc = running_process()
        live_phase = proc[1] if proc else None
        L = []
        L.append(f"twomemories · {self.model} 맥 본 실험      {clock(now, '%Y-%m-%d %H:%M:%S')} KST"
                 f"   ({interval:g}초마다 갱신, Ctrl+C로 모니터만 종료)")
        batt = sh("pmset", "-g", "batt").splitlines()
        power = ("AC 전원" if "AC Power" in batt[0] else "배터리 " + batt[1].split("\t")[-1].split(";")[0]) if batt else "?"
        if proc:
            L.append(f"프로세스: ● 실행 중  pid {proc[0]}  {live_phase}   {proc[2]}   {power}")
        else:
            L.append(f"프로세스: ○ 멈춤 (끝났거나 다음 phase 대기 중)   {power}")
        total, used, sw_used, sw_total = system_memory()
        if proc:
            fp = footprint_mb(proc[0]) / 1024
            self.peak_mb = max(self.peak_mb, fp * 1024)
            L.append(f"메모리: 실험 [{bar(fp / total, 20)}] {fp:4.1f} GB (최대 {self.peak_mb / 1024:.1f})"
                     f"   시스템 [{bar(used / total, 20)}] {used:4.1f} / {total:.0f} GB"
                     f"   스왑 {sw_used:.1f} / {sw_total:.1f} GB")
        else:
            L.append(f"메모리: 시스템 [{bar(used / total, 20)}] {used:4.1f} / {total:.0f} GB   스왑 {sw_used:.1f} / {sw_total:.1f} GB")
        if self.thermal:
            L.append(self.thermal.line())
        if total and used / total > 0.9:
            L.append("  ⚠ 시스템 메모리 90% 초과: 다른 앱을 닫으면 속도가 유지된다")
        L.append("")

        # overall, unique cells once
        uniq = list(dict.fromkeys(h for ph in self.phases for h in self.plan[ph]))
        tot = sum(self.cells[h]["n"] for h in uniq)
        dn = sum(self.done(h) for h in uniq)
        left = sum((self.cells[h]["n"] - self.done(h)) * self.sps(h) for h in uniq) * oh
        L.append(f"전체  [{bar(dn / tot, 40)}] {100 * dn / tot:5.1f}%   {dn:,} / {tot:,} 샘플")
        recent = [s for s in self.stream if s[0] >= now - 300]
        rate = f"최근 5분 {len(recent)}샘플" if recent else "최근 5분 결과 없음"
        L.append(f"      남은 계산 {dur(left)}  →  쉬지 않고 이어 돌면 {clock(now + left)} KST 완료"
                 f"   ({rate}, 오버헤드 x{oh:.2f})")
        L.append("")

        # phases
        L.append("단계")
        seen, t_acc, cur_phase = set(), now, None
        for i, ph in enumerate(self.phases, 1):
            hs = self.plan[ph]
            n_tot = sum(self.cells[h]["n"] for h in hs)
            n_dn = sum(self.done(h) for h in hs)
            c_dn = sum(self.done(h) >= self.cells[h]["n"] for h in hs)
            new = [h for h in hs if h not in seen]
            seen.update(hs)
            ph_left = sum((self.cells[h]["n"] - self.done(h)) * self.sps(h) for h in new) * oh
            t_acc += ph_left
            if c_dn == len(hs):
                mark, tail = "✔", "완료"
            elif ph == live_phase:
                mark, tail, cur_phase = "▶", f"진행 중 · 남은 {dur(ph_left)} · 예상 {clock(t_acc, '%H:%M')}", ph
            else:
                mark, tail = "·", f"대기 · 계산 {dur(ph_left)} · 이어 돌면 {clock(t_acc, '%H:%M')}"
            L.append(f"  {mark} {i}/{len(self.phases)} {ph:16s}{pad(PHASE_KO.get(ph, ''), 25)}"
                     f"[{bar(n_dn / n_tot, 20)}] {100 * n_dn / n_tot:5.1f}%  셀 {c_dn:2d}/{len(hs):<2d}  {tail}")
        L.append("")

        # current cell = the cell of the newest row
        latest = max((r for rs in self.rows.values() for r in rs.values()), key=lambda r: r["_t"], default=None)
        if latest:
            h = latest["cell_hash"]
            c = self.cells[h]
            d, n = self.done(h), c["n"]
            state = "작성 중" if d < n else "방금 끝남"
            L.append(f"지금 셀 ({state})  {TASK_KO[c['task']]} · {cond_ko(c['cond'])}")
            L.append(f"  [{bar(d / n, 40)}] {d} / {n} 샘플   이 셀 남은 {dur((n - d) * self.sps(h) * oh)}"
                     f"   샘플당 {self.sps(h):.2f}s   마지막 결과 {now - latest['_t']:.0f}초 전")
            gold = latest.get("meta", {}).get("gold", "")
            L.append(f"  최근 예측 \"{latest['pred'][:40]}\"  정답 \"{str(gold)[:30]}\"  "
                     f"점수 {latest['score']:.2f}  NLL/토큰 {latest['gold_nll'] / max(1, latest['gold_ntok']):.2f}")
            order = self.plan.get(cur_phase or "", [])
            nxt = [k for k in order[order.index(h) + 1:] if self.done(k) < self.cells[k]["n"]] if h in order else []
            if nxt:
                L.append("  다음: " + "  →  ".join(f"{TASK_KO[self.cells[k]['task']]} {cond_ko(self.cells[k]['cond'])}"
                                                   for k in nxt[:2]))
        else:
            L.append("지금 셀: 아직 결과 없음")
        L.append("")

        # cell map of the running (or first unfinished) phase
        ph = cur_phase or next((p for p in self.phases if any(self.done(h) < self.cells[h]["n"]
                                                               for h in self.plan[p])), None)
        if ph:
            L.append(f"셀 지도 {ph}  (✔ 끝  ▶ 작성 중  ◐ 일부  · 아직)")
            for task in TASK_KO:
                hs = [h for h in self.plan[ph] if self.cells[h]["task"] == task]
                if not hs:
                    continue
                marks = "".join("✔" if self.done(h) >= self.cells[h]["n"] else
                                "▶" if latest and h == latest["cell_hash"] else
                                "◐" if self.done(h) else "·" for h in hs)
                L.append(f"  {TASK_KO[task]:8s} {marks}")
            L.append("")

        # unaltered accuracy and NLL per task (sanity: should be high; blind low)
        acc = []
        for task in TASK_KO:
            full = [r for h, rs in self.rows.items() if self.cells[h]["task"] == task
                    and self.cells[h]["cond"] and self.cells[h]["cond"]["kv"] == "full"
                    and self.cells[h]["cond"]["state"] == "full" and self.cells[h]["cond"].get("token_keep", 1) == 1
                    for r in rs.values()]
            blind = [r for h, rs in self.rows.items() if self.cells[h]["task"] == task and self.cells[h]["cond"] is None
                     for r in rs.values()]
            if full:
                a = st.mean(r["score"] for r in full)
                b = f" / blind {st.mean(r['score'] for r in blind):.2f}" if blind else ""
                acc.append(f"{TASK_KO[task]} {a:.2f}{b}")
        L.append("정확도 원본" + (" / blind:  " + "   ".join(acc) if acc else ": 아직 없음"))

        logs = sorted(self.log_dir.glob("mac_phase*.log"), key=lambda p: p.stat().st_mtime) if self.log_dir.exists() else []
        if logs:
            with logs[-1].open("rb") as f:
                f.seek(max(0, logs[-1].stat().st_size - 8192))
                lines = [x for x in f.read().decode(errors="replace").splitlines()
                         if x.strip() and not re.search(r"warn|falling back|NOTE|Loading weights", x, re.I)]
            L += ["", f"--- 로그 {logs[-1].name} (마지막 3줄)", *lines[-3:]]
        return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(REPO / "results" / "mac"))
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--phases", nargs="*", default=PHASES)
    ap.add_argument("--logs", default=str(REPO / "logs"))
    ap.add_argument("--interval", type=float, default=3.0)
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args(argv)
    w = Watch(Path(a.out), a.model, a.phases, Path(a.logs))
    try:
        while True:
            text = w.render(a.interval)
            if a.once:
                print(text)
                return 0
            sys.stdout.write("\033[H\033[2J" + text + "\n")
            sys.stdout.flush()
            time.sleep(a.interval)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
