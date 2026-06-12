"""
Compara TTS con retry=10min (original) vs retry=4h (nuevo) usando los mismos 10 seeds.
Usa benchmark_mode=True para acelerar (idéntico a False en cuanto a TTS).
Métrica: tts_full_days_mean (la misma que usa el harness de optimización).
"""
import sys, bisect, numpy as np
sys.path.insert(0, '/home/user/SO')
import simulador_clinica_baseline as sim_mod

def make_patched_book(retry_minutes):
    def _book_from_pool(self, pool, lock_res, not_before, waitslot_key, pid=None):
        import bisect, logging
        t_request = self.env.now
        retries   = 0
        max_retries = max(1, int(self.cfg.sim_time_min // retry_minutes))
        _is_list  = isinstance(pool, list)
        while True:
            if self.env.now >= self.cfg.sim_time_min or retries > max_retries:
                return None
            t0  = max(not_before, self.env.now)
            t_q = self.env.now
            with lock_res.request() as req:
                yield req
                self._trace_add_waitq(pid, 'pool_lock', self.env.now - t_q)
                if _is_list:
                    while pool and pool[0] <= self.env.now:
                        pool.pop(0)
                    idx = bisect.bisect_left(pool, t0)
                else:
                    while pool and pool[0] <= self.env.now:
                        pool.popleft()
                    idx = None
                    for i, s in enumerate(pool):
                        if s >= t0:
                            idx = i
                            break
                    if idx is None:
                        idx = len(pool)
                if idx < len(pool):
                    slot_t = pool[idx]
                    del pool[idx]
                    self._trace_add_waitslot(pid, waitslot_key, slot_t - t_request)
                    return slot_t
            retries += 1
            yield self.env.timeout(retry_minutes * 60)
    return _book_from_pool


def run_seed(seed, retry_minutes):
    original = sim_mod.ClinicModelAdjusted._book_from_pool
    sim_mod.ClinicModelAdjusted._book_from_pool = make_patched_book(retry_minutes)
    try:
        cfg = sim_mod.SimConfig()
        cfg.random_seed_base = seed
        cfg.benchmark_mode = True
        res = sim_mod.run_once(seed_offset=0, cfg=cfg)
        return res['tts_full_days_mean']
    finally:
        sim_mod.ClinicModelAdjusted._book_from_pool = original


if __name__ == '__main__':
    import time
    n = 10
    tts_10 = []
    tts_4h = []

    print("=== retry=10min ===", flush=True)
    t0 = time.time()
    for s in range(n):
        v = run_seed(s, 10)
        tts_10.append(v)
        print(f"  seed {s}: {v:.2f}d  ({time.time()-t0:.0f}s)", flush=True)

    print("\n=== retry=4h (sin parcheo) ===", flush=True)
    t1 = time.time()
    for s in range(n):
        # Usa el código actual sin parcheo (retry=4h ya hardcodeado)
        cfg = sim_mod.SimConfig()
        cfg.random_seed_base = s
        cfg.benchmark_mode = True
        res = sim_mod.run_once(seed_offset=0, cfg=cfg)
        v = res['tts_full_days_mean']
        tts_4h.append(v)
        print(f"  seed {s}: {v:.2f}d  ({time.time()-t1:.0f}s)", flush=True)

    a = np.array(tts_10)
    b = np.array(tts_4h)

    print(f"\n{'':=<55}")
    print(f"  retry=10min : media={a.mean():.2f}d  sd={a.std(ddof=1):.2f}d  [min={a.min():.1f} max={a.max():.1f}]")
    print(f"  retry=4h    : media={b.mean():.2f}d  sd={b.std(ddof=1):.2f}d  [min={b.min():.1f} max={b.max():.1f}]")
    print(f"  Δmedia      : {b.mean()-a.mean():+.2f}d  ({abs(b.mean()-a.mean())/a.std(ddof=1)*100:.1f}% de 1 SD)")
    try:
        from scipy import stats as sc
        stat, pval = sc.wilcoxon(a, b)
        print(f"  Wilcoxon    : p={pval:.4f}  {'NO sig.' if pval>0.05 else '** SIGNIFICATIVO **'}")
    except Exception as e:
        print(f"  (scipy no disponible: {e})")

    print(f"\n{'seed':>6}  {'10min':>8}  {'4h':>8}  {'diff':>8}")
    for s in range(n):
        print(f"  {s:3d}    {tts_10[s]:8.2f}  {tts_4h[s]:8.2f}  {tts_4h[s]-tts_10[s]:+8.2f}")
