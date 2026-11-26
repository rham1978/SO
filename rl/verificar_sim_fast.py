"""Compara KPIs del simulador original vs parcheado (mismas semillas)."""
import logging, sys, time, math
import sim_fast
from sim_fast import sim
logging.disable(logging.CRITICAL)
weeks = int(sys.argv[1]) if len(sys.argv) > 1 else 10
seeds = [int(s) for s in sys.argv[2:]] or [0, 1]
orig = sim.ClinicModelAdjusted._book_from_pool
def scalars(r):
    return {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
ok = True
for sd in seeds:
    cfg = sim.SimConfig(benchmark_mode=True, weeks_to_simulate=weeks)
    sim_fast.remove_patch(orig); t = time.time(); a = scalars(sim.run_once(sd, cfg)); ta = time.time() - t
    sim_fast.apply_patch();      t = time.time(); b = scalars(sim.run_once(sd, cfg)); tb = time.time() - t
    diff = [k for k in a if not (a[k] == b.get(k) or (isinstance(a[k], float) and math.isnan(a[k]) and math.isnan(b.get(k, 0))))]
    print(f"seed {sd}: original {ta:.1f}s, rapido {tb:.1f}s, x{ta/tb:.1f}, KPIs distintos: {len(diff)}/{len(a)}", diff[:8])
    ok &= not diff
print("IDENTICO" if ok else "DIFERENCIAS")
sys.exit(0 if ok else 1)
