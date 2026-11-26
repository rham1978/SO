"""
Aceleración del simulador sin cambiar su lógica.

El cuello de botella del simulador original es `_book_from_pool`: cuando un
paciente no encuentra cupo de ecografía/laboratorio, reintenta cada 10 minutos
hasta que el pool se rellena (una vez por semana, a las 8:00 del lunes). Eso
genera ~10 millones de eventos por año simulado.

Como los pools SOLO se rellenan en `week_start + work_start_h*60`, un reintento
fallido no puede tener éxito antes del siguiente relleno. Este parche salta
directamente al primer instante de la grilla de reintentos (t_fallo + 10k) que
es >= al siguiente relleno, que es exactamente el instante en que el
simulador original habría tenido éxito. El resultado esperado es idéntico; se
verifica con `verificar_sim_fast.py`.
"""
import bisect
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import simulador_clinica_baseline as sim  # noqa: E402

WEEK_MIN = 7 * 24 * 60
RETRY_MIN = 10


def _next_refill_time(now: float, cfg) -> float:
    """Primer instante de relleno de pools igual o posterior a `now`.

    Incluye `now` si cae justo en un relleno: el evento de relleno de ese mismo instante
    puede no haberse procesado todavía, y el original lo encontraría en su reintento
    de +10 min. En ese caso el salto es de un solo reintento, igual que el original."""
    offset = cfg.work_start_h * 60
    k = math.ceil((now - offset) / WEEK_MIN)
    return k * WEEK_MIN + offset


def _book_from_pool_fast(self, pool, lock_res, not_before, waitslot_key, pid=None):
    t_request = self.env.now
    retries = 0
    max_retries = max(1, int(self.cfg.sim_time_min // RETRY_MIN))
    _is_list = isinstance(pool, list)
    while True:
        if self.env.now >= self.cfg.sim_time_min or retries > max_retries:
            sim.log.warning("Pool agotado pid=%s key=%s t=%.0f", pid, waitslot_key, self.env.now)
            return None
        t0 = max(not_before, self.env.now)
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
        # Salto: primer punto de la grilla de reintentos >= próximo relleno
        t_fail = self.env.now
        t_target = min(_next_refill_time(t_fail, self.cfg), self.cfg.sim_time_min)
        k = max(1, math.ceil((t_target - t_fail) / RETRY_MIN - 1e-12))
        retries += k
        yield self.env.timeout(RETRY_MIN * k)


def apply_patch():
    sim.ClinicModelAdjusted._book_from_pool = _book_from_pool_fast


def remove_patch(original):
    sim.ClinicModelAdjusted._book_from_pool = original
