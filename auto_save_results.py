"""
Auto-save: sube JSONs nuevos a GitHub cada 5 min.
Requiere env var GITHUB_PAT con token de acceso.
"""
import os, time, base64, json, urllib.request, urllib.error

PAT    = os.environ.get("GITHUB_PAT", "")
OWNER  = "rham1978"
REPO   = "SO"
BRANCH = "claude/wonderful-hopper-Nsrsu"
DIR    = os.path.dirname(os.path.abspath(__file__))

WATCH_FILES = [
    f"resultado_comparativa_m{n}.json" for n in [11,12,13,14]
] + [
    f"resultado_comparativa_m{n}.png" for n in [11,12,13,14]
] + [
    "resultados_comparativa.json",
    "comparativa_convergencia.png",
    "comparativa_configuraciones.png",
    "comparativa_tabla_resumen.png",
]
uploaded = set()

def upload(fname):
    local = os.path.join(DIR, fname)
    if not os.path.exists(local):
        return False
    with open(local, "rb") as f:
        content_b64 = base64.b64encode(f.read()).decode()
    url_get = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{fname}?ref={BRANCH}"
    sha = None
    try:
        req = urllib.request.Request(url_get,
            headers={"Authorization": f"token {PAT}", "Accept": "application/vnd.github.v3+json"})
        with urllib.request.urlopen(req) as r:
            sha = json.loads(r.read())["sha"]
    except urllib.error.HTTPError as e:
        if e.code != 404:
            return False
    payload = {"message": f"Auto-save: {fname}", "content": content_b64, "branch": BRANCH}
    if sha:
        payload["sha"] = sha
    url_put = f"https://api.github.com/repos/{OWNER}/{REPO}/contents/{fname}"
    req2 = urllib.request.Request(url_put, data=json.dumps(payload).encode(), method="PUT",
        headers={"Authorization": f"token {PAT}", "Content-Type": "application/json",
                 "Accept": "application/vnd.github.v3+json"})
    try:
        with urllib.request.urlopen(req2) as r:
            json.loads(r.read())
            return True
    except urllib.error.HTTPError:
        return False

print(f"[{time.strftime(\'%H:%M:%S\')}] Auto-save iniciado.", flush=True)
while True:
    for fname in WATCH_FILES:
        local = os.path.join(DIR, fname)
        if os.path.exists(local) and fname not in uploaded:
            if upload(fname):
                uploaded.add(fname)
                print(f"[{time.strftime(\'%H:%M:%S\')}] ✓ {fname}", flush=True)
    done = all(f"resultado_comparativa_m{n}.json" in uploaded for n in [11,12,13,14])
    if done:
        print("Todos los módulos subidos. Fin.", flush=True)
        break
    time.sleep(300)
