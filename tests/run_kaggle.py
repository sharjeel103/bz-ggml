import asyncio, json, os, uuid, ssl, sys, urllib.request, websockets

PROXY_URL = os.environ.get("KAGGLE_PROXY_URL", "").strip()
CTX = ssl.create_default_context()


def _require_proxy_url() -> str:
    if not PROXY_URL:
        raise RuntimeError(
            "KAGGLE_PROXY_URL is not set. Export KAGGLE_PROXY_URL='https://.../proxy' before running remote Kaggle scripts."
        )
    return PROXY_URL.rstrip("/")


def list_kernels():
    req = urllib.request.Request(_require_proxy_url() + "/api/kernels")
    with urllib.request.urlopen(req, context=CTX, timeout=10) as r:
        return json.loads(r.read().decode("utf-8"))


def get_or_create_idle_kernel():
    base = _require_proxy_url()
    kernels = list_kernels()
    for k in kernels:
        if k.get("execution_state") == "idle":
            return k["id"]
    req = urllib.request.Request(
        base + "/api/kernels",
        data=json.dumps({"name": "python3"}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=CTX, timeout=10) as r:
        k = json.loads(r.read().decode("utf-8"))
        return k["id"]


def purge_extra_kernels(keep_kernel_id=None):
    """Delete all other Kaggle Jupyter kernels so none hold stale GPU VRAM."""
    base = _require_proxy_url()
    killed = []
    for k in list_kernels():
        kid = k.get("id")
        if not kid or kid == keep_kernel_id:
            continue
        req = urllib.request.Request(base + f"/api/kernels/{kid}", method="DELETE")
        try:
            with urllib.request.urlopen(req, context=CTX, timeout=10) as r:
                if r.status in (200, 204):
                    killed.append(kid)
        except Exception as e:
            print(f"[purge_extra_kernels] Warning deleting kernel {kid}: {e}", flush=True)
    return killed


def restart_kernel(kernel_id=None):
    base = _require_proxy_url()
    if not kernel_id:
        kernel_id = get_or_create_idle_kernel()
    req = urllib.request.Request(
        base + f"/api/kernels/{kernel_id}/restart",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, context=CTX, timeout=15) as r:
        return r.status == 200


async def execute_code(code_str, kernel_id=None, timeout_sec=900):
    base = _require_proxy_url()
    if not kernel_id:
        kernel_id = get_or_create_idle_kernel()

    ws_url = base.replace("https://", "wss://").replace("http://", "ws://") + f"/api/kernels/{kernel_id}/channels"
    
    async with websockets.connect(ws_url, ssl=CTX, max_size=50*1024*1024, ping_interval=30, ping_timeout=180) as ws:
        msg_id = str(uuid.uuid4())
        await ws.send(json.dumps({
            'header': {'msg_id': msg_id, 'username': 'admin', 'session': str(uuid.uuid4()), 'msg_type': 'execute_request', 'version': '5.3'},
            'parent_header': {}, 'metadata': {},
            'content': {'code': code_str, 'silent': False, 'store_history': True, 'user_expressions': {}, 'allow_stdin': False},
            'channel': 'shell'
        }))
        
        stdout_buf = []
        stderr_buf = []
        status = 'ok'
        got_reply = False
        got_idle = False
        while not (got_reply and got_idle):
            raw = await asyncio.wait_for(ws.recv(), timeout=timeout_sec)
            msg = json.loads(raw)
            if msg.get('parent_header', {}).get('msg_id') == msg_id:
                mtype = msg.get('header', {}).get('msg_type')
                if mtype == 'stream':
                    text = msg['content']['text']
                    if msg['content']['name'] == 'stdout':
                        sys.stdout.write(text)
                        sys.stdout.flush()
                        stdout_buf.append(text)
                    else:
                        sys.stderr.write(text)
                        sys.stderr.flush()
                        stderr_buf.append(text)
                elif mtype == 'error':
                    err_txt = '\n'.join(msg['content']['traceback'])
                    sys.stderr.write(err_txt + '\n')
                    sys.stderr.flush()
                    status = 'error'
                    stderr_buf.append(err_txt)
                    got_reply = True
                    got_idle = True
                elif mtype == 'execute_reply':
                    status = msg['content']['status']
                    got_reply = True
                elif mtype == 'status' and msg['content'].get('execution_state') == 'idle':
                    got_idle = True
        return ''.join(stdout_buf), ''.join(stderr_buf), status

if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--file':
        with open(sys.argv[2]) as f:
            code = f.read()
    elif len(sys.argv) > 1:
        code = sys.argv[1]
    else:
        code = 'import os; print(os.uname())'
    asyncio.run(execute_code(code))
