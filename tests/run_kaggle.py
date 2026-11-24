import asyncio, json, uuid, ssl, sys, urllib.request, websockets

PROXY_URL = 'https://kkb-production.jupyter-proxy.kaggle.net/k/373264596/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2IiwidHlwIjoiSldUIn0..QHEFci2NTigKJL_ugaju1w.3FpTtsG2wHAHUukofkwu_R8m6cfm1Z6YNUJQ1nMO3MoQBlv4hVIOV5avcHI3xjvPrhbUsnMjz5uNOMdV82cn6iQXiZS2steY_b64e3tq97gd15VmbqrGIqtHR_rPi-XBHH2QPPnnMFWIbOD6ApkluYxCsDVMvuO6ACV_T12lHMcKBArw4nNgJqX3I00_pqv58Z2-Nu3PPa0VsYZZeQ9WChYHNJp7W0S0dKwOhOl9090LpAVUiwds15pOhhDBSffx.9eu1dG2E1kVzgfSWX-Uxrw/proxy'
CTX = ssl.create_default_context()

def get_or_create_idle_kernel():
    req = urllib.request.Request(PROXY_URL.rstrip('/') + '/api/kernels')
    with urllib.request.urlopen(req, context=CTX, timeout=10) as r:
        kernels = json.loads(r.read().decode('utf-8'))
    for k in kernels:
        if k['execution_state'] == 'idle':
            return k['id']
    # Create new
    req = urllib.request.Request(
        PROXY_URL.rstrip('/') + '/api/kernels',
        data=json.dumps({'name': 'python3'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'},
        method='POST'
    )
    with urllib.request.urlopen(req, context=CTX, timeout=10) as r:
        k = json.loads(r.read().decode('utf-8'))
        return k['id']

async def execute_code(code_str, kernel_id=None, timeout_sec=900):
    if not kernel_id:
        kernel_id = get_or_create_idle_kernel()
    
    ws_url = PROXY_URL.replace('https://', 'wss://').replace('http://', 'ws://').rstrip('/') + f'/api/kernels/{kernel_id}/channels'
    
    async with websockets.connect(ws_url, ssl=CTX, max_size=20*1024*1024, ping_interval=None, ping_timeout=None) as ws:
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
        while True:
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
                    break
                elif mtype == 'execute_reply':
                    status = msg['content']['status']
                    break
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
