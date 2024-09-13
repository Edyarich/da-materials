import subprocess
import os
import requests
import json
from settings import *


response = requests.get(NGROK_DOWNLOADING_URL)
with open(TGZ_FILENAME, "wb") as fd:
    fd.write(response.content)
    
os.system(f"tar xzf {TGZ_FILENAME}")
os.remove(TGZ_FILENAME)

subprocess.call(['./ngrok', 'config', 'add-authtoken', NGROK_AUTH_TOKEN])
first_proc = subprocess.Popen(args=["./ngrok", "http", str(PORT)], 
                              stdout=subprocess.DEVNULL)
try:
    first_proc.wait(timeout=1)
except subprocess.TimeoutExpired:
    pass

os.mknod(OUT_ADDR_FILENAME)
os.system(f'curl http://localhost:4040/api/tunnels > {OUT_ADDR_FILENAME}')

with open(OUT_ADDR_FILENAME, "r") as fd:
    data = json.load(fd)
    ngrok_addr = data["tunnels"][0]["public_url"]
    
os.remove(OUT_ADDR_FILENAME)

second_proc = subprocess.Popen(args=['python3', 'server.py', ngrok_addr])
