from config import *
import subprocess
import json
import os
import signal
import time

def open_ssh_tunnel():
    try:
        print("Opening remote SSH tunnel for remote redis...")
        with open('secrets.secret','r') as f:
            secrets = json.load(f)

        # Build the SSH command to open the tunnel
        ssh_command = f"ssh -N -L {secrets['redis']['local_port']}:localhost:{secrets['redis']['tunnel_port']} {secrets['redis']['tunnel_host']}"
        print(f"running command ({ssh_command})")
        # Run the SSH command in a subprocess
        pro = subprocess.Popen(ssh_command, stdout=subprocess.PIPE, 
                       shell=True, preexec_fn=os.setsid) 
        while(True):
            time.sleep(10)

    except:
        print("Killing ssh tunnel...")
        os.killpg(os.getpgid(pro.pid), signal.SIGTERM)
    print(f"Exiting SSH shell functions")

if __name__ == "__main__":
    open_ssh_tunnel()

