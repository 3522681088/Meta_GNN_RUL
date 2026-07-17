import subprocess, sys

for domain in ["FD001","FD002","FD003","FD004"]:
    subprocess.run([sys.executable,"main.py","--suite","baselines","--target",domain],check=True)

