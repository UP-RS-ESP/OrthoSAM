import sys
import subprocess
print(sys.argv[0])

noti='/DATA/vito/code/notification.py'
subprocess.run(["python", noti, sys.argv[0]])