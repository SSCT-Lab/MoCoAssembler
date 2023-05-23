import subprocess
from queue import Queue

que = Queue()

if __name__ == "__main__":
    process = subprocess.Popen("python demo1.py", stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, error = process.communicate()

    if error:
        print("错误信息:\n", error.decode("utf-8"))
    else:
        que.put("demo1")
