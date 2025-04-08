'''
This file performs the CMake building on Windows,
Cleans directories needed by the ray tracing program
And runs the program once cleaning and building is complete
'''
import subprocess
import os

def create_dir(dir_name):
    os.mkdir(dir_name)

def clean_dir(dir_name):
    files = os.listdir(dir_name)

    for file in files:
        os.remove(dir_name + "/" + file)

if __name__ == "__main__":
    subprocess.run(["cmake", "-B", "build"])
    subprocess.run(["cmake", "--build", "build"])

    if os.path.exists("images"):
        clean_dir("images")
    else:
        create_dir("images")

    if os.path.exists("output"):
        clean_dir("output")
    else:
        create_dir("output")

    subprocess.run(["./build/Debug/my_cuda_app"])