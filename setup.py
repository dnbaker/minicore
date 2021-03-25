from subprocess import check_call
import sys, os

'''Monkey-patch mapping the setup.py in the base directory to the python/setup.py.
'''
wd = os.getcwd()
print("Working directory: " + wd, file=sys.stderr)
check_call([sys.executable] + sys.argv)
