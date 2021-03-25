from subprocess import check_call
import sys, os

'''Monkey-patch mapping the setup.py in the base directory to the python/setup.py.
'''
os.chdir("python")
check_call([sys.executable] + sys.argv)
