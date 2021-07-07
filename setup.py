from setuptools import setup, Extension, find_packages, distutils
import sys
from os import environ, path, makedirs
from setuptools.command.build_ext import build_ext
from glob import glob
import multiprocessing
import multiprocessing.pool
from subprocess import check_output, check_call
SLEEFLIB="libsleef.a"
sleefdir = environ.get("SLEEF_DIR", "sleef/build")

def main():

    if not path.isfile(SLEEFLIB):
        print("Making sleef")
        check_call(f"make {SLEEFLIB}", shell=True)
    if not path.isfile("libkl.a"):
        check_call(f"make libkl.a", shell=True)
    if not path.isfile("libsimdsampling/libsimdsampling.a"):
        print("Making libss.a")
        check_call(f"make libsimdsampling/libsimdsampling.a", shell=True)
    if not path.isfile("libgomp.a"):
        check_call(f"ln -s `{environ.get('CXX', 'g++')} --print-file-name=libgomp.a`", shell=True, executable="/bin/bash")

    # from https://stackoverflow.com/questions/11013851/speeding-up-build-process-with-distutils
    # parallelizes extension compilation
    def parallelCCompile(self, sources, output_dir=None, macros=None,
            include_dirs=None, debug=0, extra_preargs=None, extra_postargs=None,
            depends=None):
        # those lines are copied from distutils.ccompiler.CCompiler directly
        macros, objects, extra_postargs, pp_opts, build = \
                self._setup_compile(output_dir, macros, include_dirs, sources,
                        depends, extra_postargs)
        cc_args = self._get_cc_args(pp_opts, debug, extra_preargs)
        # parallel code
        N_cores = int(environ.get("OMP_NUM_THREADS", 1))
        def _single_compile(obj):
            try: src, ext = build[obj]
            except KeyError: return
            self._compile(obj, src, ext, cc_args, extra_postargs, pp_opts)
        # convert to list, imap is evaluated on-demand
        list(multiprocessing.pool.ThreadPool(N_cores).imap(_single_compile,objects))
        return objects


    import distutils.ccompiler
    distutils.ccompiler.CCompiler.compile=parallelCCompile

    LIBOBJS = [SLEEFLIB, "libkl.a", "libsimdsampling/libsimdsampling.a", "libgomp.a"]


    class get_pybind_include(object):
        """Helper class to determine the pybind11 include path
        The purpose of this class is to postpone importing pybind11
        until it is actually installed, so that the ``get_include()``
        method can be invoked. """

        def __init__(self, user=False):
            self.user = user

        def __str__(self):
            import pybind11
            return pybind11.get_include(self.user)


    #EXTRAS = environ.get("EXTRA", "")

    extra_compile_args = ['-march=native', '-DNDEBUG',
                          '-Wno-char-subscripts', '-Wno-unused-function', '-Wno-ignored-qualifiers',
                          '-Wno-strict-aliasing', '-Wno-ignored-attributes', '-fno-wrapv',
                          '-Wall', '-Wextra', '-Wformat',
                          '-lz', '-fopenmp', "-lgomp", "-DEXTERNAL_BOOST_IOSTREAMS=1",
                          "-DBLAZE_USE_SLEEF=1", "-pipe",
                          '-Wno-deprecated-declarations', '-O3']

    if 'BOOST_DIR' in environ:
        extra_compile_args.append("-I%s" % environ['BOOST_DIR'])


    include_dirs=[
        # Path to pybind11 headers
        get_pybind_include(),
        get_pybind_include(user=True),
       "./",
       "./include",
       sleefdir + "/include",
       "./include/minicore",
       "./blaze",
       "./pybind11/include",
       "./python"
    ]

    ext_modules = [
        Extension(
            'pyminicore',
             glob('python/*.cpp') + [
             "include/minicore/util/boost/zlib.cpp", "include/minicore/util/boost/gzip.cpp"
             ],
            include_dirs=include_dirs,
            language='c++',
            extra_compile_args=extra_compile_args + ["-DEXTERNAL_BOOST_IOSTREAMS=1"],
            extra_objects=LIBOBJS
        )
    ]


    # As of Python 3.6, CCompiler has a `has_flag` method.
    # cf http://bugs.python.org/issue26689
    def has_flag(compiler, flagname):
        """Return a boolean indicating whether a flag name is supported on
        the specified compiler.
        """
        import tempfile
        with tempfile.NamedTemporaryFile('w', suffix='.cpp') as f:
            f.write('int main (int argc, char **argv) { return 0; }')
            try:
                compiler.compile([f.name], extra_postargs=[flagname])
            except distutils.errors.CompileError:
                return False
        return True


    def cpp_flag(compiler):
        """Return the -std=c++[11/14/17] compiler flag.
        The newer version is prefered over c++11 (when it is available).
        """
        flags = ['-std=c++17', '-std=c++14', '-std=c++11']

        for flag in flags:
            if has_flag(compiler, flag): return flag

        raise RuntimeError('Unsupported compiler -- at least C++11 support '
                           'is needed!')

    extra_link_opts = ["-fopenmp", "-lgomp", "-lz", "-DEXTERNAL_BOOST_IOSTREAMS=1"] + LIBOBJS


    class BuildExt(build_ext):
        """A custom build extension for adding compiler-specific options."""
        c_opts = {
            'msvc': ['/EHsc'],
            'unix': [],
        }
        l_opts = {
            'msvc': [],
            'unix': [],
        }

        from sys import platform
        if platform == 'darwin':
            darwin_opts = ['-mmacosx-version-min=10.7']# , '-libstd=libc++']
            # darwin_opts = []
            c_opts['unix'] += darwin_opts
            l_opts['unix'] += darwin_opts

        def build_extensions(self):
            ct = self.compiler.compiler_type
            opts = self.c_opts.get(ct, [])
            link_opts = self.l_opts.get(ct, [])
            if ct == 'unix':
                opts.append('-DVERSION_INFO="%s"' % self.distribution.get_version())
                opts.append(cpp_flag(self.compiler))
                if has_flag(self.compiler, '-fvisibility=hidden'):
                    opts.append('-fvisibility=hidden')
            elif ct == 'msvc':
                opts.append('/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version())
            for ext in self.extensions:
                ext.extra_compile_args = opts
                ext.extra_compile_args += extra_compile_args
                ext.extra_link_args = link_opts + extra_link_opts
            build_ext.build_extensions(self)


    __version__ = check_output("git describe --abbrev=4", shell=True).decode().split("-")[0]
    setup(
        name='minicore',
        version=__version__,
        author='Daniel Baker',
        author_email='dnb@cs.jhu.edu',
        url='https://github.com/dnbaker/minicore',
        description='A python module for clustering and coresets',
        long_description='',
        ext_modules=ext_modules,
        install_requires=['pybind11', 'numpy>=0.19', 'scipy'],
        setup_requires=['pybind11'],
        cmdclass={'build_ext': BuildExt},
        zip_safe=False,
        packages=find_packages()
    )

if __name__ == "__main__":
    sys.exit(main())
