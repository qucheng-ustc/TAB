from distutils.core import setup, Extension

mymetismodule = Extension('mymetis',
                    define_macros = [('MAJOR_VERSION', '1'),
                                     ('MINOR_VERSION', '0')],
                    include_dirs = ['/usr/local/include', '/mnt/ssd/qucheng/anaconda3/envs/arrl/include/python3.10'],
                    libraries = ['metis'],
                    library_dirs = ['/usr/local/lib'],
                    sources = ['mymetis.cpp'])

long_description = '''
This is a package for partition large graph with metis.
'''

setup(name = 'MyMetis',
      version = '1.0',
      description = 'Partition graph using metis',
      author = 'Cheng Qu',
      author_email = 'qucheng@mail.ustc.edu.cn',
      url = 'https://docs.python.org/extending/building',
      long_description = long_description,
      ext_modules = [mymetismodule])