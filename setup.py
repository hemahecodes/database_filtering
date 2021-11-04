from distutils.core import setup
setup(
  name = 'database_filtering',
  packages = ['database_filtering', 'database_filtering/utils'],
  version = '1.1.0',
  license='MIT',
  description = 'Filter database with r_group specification.',
  author = 'Helena Martín',
  author_email = 'helena.martin@nostrumbiodiscovery.com',
  url = 'https://github.com/hemahecodes/database_filtering',
  download_url = 'https://github.com/hemahecodes/database_filtering/archive/refs/tags/0.1.zip',
  keywords = ['filtering', 'database', 'rdkit'],   #
  install_requires=[
          'validators',
          'beautifulsoup4',
          'argparse',
          'pandas',
          'networkx',
      ],
  classifiers=[
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',   
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
