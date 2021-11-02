from distutils.core import setup
setup(
  name = 'filtering',
  packages = ['filtering'],
  version = '0.1',
  license='MIT',
  description = 'Filter database with r_group specification.',
  author = 'Helena Martín',
  author_email = 'helena.martin@nostrumbiodiscovery.com',
  url = 'https://github.com/hemahecodes/filtering',
  download_url = 'https://github.com/hemahecodes/filtering/archive/refs/tags/0.1.zip',
  keywords = ['filtering', 'database', 'rdkit'],   #
  install_requires=[
          'validators',
          'beautifulsoup4',
          'rdkit'
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
