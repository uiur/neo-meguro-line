# just simple example test
# generate result from images in test/ and human will check it.

import os
import glob

os.system('rm -rf result && mkdir -p result')
for path in glob.glob('./data/*'):
    os.system('python convert.py %s > %s' % (path, path.replace('/data/', '/result/')))
