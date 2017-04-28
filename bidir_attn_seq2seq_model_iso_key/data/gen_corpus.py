#!/usr/bin/env python3

import sys
import re

with open(sys.argv[1]) as org:
  with open(sys.argv[1] + '.from', 'w') as fr:
    with open(sys.argv[1] + '.to', 'w') as to:
      for line in org:
        (title, poem) = line.strip().split(':')
        poem = re.sub(r'[。，]', ':', poem)
        poem = re.sub(r':$', '', poem)
        s = [title] + poem.split(':')
        for i in range(1, len(s)):
          print(''.join(s[:i]), file=fr)
          print(s[i], file=to)
