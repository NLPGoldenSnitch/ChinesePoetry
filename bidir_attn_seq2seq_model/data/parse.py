#!/usr/bin/env python3

with open('q7_with_key.txt') as f:
  with open('q7.key', 'w') as ky:
    with open('q7.from', 'w') as fr:
      with open('q7.to', 'w') as to:
        for line in f:
          k, p = line.strip().split(':')
          k = k.split(',')
          p = p.split(',')
          while len(k) >= 4:
            ck = k[:4]
            cp = p[:4]
            k = k[4:]
            p = p[4:]

            for i in range(len(cp)):
              print(''.join(['{}'.format(k) for k in ck[i]]), file=ky)
              print(''.join(cp[:i]), file=fr)
              print(cp[i], file=to)
