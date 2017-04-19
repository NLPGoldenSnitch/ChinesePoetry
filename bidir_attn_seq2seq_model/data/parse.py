#!/usr/bin/env python3

with open('q7.key') as f:
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
            print(''.join(cp[:i]) + ''.join(['{}'.format(k) for k in ck[i]]), file=fr)
            print(cp[i], file=to)
