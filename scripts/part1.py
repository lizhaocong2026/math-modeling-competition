# -*- coding: utf-8 -*-
import os
BASE = os.path.join(os.getcwd(), 'paper', 'texfile')
os.makedirs(BASE, exist_ok=True)

def w(name, content):
    path = os.path.join(BASE, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    print(f'Wrote {name}: {len(content)} bytes')

print('Ready to write templates')
