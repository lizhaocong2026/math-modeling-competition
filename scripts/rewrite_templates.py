import os
BASE = r'D:\\本地的知识库构建\\math-modeling-competition\\paper\\texfile'

def w(name, content):
    path = os.path.join(BASE, name)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)
    sz = os.path.getsize(path)
    print(f'{name}: {sz}B')

def nl(): return chr(10)
def bs(): return chr(92)
