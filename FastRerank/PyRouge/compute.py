
from Rouge import Rouge
rouge = Rouge.Rouge()

ref_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/test/ref-word.txt"

system_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/models/0815-171759  512 linear 18.41/dev.out.107"

inputs = []
systems = []
refs = []

f0 = open('input-word.txt', encoding='utf-8')
f1 = open(system_file, encoding='utf-8')
f2 = open(ref_file, encoding='utf-8')

for l0, l1, l2 in zip(f0, f1, f2):
    if not l0:
        break
    l0 = l0.strip()
    if l0 != 'unknown_word':
        systems.append(l1.strip())
        refs.append([l2.strip()])

print(len(systems))
scores = rouge.compute_rouge(refs, systems)
print(scores)


ref_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref0.txt"
system_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/models/0821-213049/dev.out.1"

inputs = []
systems = []
refs = []

f0 = open('input-word.txt', encoding='utf-8')
f1 = open(system_file, encoding='utf-8')
f2 = open(ref_file, encoding='utf-8')

for l0, l1, l2 in zip(f0, f1, f2):
    if not l0:
        break
    l0 = l0.strip()
    # if l0 != 'unknown_word':
    systems.append(l1.strip()[:75])
    refs.append([l2.strip()])

print(len(systems))
scores = rouge.compute_rouge(refs, systems)
print(scores)

ref_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref1.txt"
system_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/models/0821-213049/dev.out.1"

inputs = []
systems = []
refs = []

f0 = open('input-word.txt', encoding='utf-8')
f1 = open(system_file, encoding='utf-8')
f2 = open(ref_file, encoding='utf-8')

for l0, l1, l2 in zip(f0, f1, f2):
    if not l0:
        break
    l0 = l0.strip()
    # if l0 != 'unknown_word':
    systems.append(l1.strip()[:75])
    refs.append([l2.strip()])

print(len(systems))
scores = rouge.compute_rouge(refs, systems)
print(scores)

ref_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref2.txt"
system_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/models/0821-213049/dev.out.1"

inputs = []
systems = []
refs = []

f0 = open('input-word.txt', encoding='utf-8')
f1 = open(system_file, encoding='utf-8')
f2 = open(ref_file, encoding='utf-8')

for l0, l1, l2 in zip(f0, f1, f2):
    if not l0:
        break
    l0 = l0.strip()
    # if l0 != 'unknown_word':
    systems.append(l1.strip()[:75])
    refs.append([l2.strip()])

print(len(systems))
scores = rouge.compute_rouge(refs, systems)
print(scores)

ref_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref3.txt"
system_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/models/0821-213049/dev.out.1"

inputs = []
systems = []
refs = []

f0 = open('input-word.txt', encoding='utf-8')
f1 = open(system_file, encoding='utf-8')
f2 = open(ref_file, encoding='utf-8')

for l0, l1, l2 in zip(f0, f1, f2):
    if not l0:
        break
    l0 = l0.strip()
    # if l0 != 'unknown_word':
    systems.append(l1.strip()[:75])
    refs.append([l2.strip()])

print(len(systems))
scores = rouge.compute_rouge(refs, systems)
print(scores)
# ref_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref.txt"
# system_file = "/data/QJXMS/SEASS-FASTCNN/data/giga/models/0821-213049/dev.out.1"
# ref_file0 = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref0.txt"
# ref_file1 = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref1.txt"
# ref_file2 = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref2.txt"
# ref_file3 = "/data/QJXMS/SEASS-FASTCNN/data/giga/duc/task1_ref3.txt"
# inputs = []
# systems = []
# refs = []
#
# f0 = open('input-word.txt', encoding='utf-8')
# f1 = open(system_file, encoding='utf-8')
# f20 = open(ref_file0, encoding='utf-8')
# f21 = open(ref_file1, encoding='utf-8')
# f22 = open(ref_file2, encoding='utf-8')
# f23 = open(ref_file3, encoding='utf-8')
#
# for l0, l1, l20,l21,l22,l23 in zip(f0, f1, f20,f21,f22,f23):
#     if not l0:
#         break
#     l0 = l0.strip()
#     # if l0 != 'unknown_word':
#     systems.append(l1.strip()[:75])
#     refs.append([l20.strip(),l21.strip(),l22.strip(),l23.strip()])
#
# print(len(systems))
# scores = rouge.compute_rouge(refs, systems)
# print(scores)