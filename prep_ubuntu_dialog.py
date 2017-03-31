import csv
import glob


for fn in glob.glob('./datasets/ubuntu/*'):
    reader   = csv.reader(open(fn), delimiter=',')
    filtered = filter(lambda p: p[2] == '1', reader)
    csv.writer(open(fn[:-4] + '-filtered.csv', 'w'), delimiter=',').writerows(filtered)
