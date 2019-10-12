import csv
import re
from os import path

avg_col = 7
metric_index = 3
#all_level1_bench = ['backprop', 'bfs', 'b+tree', 'cfd', 'dwt2d', 'gaussian', 'heartwall', 'hotspot', 'hotspot3D', 'huffman', 'hybridsort', 'kmeans', 'lavaMD', 'leukocyte', 'lud', 'myocyte', 'nn', 'nw', 'particlefilter', 'pathfinder', 'srad_v1', 'srad_v2', 'streamcluster']
all_level1_bench = ['bfs','fft','gemm','md','md5hash','neuralnet','reduction','scan','sort','spmv','stencil2d','triad']
all_level2_bench = ['s3d', 'qtclustering']

metrics_path = '/home/edwardhu/shoc/src/cuda/'
level1_metrics_path = '/home/edwardhu/shoc/src/cuda/' + 'level1/'
level2_metrics_path = '/home/edwardhu/shoc/src/cuda/' + 'level2/'
small_metrics_filename = '/metrics_small_zemaitis.csv'
big_metrics_filename = '/metrics_big_zemaitis.csv'

def open_file(filename, benchmark):
    dict_to_return = {"benchmark name": benchmark}
    # check whether the file exists
    if not path.exists(filename):
        print('no such file')
        exit(1)

    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count < 5:
                pass
            elif line_count == 5:
                #print(row)
                pass
            elif line_count > 5 and row[0].startswith("Dev"):
                pass
            else:
                if row[0].startswith("=") and line_count > 7:
                    print("fuck this shit\n\n")
                    break 
                key = row[metric_index]
                avg = row[avg_col]

                if avg.endswith("%"):
                    avg_val = re.findall('\d*\.?\d+', avg)
                    if (len(avg_val) > 1):
                        print("wrong parsing 1")
                        exit(1)
                    avg_val[0] = float(avg_val[0]) / 100
                else:
                    avg_val = re.findall('\d*\.?\d+', avg)
                    if (len(avg_val) > 1):
                        print("wrong parsing 2")
                        exit(1)

                # print(avg_val[0])
                
                if key not in dict_to_return:
                    dict_to_return[key] = avg_val[0]
                else:
                    tmp = dict_to_return[key]
                    if float(tmp) < float(avg_val[0]):
                        dict_to_return[key] = avg_val[0]
    
            line_count += 1
    
    return dict_to_return

def main():
    # small metrics
    with open('all_small_metrics.csv', mode='w') as csv_file:
        # level1 first
        for i, bench in enumerate(all_level1_bench):
            full_path = level1_metrics_path + bench + small_metrics_filename
            result = open_file(full_path, bench)
            if i == 0:
                fieldnames = result.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(result)
            else:
                writer.writerow(result)
        # level2
        for i, bench in enumerate(all_level2_bench):
            full_path = level2_metrics_path + bench + small_metrics_filename
            result = open_file(full_path, bench)
            if i == 0:
                fieldnames = result.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                #writer.writeheader()
                writer.writerow(result)
            else:
                writer.writerow(result)

        # remove header row introduced in level2

    # big events
    with open('all_big_metrics.csv', mode='w') as csv_file:
        # level1 first
        for i, bench in enumerate(all_level1_bench):
            full_path = level1_metrics_path + bench + big_metrics_filename
            result = open_file(full_path, bench)
            if i == 0:
                fieldnames = result.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(result)
            else:
                writer.writerow(result)
        # level2
        for i, bench in enumerate(all_level2_bench):
            full_path = level2_metrics_path + bench + big_metrics_filename 
            result = open_file(full_path, bench)
            if i == 0:
                fieldnames = result.keys()
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                #writer.writeheader()
                writer.writerow(result)
            else:
                writer.writerow(result)




            # for key in result:
            #     print(key, "    ", result[key])


if __name__ == "__main__":
    main()
