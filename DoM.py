import os
import numpy
import scipy.spatial as sp

def get_dom(path):
    compare_matrixes=list()
    matrix3 = None
    matrix4 = None
    for folder in os.listdir(path):
        all_matrix = list()
        first = True
        for filename in os.listdir(path+'/'+str(folder)):
            data = numpy.load(f'{path}/{folder}/{filename}')
            if first:
                if matrix4 is None:
                    matrix4 = data
                else:
                    matrix3 = data
                first = False
                continue
            all_matrix.append(numpy.asmatrix(data))
        mean_matrix = numpy.mean(all_matrix, axis=0)
        numpy.save(f'{path}/{folder}/mean_map.npy', mean_matrix)
        compare_matrixes.append(mean_matrix)
    dist = numpy.linalg.norm(compare_matrixes[0] - compare_matrixes[1])
    dist3_3 = numpy.linalg.norm(compare_matrixes[1] - matrix3)
    dst = 1 - sp.distance.cosine(compare_matrixes[1].ravel(), matrix3.ravel())
    dst2 = 1 - sp.distance.cosine(compare_matrixes[1].ravel(), matrix4.ravel())
    dist4_4 = numpy.linalg.norm(compare_matrixes[0] - matrix4)
    dist3_4 = numpy.linalg.norm(compare_matrixes[0] - matrix3)
    dist4_3 = numpy.linalg.norm(compare_matrixes[1] - matrix4)
    print(f'distance between the means:{dist}')
    print(f'distance between the same(3-3, should be small):{dist3_3}')
    print(f'distance between the same(3-3, should be small):{dst}')
    print(f'distance between different(3-4):{dst2}')
    print(f'distance between different(3-4):{dist3_4}')
    print(f'distance between the same(4-4, should be small):{dist4_4}')
    print(f'distance between different(4-3):{dist4_3}')


def main():
    cwd = os.getcwd()
    results_path = f'{cwd}/train'
    get_dom(results_path)

if __name__ == '__main__':
    main()