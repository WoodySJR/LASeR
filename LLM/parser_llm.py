import numpy as np

def parser(output):
    array = output.split("<begin>")[1].split("<end>")[0]
    array = array.split("\n")[1:-1]
    rows = []
    for i in array:
        row = []
        for j in i:
            if j in "0123456789":
                row.append(int(j))
        rows.append(row)
    return np.array(rows)