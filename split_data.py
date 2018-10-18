import os, csv
from glob import glob
from sklearn.cross_validation import train_test_split

t2_dir = '/data/NormalVentricle/jpgs/t2'
t2_segs_dir = '/data/NormalVentricle/jpgs/t2_segs'
spgr_dir = '/data/NormalVentricle/jpgs/spgr'
spgr_segs_dir = '/data/NormalVentricle/jpgs/spgr_segs'

t2_csv = "/data/NormalVentricle/labels/t2"
spgr_csv = "/data/NormalVentricle/labels/spgr"


def write_csv(csvroot, suffix, x, y):
    csvfile = csvroot + "_" + suffix + ".csv"
    print(csvfile)
    print(len(x), len(y))

    with open(csvfile, 'w+') as csvfile:
        for i in range(len(x)):
            writer = csv.writer(csvfile)  
            writer.writerow([x[i], y[i]])


def subj_to_files(subjs, files):
    x = []
    for subj in subjs:
        x.append([filename for filename in files if subj in filename])
    
    return [item for sublist in x for item in sublist]


def split(x_dir, y_dir, csvpath):
    x_files = sorted([os.path.basename(x) for x in glob('%s/*' %x_dir)])
    y_files = sorted([os.path.basename(x) for x in glob('%s/*' %y_dir)])

    x, y = [], []
    for f in x_files:
        f = f.split('-')[0]
        f_seg = f + "_seg" 
        x.append(f)
        y.append(f_seg)
    x = sorted(set(x))
    y = sorted(set(y))

    #split data by subject ID
    x_train_subj, x_test_subj, y_train_subj, y_test_subj = train_test_split(x, y, test_size = 0.3)
    x_test_subj, x_val_subj, y_test_subj, y_val_subj = train_test_split(x_test_subj, y_test_subj, test_size = 0.5)

    #get all images for each subject 
    x_train = subj_to_files(x_train_subj, x_files)
    y_train = subj_to_files(y_train_subj, y_files)
    x_val = subj_to_files(x_val_subj, x_files)
    y_val = subj_to_files(y_val_subj, y_files)
    x_test = subj_to_files(x_test_subj, x_files)
    y_test = subj_to_files(y_test_subj, y_files)

    #write the csv
    write_csv(csvpath, "train", x_train, y_train)
    write_csv(csvpath, "val", x_val, y_val)
    write_csv(csvpath, "test", x_test, y_test)




split(t2_dir, t2_segs_dir, t2_csv)
split(spgr_dir, spgr_segs_dir, spgr_csv)

