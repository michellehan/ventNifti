arguments = ""
parameters = {
    ######## dataset #######
    "raw-dir"           : "/data/NormalVentricle/niftis/t2",
    "segs-dir"          : "/data/NormalVentricle/niftis/t2_segs",

    "train-csv"         : "/data/NormalVentricle/labels/nifti_t2_train.csv",
    "val-csv"           : "/data/NormalVentricle/labels/nifti_t2_val.csv",
    "test-csv"          : "/data/NormalVentricle/labels/nifti_t2_test.csv",
    "test-slices"          : "/data/NormalVentricle/labels/t2_test.csv",

    ####### model variables #######
    "dataset"           : "ventricleNormal",
    "encoder"           : "vgg11",
    #"encoder"           : "resnet50",
   # "arch"              : "DynamicUnet",
    "arch"              : "UNet",
    "batch-size"        : 16,
    "labeled-batch-size": 16,
    "start-epoch"      	: 0,
    "epochs"            : 50,
    "evaluation-epochs" : 1,            # how often do you check in during training
    "checkpoint-epochs" : 1,          # how often do save a checkpoint during training

    "num-classes"       : 2,
    "lr"                : 0.05,
    "lr-decay"          : 5,                       # lr *= 0.25 every n epochs
    "momentum"          : 0.8,
    "weight-decay"      : 0.0001,
    "ema-decay"         : 0.999,
    "nesterov"          : False,            # use nesterov momentum
    "consistency"       : 0,             # use consistency loss with given weight


    ####### running parameters #######
    "flag"              : "full",             # full, balanced, or unbalanced training
    "evaluate"          : 0,             # evaluate model? 0 or 1; if =1, log must be turned off
    "resume"          	: 0,                        
    "ckpt"             	: "best",                   # specify best or final

    "log"               : 1,             # log to text file
    "print-freq"        : 25,                 # console print progress for every n batches


    ####### GPU parameters #######
    "seed"              : 2,
    "workers"           : 4,
}

for key, value in parameters.items(): arguments += "--" + str(key) + " "+ str(value) + " "
print(arguments)

