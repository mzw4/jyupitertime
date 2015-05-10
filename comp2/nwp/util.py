import numpy as np

#make emission
with open("pytest.txt", 'w+') as my_file:
    for i in range(2,83):
        my_file.write("<State> "+str(i)+"\n")
        my_file.write("\t<Mean> 13\n")
        my_file.write("\t\t0.2 0.1 0.1 0.9 0.2 0.1 0.1 0.9 0.2 0.1 0.1 0.9 0.9\n")
        my_file.write("\t<Variance> 13\n")
        my_file.write("\t\t1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0\n")

#make transition
T= np.eye(83, k=1, dtype=float)
np.savetxt("my_T.txt",  T, fmt="%1.1f")


#vector2mfc stuff
import struct
def vector_2_mfc(vector, output_filename="test.mfc", path=""):
    #Set parameters
    n_samples = 83
    samp_period = 100000
    samp_size = 52 #num of bytes per sample (13 floats = 13 * 4 = 52)
    parm_kind = 6 #mfc

    #convert to byes
    samples_bytes = struct.pack(">i", n_samples)
    samp_period_bytes = struct.pack(">i", samp_period)
    samp_size_bytes = struct.pack(">h", samp_size)
    parm_kind_bytes = struct.pack(">h", parm_kind)
    
    feats_bytes = None
    for feat in vector:
        feat_bytes  = struct.pack(">f", feat)
        feats_bytes = feat_bytes if feats_bytes is None else (feats_bytes+feat_bytes)
    
    #Combine all
    combined_bytes = samples_bytes + samp_period_bytes + samp_size_bytes + parm_kind_bytes + feats_bytes
    
    with open(path+output_filename, 'w+') as outfile:
        outfile.write(combined_bytes)
    print "Done writing mfc file"     

#make data for HMM input
#train_data = np.loadtxt('train.data', delimiter=',')
#train_labels = np.loadtxt('train.labels', delimiter=',', skiprows=1)

data_path = "data/"
prev_label = 0
cnt = 0
for label, vector in zip(train_labels[:,1], train_data):
    label = int(label)
    if label != prev_label:
        cnt = 0
    vector_filename = "vector_l"+str(label)+"_"+str(cnt)+".mfc"
    
    vector_2_mfc(vector, output_filename=vector_filename, path=data_path) #also saves file to disk
    
    with open("trainlist_"+str(label), 'a+') as trainlist_file:
        trainlist_file.write(data_path+vector_filename+"\n")
    cnt += 1
    prev_label = label       