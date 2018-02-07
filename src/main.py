import numpy as np
from load import load_data
from factorize import PMFactorizer



DATA_FILE='../sample_input/ratings.csv'
train_data = np.genfromtxt(DATA_FILE, delimiter = ",")
train_data=train_data[1:,:-1]
M=load_data(train_data)
M=M[:500,:300]
factor=PMFactorizer(M)

L, U_matrices, V_matrices = factor.run_multiple_iter(50)

np.savetxt("../sample_output/objective.csv", L, delimiter=",")
np.savetxt("../sample_output/U-10.csv", U_matrices[9], delimiter=",")
np.savetxt("../sample_output/U-25.csv", U_matrices[24], delimiter=",")
np.savetxt("../sample_output/U-50.csv", U_matrices[49], delimiter=",")

np.savetxt("../sample_output/V-10.csv", V_matrices[9], delimiter=",")
np.savetxt("../sample_output/V-25.csv", V_matrices[24], delimiter=",")
np.savetxt("../sample_output/V-50.csv", V_matrices[49], delimiter=",")


