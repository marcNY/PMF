import numpy as np


## PMF to find U,V so that M=transpose(U)*V
class PMFactorizer():
    def __init__(self, M, lbda=2, sg2=0.1, d=5):
        self.M = M
        self.n_users, self.n_objects = M.shape
        self.lbda = lbda
        self.sigma2 = sg2
        self.d = d
        self.U, self.V = self.__init_factors()

    def __init_factors(self):
        [m, n] = self.M.shape
        U = np.random.normal(0, 1 / float(self.lbda), [self.d, m])
        V = np.random.normal(0, 1 / float(self.lbda), [self.d, n])
        return U, V

    @staticmethod
    def __update_vect(V1, V2, M, lbda, sigma2):
        def update_vect_index(V2, M, lbda, sigma2, index):
            d = V2.shape[0]
            Id = np.identity(d)

            A = np.zeros((d, d))
            B = np.zeros((d, 1))
            for j in range(0, V2.shape[1]):
                if M[index, j] >= 0:  # not empty
                    Vj = V2[:, j]
                    Vj=Vj[:,np.newaxis]
                    A += np.dot(Vj, np.transpose(Vj))
                    B += M[index, j] * Vj
            A = np.linalg.inv(lbda * sigma2 * Id + A)
            Ui = np.dot(A, B)

            return Ui

        for i in range(0, V1.shape[1]):
            V1[:, i] = np.ravel(update_vect_index(V2, M, lbda, sigma2, i))

        return V1

    def update_factors(self):
        self.U = PMFactorizer.__update_vect(self.U, self.V, self.M, self.lbda, self.sigma2)
        self.V = PMFactorizer.__update_vect(self.V, self.U, np.transpose(self.M), self.lbda, self.sigma2)
        objfunct = self.compute_obj_fct()
        return objfunct, self.U, self.V

    def compute_obj_fct(self):
        E1 = self.M - np.dot(np.transpose(self.U), self.V)
        E1 = (2 * self.sigma2) **-1 * np.square(E1)
        E1[self.M < 0] = 0
        E1 = -np.sum(E1)

        E2 = -self.lbda / 2 * np.sum(np.square(self.U))
        E3 = -self.lbda / 2 * np.sum(np.square(self.V))

        return E1 + E2 + E3

    def run_multiple_iter(self,n_iteration):
        L, U_matrices, V_matrices=[],[],[]
        for i in range(0,n_iteration):
            print('running iteration %s'%i)
            obfct,u,v=self.update_factors()
            L.append(obfct)
            U_matrices.append(np.transpose(u))
            V_matrices.append(np.transpose(v))
        return np.array(L), U_matrices, V_matrices
