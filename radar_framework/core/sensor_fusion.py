import numpy as np

class SensorFusion:
    @staticmethod
    def fuse(states, covariances):
        """
        Fuse N state estimates with covariances via optimal linear fusion.
        P_f = inv(sum(inv(P_i)))
        x_f = P_f * sum(inv(P_i) * x_i)
        """
        inv_sum = sum(np.linalg.inv(P) for P in covariances)
        P_f = np.linalg.inv(inv_sum)
        x_f = P_f @ sum(np.linalg.inv(P) @ x for x, P in zip(states, covariances))
        return x_f, P_f