#====================================================================
    # Gradient Test Fucntion
    # translate Sasha's MATLAB code into Julia
    #
    # input:
    #     function handle fcn: ℝⁿ -> ℝ
    #     initial point x
    #     pert: maximum element-wise perturbation, e.g. e-3.
    #
    # output:
    #     err is the deviation, as explained in gradientTest.pdf,
    #     between 1 and the quantity that should be close to 1.

#====================================================================#

import numpy as np

def gradientTest(fcn, x0, pert):

    epsilon = pert*np.random.randn(len(x0));

    c = 0.1;
    epsilon *= 1.0/c**2;
    print("Running gradient test:");

    f0, g0 = fcn(x0);
    x1 = np.zeros(len(x0));

    print("Change, result");
    for iter in range(6):
        epsilon *= c;
        for i in range(len(x0)):
            x1[i] = x0[i] + epsilon[i];
        f1, g1 = fcn(x1);

        err = 0.0;
        for i in range(len(x0)):
            err += (g0[i] + g1[i])*epsilon[i];
        err = 1.0 - 0.5*err/(f1 - f0);
        print("‖epsilon‖:",np.linalg.norm(epsilon),"err :", err);
