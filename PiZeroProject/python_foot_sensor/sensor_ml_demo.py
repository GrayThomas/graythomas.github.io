import numpy as np
import matplotlib.pyplot as plt

sensor_A_true = lambda f, tau: np.sin(f/1000-.2)*50+23 + np.random.normal()* 0.5
sensor_B_true = lambda f, tau: np.sin(f/1000+.2)*50-12+tau*.1+ np.random.normal()* 0.7
sensor_C_true = lambda f, tau: np.sin(f/100+.1)*10+6+tau*.25+ np.random.normal()* 1.2


regressor_row = lambda s1, s2, s3: np.array([
        1, s1, s2, s3, s1**2, s2**2, s3**2, s1*s2, s2*s3, s1*s3])

def true_model(time):
    step_freq = 1.0
    grf = 100*np.sin(time*step_freq*2*np.pi)
    if grf<0:
        grf=0

    phase = time*step_freq % 1.0
    tau_grf = phase*40 if phase<.5 else 0.0


    all_sensor_data = np.array([time, phase, grf, tau_grf,
        sensor_A_true(grf, tau_grf), 
        sensor_B_true(grf, tau_grf), 
        sensor_C_true(grf, tau_grf),
        ])
    return all_sensor_data


def generate_fake_data(filename):
    file_data = []
    for t in np.linspace(0, 10, 1000):
        file_data.append(true_model(t))
    file_data = np.array(file_data)
    np.savetxt(filename, file_data)

def fit_model(data):

    

    RtR_accumulator = 0
    RtY_accumulator = 0

    # look at the data elementwise
    for time, phase, grf, tau, s1, s2, s3 in data:
        # print (time, phase, grf, tau, s1, s2, s3)
        rr = regressor_row(s1, s2, s3)
        y = np.array([grf, tau])

        RtR_accumulator += rr.reshape((-1,1)) @ rr.reshape((1,-1))
        RtY_accumulator += rr.reshape((-1,1)) @ y.reshape((1, -1))

    # solve the regression y = Rx for x using least squares
    x = np.linalg.solve(RtR_accumulator, RtY_accumulator)
    # print(x)
    row_model = lambda s1, s2, s3: (regressor_row(s1, s2, s3) @ x).reshape((-1,))
    model = lambda mat: np.array([row_model(*row) for row in mat])

    return model




def main():
    generate_fake_data("tmp1.csv")
    data = np.loadtxt("tmp1.csv")
    # plt.plot(data[:,0], data[:,1:]) # plot everything from the data
    model = fit_model(data)

    fit = model(data[:, 4:])

    plt.plot(data[:, 0], data[:, 1], label="phase")
    plt.plot(data[:, 0], fit[:,0], label="grf_est")
    plt.plot(data[:, 0], fit[:,1], label="tau_est")
    plt.plot(data[:, 0], data[:, 2], label="grf")
    plt.plot(data[:, 0], data[:, 3], label="grf_tau")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()