% Run this as it is the main function, wait a minute as it will take a
% minute or so to run

data = make_mg(0.5);

[Y, gt, Wout] = esn_clf(data);

figure;
plot(Y); hold on; plot(gt)
title("Mackey Glass Time Series Prediction Closed-Loop-Forecasting")
xlabel("time (s)")
ylabel("Mackey-Glass Signal")
legend(["Prediction","Ground Truth"])

[Y_next_step, gt, Wout] = esn_next_step(data);

figure;
plot(Y_next_step); hold on; plot(gt)
title("Mackey Glass Time Series Prediction Next-Step-Forecasting")
xlabel("time (s)")
ylabel("Mackey-Glass Signal")
legend(["Prediction","Ground Truth"])
