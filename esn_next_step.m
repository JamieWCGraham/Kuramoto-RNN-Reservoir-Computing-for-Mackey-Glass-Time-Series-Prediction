function [Y, gt, Wout] = esn_next_step(data, args)

arguments
    data
    args.trainLen = 2000
    args.testLen = 2000
    args.initLen = 100
    args.resSize = 1500
    args.leak = .5
    args.rho = 1.25
    args.Wout
end


leak = args.leak;
initLen = args.initLen;
resSize = args.resSize;
inSize = 1;
outSize = 1;
rng('default');
Win = (rand(args.resSize,1+inSize)-0.5) .* 1;
W = rand(args.resSize,args.resSize)-0.5;

% normalizing and setting spectral radius
opt.disp = 0;
rhoW = abs(eigs(W,1,'LM',opt));
W = W .* (args.rho / rhoW);

X = zeros(1+inSize+args.resSize,args.trainLen-args.initLen);
Yt = data(args.initLen+2:args.trainLen+1)';
x = zeros(args.resSize,1);

for t = 1:args.trainLen
	u = data(t);
	x = (1-leak)*x + leak*tanh(Win*[1;u] + W*x );
	if t > initLen
		X(:,t-initLen) = [1;u;x];
	end
end

% train the output by ridge regression
reg = 1e-8;  % regularization coefficient
if ~isfield(args,'Wout')
    Wout = ((X*X' + reg*eye(1+inSize+resSize)) \ (X*Yt'))'; 
else
    Wout = args.Wout;
end

% run the trained ESN in a generative mode. no need to initialize here, 
% because x is initialized with training data and we continue from there.
Y = zeros(outSize, args.testLen);
u = data(args.trainLen+1);
for t = 1:args.testLen 
	x = (1-leak)*x + leak*tanh((Win*[1;u]) + W*x );
	y = Wout*[1;u;x];
	Y(:,t) = y;
	% generative mode:
% 	u = y;
	% this would be a predictive mode:
	u = data(args.trainLen+t+1);
    gt(t) = data(args.trainLen+t+1);
end

end