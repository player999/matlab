function [net_out tr] = trainscg2(net, alphabet, targets)    
    num_threads = net.userdata.num_threads;
    numLayers = net.numLayers;
    Hs = zeros(numLayers, 1);
    Ws = zeros(numLayers, 1);
    weights = reshape(net.IW{1}, size(net.IW{1}, 1) * size(net.IW{1}, 2), 1);
    weights = cat(1, weights, net.b{1});
    Hs(1) = size(net.IW{1}, 1);
    Ws(1) = size(net.IW{1}, 2) + 1;
    
    for i=2:numLayers
        weights = cat(1, weights, reshape(net.LW{i, i-1}, size(net.LW{i, i-1}, 1) * size(net.LW{i,i-1}, 2), 1));
        weights = cat(1, weights, net.b{i});
        Hs(i) = size(net.LW{i, i-1}, 1);
        Ws(i) = size(net.LW{i, i-1}, 2) + 1;
    end
    set_size = size(alphabet,2);  
    desired_error = net.trainParam.goal;
    max_epochs = net.trainParam.epochs;
    max_time = net.trainParam.time;
    lambda = net.userdata.lambda;
	sigma = net.userdata.sigma;
	ipaddress = net.userdata.ip;
	port = net.userdata.port;
    report_interval = net.userdata.report_interval;
    net.inputs{1}.processFcns = {};
    net.outputs{numLayers-1}.processFcns = {};
    net.divideFcn = '';
    [nweights, mse, time] = trainscg2_call(double(weights), uint32(Hs),...
        uint32(Ws), double(alphabet), double(targets), uint32(numLayers),...
        uint32(num_threads), uint32(set_size), double(desired_error),...
        double(max_epochs), double(max_time), double(lambda),...
        double(sigma), uint32(report_interval), ipaddress, uint32(port)); 
    tr = cat(2, double(mse), double(time));
    
    net_out = net;
    net_out.IW{1} = reshape(nweights(1:Hs(1) * (Ws(1) - 1)), Hs(1), Ws(1) - 1);
    net_out.b{1} = nweights(Hs(1) * (Ws(1) - 1) + 1 : Hs(1) * Ws(1)); 
    bias = 0;
    for i=2:numLayers
       bias = bias + Hs(i - 1) * Ws(i - 1);
       net_out.LW{i, i - 1} = reshape(nweights(bias + 1 : bias + Hs(i) * (Ws(i) - 1)), Hs(i), Ws(i) - 1);
       net_out.b{i} = nweights(bias + Hs(i) * (Ws(i) - 1) + 1 : bias + Hs(i) * Ws(i));       
    end
    