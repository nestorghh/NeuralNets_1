% activation function
% takes the net input and outputs 1 or -1.
function [y]=activation(y_net)
    if y_net>=0
        y=1;
    else
        y=-1;
    end
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
