% The hessian is stored in the matrix format instead of tensor 
% The matrix is of the form
% [\nabla x_1 \nabla g(x); \nabla x_2 \nabla g(x);...;\nabla x_n \nabla g(x)]
% Here the hessian is a matrix of size [(size-2)^2, (size-2)^2] matrix since we
% ignore the boundary. The index of a movable point xi,j is given by 
% (i - 2) * (size - 2) + j - 1
function hessian = get_graph_hessian(total_graph, constraint_graph, size, length, gradient_diff)
    hessian = zeros((size-2)^2);
    for i=2:size-1
        for j=2:size-1
            % We assume that hessian(i, j) = hessian(j, i). Thus we only
            % need to evaluate on of them 
            if i < j
                continue;
            end
            
            % The gradient of g w.r.t xk only contains variable xt if and 
            % only if xk and xt are in the same triangle. Therefore, we 
            % only the xt such that xk and xt are in the same triangle
            for k=-1:1
                for l=-1:1
                    xt_i = i + k;
                    xt_j = j + l;
                    xk_i = i;
                    xk_j = j;
                    
                    if is_on_boundary(size, xt_i, xt_j) == 1 ||...
                            (k == -1 && l == 1) || (k == 1 && l == -1)
                        continue;
                    end
%                     fprintf("i:%d,j:%d,k:%d,l:%d", xk_i, xk_j,xt_i, xt_j);
                    samples_grads = zeros(1,3); % Storing the gradient of g(xt+h, xk)
                                  % g(xt,xk) and g(xt-h,xk) where we want is
                                  % the hessian at entry [xk,xt]
                    tmp_val_xt = total_graph(xt_i, xt_j);
                    
                    % calculating dg(xt+h, xk) / dxk, dg(xt, xk) / dxk
                    % and dg(xt-h, xk) / dxk
                    for sample_count1=-1:1
                        
                        samples = zeros(1, 3);
                        total_graph(xt_i, xt_j) = tmp_val_xt + sample_count1 * gradient_diff;
                        
                        tmp_val_xk = total_graph(xk_i, xk_j);
                        
                        % Calculating g(xt+ah, xk+h), g(xt+ah, xk) and
                        % g(xt+ah, xk-h) where a = -1, 0, 1
                        for sample_count2=-1:1
                            total_graph(xk_i, xk_j) = tmp_val_xk + sample_count2 * gradient_diff;
%                             total_graph
                            samples(sample_count2 + 2) = eval_triag_area_relative_to_point(xk_i, xk_j, total_graph, length); 
                        end
%                         samples
                        grad = gradient(samples, gradient_diff);
                        samples_grads(sample_count1 + 2) = grad(2);
                        total_graph(xk_i, xk_j) = tmp_val_xk;
                    end
%                     samples_grads
                    grad = gradient(samples_grads, gradient_diff);
%                     grad
                    xt_idx = (xt_i - 2) * (size -2) + xt_j - 1; % Here we treat the xi,j as a column vector instead of a 2d matrix
                                         % Therefore, we need to calculate
                                         % the corresponding index in the
                                         % column.
                    xk_idx = (xk_i - 2) * (size -2) + xk_j - 1;
                    
                    hessian(xk_idx, xt_idx) = grad(2);
                    hessian(xt_idx, xk_idx) = grad(2);
                    
                    total_graph(xt_i, xt_j) = tmp_val_xt;
                end
            end
%                 samples

        end
    end
end

% Decide whether a point is on boundary or not
function val = is_on_boundary(size, i, j)
    if (j == 1) || (j == size) || (i == 1) || (i == size)
        val = 1;
        return;
    end
    
    val = 0;
    
end


% The arrangement of x1, x2, x3, and x4 is as follows
% The length is the length of the square
% x1-x4
% | \ |
% x2-x3
function val = eval_square(x1, x2, x3, x4, length)
    val = 1/2 * sqrt( ((x2 - x1)^2 + length^2)*((x2 - x3)^2 + length^2) - ((x2 - x3)*(x2 - x1))^2 ) ...
        + 1/2 * sqrt( ((x4 - x1)^2 + length^2)*((x4 - x3)^2 + length^2) - ((x4 - x3)*(x4 - x1))^2 );
end

% The arrange of x1, x2, and x3 is as follows
% x1         x1-x2
% | \   or     \|
% x2-x3         x3
function val = eval_triag(x1, x2, x3, length)
    val = 1/2 * sqrt( ((x2 - x1)^2 + length^2)*((x2 - x3)^2 + length^2) - ((x2 - x3)*(x2 - x1))^2 );
end

% The arrangement of the square is as follows
% Here we calculate the sum of the area of the triangulars that
% uses x5 as a pivot. x5 is the @total_graph(i,j) here
% x1-x4
% | \| \
% x2-x5-x8
%   \| \|
%    x6-x9
function val = eval_triag_area_relative_to_point(i, j, total_graph, length)
    val = 0;
    tmp = [0,0;1,1]; % Used to simplify the operation of calculating the triangular connecting to a point

    for count = 1:2
        x1 = total_graph(i + tmp(count,1) - 1, j + tmp(count,2) - 1);
        x2 = total_graph(i + tmp(count,1), j + tmp(count,2) - 1);
        x3 = total_graph(i + tmp(count,1), j + tmp(count, 2));
        x4 = total_graph(i + tmp(count,1) - 1, j + tmp(count,2));

        val = val + eval_square(x1, x2, x3, x4, length);
    end
    
    x1 = total_graph(i, j-1);
    x2 = total_graph(i, j);
    x3 = total_graph(i+1, j);
    val = val + eval_triag(x1, x2, x3, length);
    
    x1 = total_graph(i-1, j);
    x3 = total_graph(i, j+1);
    val = val + eval_triag(x1, x2, x3, length);
end
