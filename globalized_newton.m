% This function performs a globalized newton step. The @newton_or_armijo
% variable indicates which method it is actually using
function [graph, obj_diff, newton_or_armijo] = globalized_newton(total_graph,...
    constraint_graph, size, length, gradient_diff, sigma, alpha, gamma, beta1, beta2, p)
    
    gradient = get_graph_gradient(total_graph, constraint_graph, size, length, gradient_diff);
    hessian = get_graph_hessian(total_graph, constraint_graph, size, length, gradient_diff);
    
    % Transforming the descent direction and gradient to vector form
    gradient_transpose = transpose(gradient(2:size-1, 2:size-1));
    gradient_vec = gradient_transpose(:);
    [descent_direction, r_condition] = linsolve(hessian, -gradient_vec);
%     hessian
%     gradient
%     gradient_vec
%     descent_direction
%     
%     hessian
    % If the matrix is ill-conditioned or does not satisfy the condition, 
    % use armijo. Otherwise use newton direction
%     r_condition
%     -dot(gradient_vec, descent_direction)
%     min(beta1, beta2 * norm(descent_direction)^p) * norm(descent_direction)^2
    if r_condition < 1e-12 || (-dot(gradient_vec, descent_direction) <...
            min(beta1, beta2 * norm(descent_direction)^p) * norm(descent_direction)^2)
        descent_direction = -gradient_vec;
        newton_or_armijo = 0;
    else
        newton_or_armijo = 1;
    end
    
    val_at_x = eval_graph(total_graph, constraint_graph, size, length);
    current_step = alpha;
    rhs_inner_prod = -dot(descent_direction, gradient_vec);
    
    % Transforming the vector form of descent direction back to matrix form
    tmp_mat = zeros(size);
    tmp_mat(2:size-1, 2:size-1) = reshape(descent_direction, [size-2,size-2]);
    descent_direction = transpose(tmp_mat);
    for i=1:100
        tmp_mat = total_graph + current_step * descent_direction;
        step_val = eval_graph(tmp_mat, constraint_graph, size, length);
        obj_diff = val_at_x - step_val;
%         current_step
%         step_val
%         val_at_x
        if step_val - val_at_x > - gamma * current_step * rhs_inner_prod
            current_step = current_step * sigma;
            continue;
        end
        break;
    end
    graph = tmp_mat;
    
end

