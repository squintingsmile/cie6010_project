% Perform an armijo line search with nesterov acceleration and return the updated matrix
% The @obj_diff the the difference between the updated function value and
% the original function value
function [graph, obj_diff, obj_val, grad_norm] = armijo_nesterov(total_graph,...
    constraint_graph, size, length, gradient_diff, sigma, alpha, gamma, prev_graph, iter)

    if iter ~= 1
        beta = 2 / (iter + 2);
        total_graph = total_graph + beta * (total_graph - prev_graph);
    end
    
    grad_at_x = get_graph_gradient(total_graph, constraint_graph, size, length, gradient_diff);
    val_at_x = eval_graph(total_graph, constraint_graph, size, length);
    current_step = alpha;
    descent_direction = -grad_at_x;
    
    rhs_inner_prod = norm(grad_at_x(2:size-1,2:size-1), 'fro')^2; % The norm squared of the gradient
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
        obj_diff = val_at_x - step_val;
        break;
    end
    graph = tmp_mat;
    
    direction = descent_direction;
    obj_val = val_at_x;
    grad_norm = norm(grad_at_x);
end

