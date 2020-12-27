% Perform an armijo line search and return the updated matrix
function graph = armijo(total_graph, active_mask, constraint_graph, size, length, gradient_diff, sigma, alpha, gamma)
    grad_at_x = get_graph_gradient(total_graph, active_mask, constraint_graph, size, length, gradient_diff);
    val_at_x = eval_graph(total_graph, constraint_graph, size, length);
    current_step = alpha;
    descent_direction = -grad_at_x;
    rhs_inner_prod = norm(grad_at_x(2:size-1,2:size-1), 'fro')^2; % The norm squared of the gradient
    for i=1:100
        tmp_mat = total_graph + current_step * descent_direction;
        step_val = eval_graph(tmp_mat, constraint_graph, size, length);
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

