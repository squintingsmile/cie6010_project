function [graph, obj_diff, obj_val, grad_norm, newton_or_armijo,...
    constraint] = penalty(total_graph, constraint_graph, size,...
    length, gradient_diff, sigma, alpha, gamma, beta1, beta2, p, a, a_max)

    if a > a_max
        a = a_max;
    end
%     fprintf("running\n");
    gradient = get_graph_gradient(total_graph, constraint_graph, size, length, gradient_diff);
    hessian = get_graph_hessian(total_graph, constraint_graph, size, length, gradient_diff);
    
    % Transforming the descent direction and gradient from the 2d matrix to vector form
    % while ignoring the boundary elements
    gradient_transpose = transpose(gradient(2:size-1, 2:size-1));
    gradient_vec = gradient_transpose(:);
    
    constraint = total_graph > constraint_graph;
    constraint_transpose = transpose(constraint(2:size-1, 2:size-1));
    constraint_mask = constraint_transpose(:);
    
    constraint_violation = constraint_graph - total_graph;
    constraint_violation_transpose = transpose(constraint_violation(2:size-1, 2:size-1));
    constraint_violation_vec = constraint_violation_transpose(:);
    constraint_violation_vec(constraint_mask) = 0;
    
    
    constraint_grad = -a * constraint_violation_vec;
    constraint_hessian = zeros((size-2)^2);
    
%     constraint_grad
    for i=1:(size-2)^2
        if constraint_mask == false
            constraint_hessian(i, i) = a;
        end
    end
    
    hessian = hessian + constraint_hessian;
    gradient_vec = gradient_vec + constraint_grad;
    
    [descent_direction, r_condition] = linsolve(hessian, -gradient_vec);
    
%     hessian
%     gradient
%     gradient_vec
%     descent_direction
%     
%       hessian
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
    
    val_at_x = eval_graph(total_graph, constraint_graph, size, length)...
        + a * norm(constraint_violation_vec)^2 / 2;
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

        constraint = tmp_mat > constraint_graph;
        constraint_transpose = transpose(constraint(2:size-1, 2:size-1));
        constraint_mask = constraint_transpose(:);

        constraint_violation = constraint_graph - tmp_mat;
        constraint_violation_transpose = transpose(constraint_violation(2:size-1, 2:size-1));
        constraint_violation_vec = constraint_violation_transpose(:);
        constraint_violation_vec(constraint_mask) = 0;
        
        step_val = step_val + a * norm(constraint_violation_vec)^2 / 2;

        if step_val - val_at_x > - gamma * current_step * rhs_inner_prod
            current_step = current_step * sigma;
            continue;
        end
        constraint = norm(constraint_violation_vec);
%         constraint_violation_vec
        break;
    end
    graph = tmp_mat;    
%     graph
    obj_val = eval_graph(total_graph, constraint_graph, size, length);
    grad_norm = norm(gradient_vec);
end

