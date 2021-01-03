function [graph, obj_diff, obj_val, grad_norm,constraint, updated_z, updated_y,...
    primal_res, dual_res] = admm(total_graph, constraint_graph, size,...
    length, gradient_diff, sigma, alpha, gamma, beta1, beta2, p, rho,...
    zk, yk)

    iter_count = 128;
    
%     total_graph
%     constraint_graph
    constraint = total_graph > constraint_graph;
    constraint_transpose = transpose(constraint(2:size-1, 2:size-1));
    constraint_mask = constraint_transpose(:);

    constraint_violation = constraint_graph - total_graph;
    constraint_violation_vec = ravel_graph_transpose(constraint_violation, size);
    constraint_violation_vec(constraint_mask) = 0;
    constraint = norm(constraint_violation_vec);
    
    obj_diff = eval_graph(total_graph, constraint_graph, size, length);
    
    xk = ravel_graph_transpose(total_graph, size);
    % Minimizing the augmented lagrangian with respect to x using the
    % globalized newton's method
    for iter=1:iter_count
    %     fprintf("running\n");
        gradient = get_graph_gradient(total_graph, constraint_graph, size, length, gradient_diff);
        
        x = ravel_graph_transpose(total_graph, size);
        
        % Transforming the descent direction and gradient from the 2d matrix to vector form
        % while ignoring the boundary elements
        gradient_vec = ravel_graph_transpose(gradient, size);
        if iter == 1
            grad_norm = norm(gradient_vec);
        end
        
        gradient_vec = gradient_vec + yk + rho * (x - zk);
        if norm(gradient_vec) < 1e-4
%             fprintf("normally exiting");
            break;
        end
        
        hessian = get_graph_hessian(total_graph, constraint_graph, size, length, gradient_diff);
        hessian = hessian + rho * eye((size - 2)^2);
       
        
        [descent_direction, r_condition] = linsolve(hessian, -gradient_vec);

        % If the matrix is ill-conditioned or does not satisfy the condition, 
        % use armijo. Otherwise use newton direction
        if r_condition < 1e-12 || (-dot(gradient_vec, descent_direction) <...
                min(beta1, beta2 * norm(descent_direction)^p) * norm(descent_direction)^2)
            descent_direction = -gradient_vec;
        end
        
%         descent_direction

        val_at_x = eval_graph(total_graph, constraint_graph, size, length)...
            + dot(yk, x - zk) + rho * norm(x - zk)^2 / 2;
        current_step = alpha;
        rhs_inner_prod = -dot(descent_direction, gradient_vec);

        % Transforming the vector form of descent direction back to matrix form
        tmp_mat = zeros(size);
        tmp_mat(2:size-1, 2:size-1) = reshape(descent_direction, [size-2,size-2]);
        descent_direction = transpose(tmp_mat);
        
        for i=1:32
            tmp_mat = total_graph + current_step * descent_direction;
            tmp_vec = ravel_graph_transpose(tmp_mat, size);
            step_val = eval_graph(tmp_mat, constraint_graph, size, length)...
                + dot(yk, tmp_vec - zk) + rho * norm(tmp_vec - zk)^2 / 2;


            if step_val - val_at_x > - gamma * current_step * rhs_inner_prod
                current_step = current_step * sigma;
                continue;
            end
    %         constraint_violation_vec
            break;
        end
        
%         tmp_mat
        total_graph = tmp_mat;
    end
    
    obj_diff = eval_graph(total_graph, constraint_graph, size, length) - obj_diff;
    
    % Minimizing the augmented lagrangian with respect to z. In this
    % problem minimizing with respect to z is actually a projection problem
    updated_x = ravel_graph_transpose(total_graph, size);
    updated_z = yk / rho + updated_x;
    constraint_vec = ravel_graph_transpose(constraint_graph, size);
    updated_z(updated_z < constraint_vec) = constraint_vec(updated_z < constraint_vec);
    
    % Update y as in admm step
    updated_y = yk + rho * (updated_x - updated_z);
    
    graph = total_graph;
%     graph
    obj_val = eval_graph(total_graph, constraint_graph, size, length);
   
    updated_r = updated_x - updated_z;
    primal_res = norm(updated_r);
    
    updated_s = rho * (zk - updated_z);
    dual_res = norm(updated_s);
end

function vec = ravel_graph_transpose(graph, size)
    graph_transpose = transpose(graph(2:size-1, 2:size-1));
    vec = graph_transpose(:);
end