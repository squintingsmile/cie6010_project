function [graph, obj_diff, obj_val, grad_norm, newton_or_armijo,...
    constraint, updated_z, updated_y] = admm(total_graph, constraint_graph, size,...
    length, gradient_diff, sigma, alpha, gamma, beta1, beta2, p, rho,...
    zk, yk)

    iter_count = 200;
    
    total_graph
    constraint_graph
    constraint = total_graph > constraint_graph;
    constraint_transpose = transpose(constraint(2:size-1, 2:size-1));
    constraint_mask = constraint_transpose(:);

    constraint_violation = constraint_graph - total_graph;
    constraint_violation_vec = ravel_graph_transpose(constraint_violation, size);
    constraint_violation_vec(constraint_mask) = 0;
    constraint = norm(constraint_violation_vec);
    yk
    zk
    for iter=1:iter_count
    %     fprintf("running\n");
        gradient = get_graph_gradient(total_graph, constraint_graph, size, length, gradient_diff);
        

        x = ravel_graph_transpose(total_graph, size);
        % Transforming the descent direction and gradient from the 2d matrix to vector form
        % while ignoring the boundary elements
        gradient_vec = ravel_graph_transpose(gradient, size);
        gradient_vec
        if iter == 1
            grad_norm = norm(gradient_vec);
        end
        
    %     gradient_vec
        gradient_vec = gradient_vec + yk + rho * (x - zk);
        gradient_vec
        total_graph

%         gradient_vec

        if norm(gradient_vec) < 1e-5
            fprintf("normally exiting");
            break;
        end
        
        hessian = get_graph_hessian(total_graph, constraint_graph, size, length, gradient_diff);
        hessian = hessian + rho * eye((size - 2)^2);
       
        
        [descent_direction, r_condition] = linsolve(hessian, -gradient_vec);
%         descent_direction
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
        
        descent_direction

        val_at_x = eval_graph(total_graph, constraint_graph, size, length)...
            + dot(yk, x - zk) + rho * norm(x - zk)^2 / 2;
        current_step = alpha;
        rhs_inner_prod = -dot(descent_direction, gradient_vec);

        % Transforming the vector form of descent direction back to matrix form
        tmp_mat = zeros(size);
        tmp_mat(2:size-1, 2:size-1) = reshape(descent_direction, [size-2,size-2]);
        descent_direction = transpose(tmp_mat);
        for i=1:100
            tmp_mat = total_graph + current_step * descent_direction;
            tmp_vec = ravel_graph_transpose(tmp_mat, size);
            step_val = eval_graph(tmp_mat, constraint_graph, size, length)...
                + dot(yk, tmp_vec - zk) + rho * norm(tmp_vec - zk)^2 / 2;
            obj_diff = val_at_x - step_val;

            if step_val - val_at_x > - gamma * current_step * rhs_inner_prod
                current_step = current_step * sigma;
                continue;
            end
    %         constraint_violation_vec
            break;
        end
        
        tmp_mat
        total_graph = tmp_mat;
    end
    
    updated_x = ravel_graph_transpose(total_graph, size);
    updated_z = -(2 * yk / rho - updated_x);
    constraint_vec = ravel_graph_transpose(constraint_graph, size);
    updated_z(updated_z < constraint_vec) = constraint_vec(updated_z < constraint_vec);
    updated_y = yk + rho * (updated_x - updated_z);
    
    graph = total_graph;
%     graph
    obj_val = eval_graph(total_graph, constraint_graph, size, length);
   
end

function vec = ravel_graph_transpose(graph, size)
    graph_transpose = transpose(graph(2:size-1, 2:size-1));
    vec = graph_transpose(:);
end