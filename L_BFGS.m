% The function performs a BFGS step. @S, @Y are matrices with 
% S = [s1,s2,...,sn], Y = [y1,y2,...,yn] where n = @num_history
% @num_history indicates the current iteration number 
% @max_history indicates how many past si and yi we need to store
% The returned value includes the updated S and updated Y
function [graph, obj_diff, obj_val, grad_norm, updated_S, updated_Y] = L_BFGS(total_graph,...
    constraint_graph, size, length, gradient_diff, sigma, alpha, gamma,...
    S, Y, num_history, max_history)

% Setting up the updatd value
    if num_history == 0
        updated_S = zeros((size-2)^2, 1);
        updated_Y = zeros((size-2)^2, 1); 
    else
        if num_history < max_history
            updated_S = zeros((size-2)^2, num_history + 1);
            updated_Y = zeros((size-2)^2, num_history + 1);   
            updated_S(:, 1:num_history) = S;
            updated_Y(:, 1:num_history) = Y;    
        else
            if num_history == max_history
                updated_S = zeros((size-2)^2, max_history);
                updated_Y = zeros((size-2)^2, max_history);   
                updated_S(:, 1:max_history - 1) = S(:, 2:max_history);
                updated_Y(:, 1:max_history - 1) = Y(:, 2:max_history);
            end
        end
    end
    
    gradient = get_graph_gradient(total_graph, constraint_graph, size, length, gradient_diff);

    % If the iteration is 0 and we do not have previous information, simply
    % set hessian to be identity matrix
    if num_history == 0
        hessian = eye((size-2)^2, (size-2)^2);
    else
        s_k_1 = S(:,num_history);
        y_k_1 = Y(:,num_history);
        gamma_k = dot(s_k_1, y_k_1) / dot(y_k_1, y_k_1);
        hessian = gamma_k * eye((size-2)^2, (size-2)^2);
    end
    
    % Transforming the descent direction and gradient to vector form
    gradient_transpose = transpose(gradient(2:size-1, 2:size-1));
    gradient_vec = gradient_transpose(:);
    
    % Estimating the hessian 
    q = gradient_vec;
    A = zeros(num_history, 1);
    rho = zeros(num_history, 1);
    for i=num_history:-1:1
        s_i = S(:,i);
        y_i = Y(:,i);
        rho(i) = dot(s_i, y_i);
        A(i) = dot(s_i, q) / rho(i);
        q = q - A(i) * y_i;
    end
    
    r = hessian * q;
    
    for i=1:num_history
        s_i = S(:,i);
        y_i = Y(:,i);
        beta = rho(i) * dot(y_i, r);
        r = r + (A(i) - beta) * s_i;
    end
    
    % After we get the estimation of the descent direction. Perform armijo
    descent_direction = -r;
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
    
    updated_gradient = get_graph_gradient(graph, constraint_graph, size, length, gradient_diff);
    updated_gradient_transpose = transpose(updated_gradient(2:size-1, 2:size-1));
    updated_gradient_vec = updated_gradient_transpose(:);
    updated_S(:, end) = -current_step * r;
    updated_Y(:, end) = updated_gradient_vec - gradient_vec;
    
    obj_val = val_at_x;
    grad_norm = norm(gradient_vec);
end

