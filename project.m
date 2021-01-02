n = 20; % Size of the matrix
length = 1 / (n - 1);
iter_count = 10000;
gradient_diff = 1e-3;
step_size = 0.2;
gradient_tol = 1e-4;

total_graph = zeros(n); % Storing the boundary and values of xi,j
active_mask = zeros(n); % Indicating which points are not boundary 
constraint_graph = zeros(n); % Indicating the inequality contraint on each
                 % element. Current thought is to set the unconstraint
                 % points to be -Inf and the constraint points to be their
                 % constraints

time_val = zeros(iter_count);
constraint_graph = rand(n, n) * 1 + 0.5;
constraint_graph(randsample(n * n, n * n - 10)) = 0;
% for i=2:n-1
%     for j=2:n-1
%         active_mask(i,j) = 1;
%         if constraint_graph(i, j) == 0
%             total_graph(i, j) = 1;
%         else
%             total_graph(i,j) = constraint_graph(i, j);
%         end
%     end
% end

for i=2:n-1
    for j=2:n-1
        active_mask(i, j) = 1;
        total_graph(i, j) = 0;
    end
end

r1 = @(x, y)1 + sin(2 * pi * x);
r2 = @(x, y)1 + cos(1 / (x + 1e-3));
r3 = @(x, y)1/2 - abs(y - 1/2);
r4 = @(x, y)(1 + exp(x * y))^(-1);
r5 = @(x, y)1 + asin(-1 + 2 * sqrt(x * y));
constant_boundary = @(x,y) 0;

total_graph = set_boundary(r3, total_graph, n);
constraint_graph = set_boundary(r3, constraint_graph, n);

gradient_norm_vec = zeros(iter_count);
optimal_gap_vec = zeros(iter_count);

% plot_boundary(r1);

% SGD codes
% for iter=1:iter_count
%     if mod(iter, 100) == 0
%         fprintf("iteration count: %d\n", iter);
%     end
%     num_grad_mat = get_graph_gradient(total_graph, constraint_graph, n, length, gradient_diff);
%     step = step_size * num_grad_mat;
%     total_graph = total_graph - step; 
% end


% Backtracking codes
% sigma = 0.5;
% alpha = 1;
% gamma = 0.05;
% tic
% for iter=1:iter_count
%     [total_graph, obj_diff, obj_val, grad_norm] = armijo(total_graph,...
%          constraint_graph, n, length, gradient_diff, sigma, alpha, gamma);
%     time_val(iter) = toc;
%     gradient_norm_vec(iter) = grad_norm;
%     optimal_gap_vec(iter) = obj_val;
%     if grad_norm < gradient_tol
%         fprintf("calculation ends after %d iterations. Norm of gradient is %f\n", iter, grad_norm);
%         break;
%     end
%     if mod(iter, 100) == 0
%         fprintf("iteration count: %d\n", iter);
%     end
% end


% Backtracking with nesterov codes
% sigma = 0.5;
% alpha = 1;
% gamma = 0.05;
% prev_graph = total_graph;
% tic
% for iter=1:iter_count
%     [new_graph, obj_diff, obj_val, grad_norm] = armijo_nesterov(total_graph,...
%          constraint_graph, n, length, gradient_diff, sigma, alpha, gamma, prev_graph, iter);
%     prev_graph = total_graph;
%     total_graph = new_graph;
%     
%     time_val(iter) = toc;
%     gradient_norm_vec(iter) = grad_norm;
%     optimal_gap_vec(iter) = obj_val;
%     if grad_norm < gradient_tol
%         fprintf("calculation ends after %d iterations. Norm of gradient is %f\n", iter, grad_norm);
%         break;
%     end
%     if mod(iter, 100) == 0
%         fprintf("iteration count: %d\n", iter);
%     end
% end


% Globalized Newton codes
% sigma = 0.5;
% alpha = 1;
% gamma = 0.05;
% beta1 = 1e-6;
% beta2 = 1e-6;
% p = 0.1;
% tic
% for iter=1:iter_count
% %     total_graph
%     [total_graph, obj_diff, obj_val, grad_norm, newton] = globalized_newton(total_graph,...
%     constraint_graph, n, length, gradient_diff, sigma, alpha, gamma, beta1, beta2, p); 
%     time_val(iter) = toc;
% %     total_graph
%     if newton == 1
%         fprintf("Performing newton step\n");
%     else
%         fprintf("Performing armijo step\n");
%     end
% 
%     gradient_norm_vec(iter) = grad_norm;
%     optimal_gap_vec(iter) = obj_val;
%     if grad_norm < gradient_tol
%         fprintf("calculation ends after %d iterations. Norm of gradient is %f\n", iter, grad_norm);
%         break;
%     end
%     
%     if mod(iter, 100) == 0
%         fprintf("iteration count: %d\n", iter);
%     end
% end

% L-BFGS codes
% S = [];
% Y = [];
% sigma = 0.5;
% alpha = 1;
% gamma = 0.05;
% num_history = 0;
% max_history = 10;
% 
% tic
% for iter=1:iter_count
%     [total_graph, obj_diff, obj_val, grad_norm, S, Y] = L_BFGS(total_graph,...
%     constraint_graph, n, length, gradient_diff, sigma, alpha, gamma,...
%     S, Y, num_history, max_history);
%     time_val(iter) = toc;
%     gradient_norm_vec(iter) = grad_norm;
%     optimal_gap_vec(iter) = obj_val;
%     if grad_norm < gradient_tol
%         fprintf("calculation ends after %d iterations. Norm of gradient is %f\n", iter, grad_norm);
%         break;
%     end
%     if mod(iter, 100) == 0
%         fprintf("iteration count: %d\n", iter);
%     end
% end

% Penalty with globalized Newton codes
 sigma = 0.5;
 alpha = 1;
 gamma = 0.05;
 beta1 = 1e-6;
 beta2 = 1e-6;
 p = 0.1;
 a = 0.5;
 constraint_tol = 1e-3;
tic
 for iter=1:iter_count
     [total_graph, obj_diff, obj_val, grad_norm, newton, constraint] = penalty(total_graph,...
     constraint_graph, n, length, gradient_diff, sigma, alpha, gamma, beta1, beta2, p, a, 1e8);
     time_val(iter) = toc;

     gradient_norm_vec(iter) = grad_norm;
     optimal_gap_vec(iter) = obj_val;
     fprintf("Constraint violation is %f, Norm of gradient is %f\n", constraint, grad_norm)
     if (grad_norm < gradient_tol) || (constraint < constraint_tol)
         fprintf("calculation ends after %d iterations", iter);
         break;
     end
    a = a + 5;
     if mod(iter, 100) == 0
         fprintf("iteration count: %d\n", iter);
     end
 end


% admm with globalized Newton codes
% sigma = 0.5;
% alpha = 1;
% gamma = 0.05;
% beta1 = 1e-6;
% beta2 = 1e-6;
% p = 0.1;
% rho = 10;
% zk = transpose(total_graph(2:n-1, 2:n-1));
% zk = zk(:);
% yk = zeros((n - 2)^2, 1);
% constraint_tol = 1e-8;
% prev_graph = total_graph;
% 
% tic
% for iter=1:iter_count
% %     total_graph
%     if rho > 10000.0
%         rho = 10000.0;
%     end
%     
%     prev_graph = total_graph;
%     
%     [total_graph, obj_diff, obj_val, grad_norm,...
%     constraint, zk, yk] = admm(total_graph, constraint_graph, n,...
%     length, gradient_diff, sigma, alpha, gamma, beta1, beta2, p, rho,...
%     zk, yk);
%     time_val(iter) = toc;
% 
%     gradient_norm_vec(iter) = grad_norm;
%     optimal_gap_vec(iter) = obj_val;
%     fprintf("Constraint violation is %f, Norm of gradient is %f\n", constraint, grad_norm);
%     obj_diff
%     if grad_norm < gradient_tol || coonstraint < 1e-4
%         fprintf("calculation ends after %d iterations", iter);
%         break;
%     end
%     
% %     rho = rho * 1.1;
%     if mod(iter, 100) == 0
%         fprintf("iteration count: %d\n", iter);
%     end
% end


% Plot the surface
figure(1);
[X,Y] = meshgrid(0:length:1,0:length:1);
T = delaunay(X,Y);
trisurf(T, X,Y,total_graph);
title('Surface plot');

% Plot norm of the gradient
figure(2);
plot(1:iter, gradient_norm_vec(1:iter));
title('Norm of gradient plot');
xlabel('Iteration number');
ylabel('Norm of gradient');

% Plot objective value
figure(3);
% title('Objective value plot');
plot(1:iter, optimal_gap_vec(1:iter));
grid on;
title('Objective value plot');
xlabel('Iteration number');
ylabel('Objective value');
grid off;

% Plot the surface
figure(4);
[X,Y] = meshgrid(0:length:1,0:length:1);
T = delaunay(X,Y);
trisurf(T, X,Y,constraint_graph);
title('Constraint plot');

figure(5);
plot(time_val(1:iter), optimal_gap_vec(1:iter));
grid on;
title('Objective value vs cpu time');
xlabel('CPU time (seconds)');
ylabel('Objective value');

figure(6);
plot(time_val(1:iter), gradient_norm_vec(1:iter));
grid on;
title('Gradient norm vs cpu time');
xlabel('CPU time (seconds)');
ylabel('Norm of gradient');

% Plotting the boundary
function status = plot_boundary(eval_func)
    status = 1;
    x1 = zeros(1, 100);
    x2 = linspace(0, 1);
    x3 = ones(1, 100);
    y1 = zeros(1, 100);
    y2 = linspace(0, 1);
    y3 = ones(1, 100);
    
    Z = zeros(4, 100);
    for i=1:100
        Z(1, i) = eval_func(x1(i), y2(i));
        Z(2, i) = eval_func(x3(i), y2(i));
        Z(3, i) = eval_func(x2(i), y1(i));
        Z(4, i) = eval_func(x2(i), y3(i));
    end
    
    plot3(x1, y2, Z(1,:));
    hold on;
    plot3(x3, y2, Z(2,:));
    plot3(x2, y1, Z(3,:));
    plot3(x2, y3, Z(4,:));
    hold off;
end

% Using the given @eval_func to calculate the value at the boundary
function graph = set_boundary(eval_func, total_graph, size)
    resolution = 1 / (size - 1);
    for x=1:size
        total_graph(x, 1) = eval_func((x - 1) * resolution, 0);
        total_graph(x, size) = eval_func((x - 1) * resolution, 1);
    end
    
    for y=1:size
        total_graph(1, y) = eval_func(0, (y - 1) * resolution);
        total_graph(size, y) = eval_func(1, (y - 1) * resolution);
    end
    graph = total_graph;
end
