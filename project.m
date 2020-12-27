n = 20; % Size of the matrix
length = 1 / (n - 1);
iter_count = 4000;
gradient_diff = 1e-3;
step_size = 0.2;

total_graph = zeros(n); % Storing the boundary and values of xi,j
active_mask = zeros(n); % Indicating which points are not boundary 
constraint_graph = zeros(n); % Indicating the inequality contraint on each
                 % element. Current thought is to set the unconstraint
                 % points to be -Inf and the constraint points to be their
                 % constraints


for i=2:n-1
    for j=2:n-1
        active_mask(i,j) = 1;
        total_graph(i,j) = 10;
    end
end

r1 = @(x, y)1 + sin(2 * pi * x);
r2 = @(x, y)1 + cos(1 / x + 1e-3);
r3 = @(x, y)1/2 - abs(y - 1/2);
r4 = @(x, y)(1 + exp(x * y))^(-1);
r5 = @(x, y)1 + asin(-1 + 2 * sqrt(x * y));

total_graph = set_boundary(r1, total_graph, n);

plot_boundary(r1);
[X,Y] = meshgrid(0:length:1,0:length:1);


% SGD codes
% for iter=1:iter_count
%     if mod(iter, 100) == 0
%         fprintf("iteration count: %d\n", iter);
%     end
%     num_grad_mat = get_graph_gradient(total_graph, active_mask, constraint_graph, n, length, gradient_diff);
%     step = step_size * num_grad_mat;
%     total_graph = total_graph - step; 
% end


% Backtracking codes
sigma = 0.5;
alpha = 1;
gamma = 0.05;
for iter=1:iter_count
    if mod(iter, 100) == 0
        fprintf("iteration count: %d\n", iter);
    end
    total_graph = armijo(total_graph, active_mask, constraint_graph, n, length, gradient_diff, sigma, alpha, gamma); 
end



% Plot the surface
surf(X,Y,total_graph);

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