n = 20; % Size of the matrix
length = 1 / (n - 1);
iter_count = 10000;
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

for iter=1:iter_count
    if mod(iter, 100) == 0
        fprintf("iteration count: %d\n", iter);
    end
    num_grad_mat = get_graph_gradient(total_graph, active_mask, constraint_graph, n, length, gradient_diff);
%     num_grad_mat
    step = step_size * num_grad_mat;
    total_graph = total_graph - step; 
%     total_graph
end

surf(X,Y,total_graph);

% The arrangement of x1, x2, x3, and x4 is as follows
% The length is the length of the square
% x1-x4
% | \ |
% x2-x3
function val = eval_square(x1, x2, x3, x4, length)
    val = 1/2 * sqrt( ((x2 - x1)^2 + length^2)*((x2 - x3)^2 + length^2) - ((x2 - x3)*(x2 - x1))^2 ) ...
        + 1/2 * sqrt( ((x4 - x1)^2 + length^2)*((x4 - x3)^2 + length^2) - ((x4 - x3)*(x4 - x1))^2 );
end

% The arrangement of the square is as follows
% Here we calculate the sum of the area of the triangulars that
% uses x5 as a pivot. x5 is the @total_graph(i,j) here
% x1-x4-x7
% | \| \|
% x2-x5-x8
% | \| \|
% x3-x6-x9
function val = eval_triag_area_relative_to_point(i, j, total_graph, length)
    val = 0;
    tmp = [0,0;1,0;1,1;0,1]; % Used to simplify the operation of calculating the triangular connecting to a point
%     tmp
    for count = 1:4
        x1 = total_graph(i + tmp(count,1) - 1, j + tmp(count,2) - 1);
        x2 = total_graph(i + tmp(count,1), j + tmp(count,2) - 1);
        x3 = total_graph(i + tmp(count,1), j + tmp(count, 2));
        x4 = total_graph(i + tmp(count,1) - 1, j + tmp(count,2));
%         x1
%         x2
%         x3
%         x4
        val = val + eval_square(x1, x2, x3, x4, length);
    end
end


% Calculating numerical gradient.
% The gradient w.r.t xi is calculated by calculating 
% f(x1,x2,...,xi - @graident_diff,...,xn), f(x1,x2,...,xi,...,xn), 
% and f(x1,x2,...,xi + @graident_diff,...,xn) and use them as input
% of matlab function @gradient. The @gradient function will then calculate
% the gradient w.r.t xi 
function numerical_grad = get_graph_gradient(total_graph, active_mask, constraint_graph, size, length, gradient_diff)
    numerical_grad = zeros(size);
    triag_areas_tmp = zeros(2*size,size); % Storing the areas of the triangles
    for i=1:size
        for j=1:size
            if active_mask(i,j) == 1
                samples = zeros(1,3);
                tmp_val = total_graph(i,j);
%                 tmp_val
                for sample_count=-1:1
                    total_graph(i,j) = tmp_val + sample_count * gradient_diff;
                    samples(sample_count + 2) = eval_triag_area_relative_to_point(i,j, total_graph, length);
%                     samples(sample_count + 2)
                end
%                 samples
                grad = gradient(samples, gradient_diff);
                numerical_grad(i, j) = grad(2);
                total_graph(i,j) = tmp_val;
            end
        end
    end
end

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