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

% The arrangement of x1, x2, x3, and x4 is as follows
% The length is the length of the square
% x1-x4
% | \ |
% x2-x3
function val = eval_square(x1, x2, x3, x4, length)
    val = 1/2 * sqrt( ((x2 - x1)^2 + length^2)*((x2 - x3)^2 + length^2) - ((x2 - x3)*(x2 - x1))^2 ) ...
        + 1/2 * sqrt( ((x4 - x1)^2 + length^2)*((x4 - x3)^2 + length^2) - ((x4 - x3)*(x4 - x1))^2 );
end

% The arrange of x1, x2, and x3 is as follows
% x1         x1-x2
% | \   or     \|
% x2-x3         x3
function val = eval_triag(x1, x2, x3)
    val = 1;
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
