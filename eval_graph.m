function val = eval_graph(total_graph, constraint_graph, size, length)
    val = 0;
    for i=1:size-1
        for j=1:size-1
            x1 = total_graph(i, j);
            x2 = total_graph(i+1, j);
            x3 = total_graph(i+1, j+1);
            x4 = total_graph(i, j+1);
            val = val + eval_square(x1, x2, x3, x4, length);
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