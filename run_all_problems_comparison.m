function run_all_problems_comparison()
% RUN_ALL_PROBLEMS_COMPARISON
% Runs all GMRES variants on the 'shaw', 'heat', and 'deriv2' test problems
% and generates a comparative plot of their final solution errors.

%% 1) Setup
problem_list = {'shaw', 'heat', 'deriv2'};
n = 32;
tol = 1e-8;
maxit = n;
lambda = 1e-3;
B_type = 'A_transpose'; % Using B = A'

% Store results
final_errors = zeros(length(problem_list), 4); % 4 methods

%% 2) Loop through problems and run simulations
fprintf('Running all methods on all test problems...\n');
for i = 1:length(problem_list)
    problem_name = problem_list{i};
    fprintf('--- Running Problem: %s ---\n', problem_name);
    
    % Generate problem
    [A, b, x_true] = generate_test_problem(problem_name, n);
    if strcmp(B_type, 'A_transpose')
        B = A';
    else
        % Placeholder for other B definitions
        B = eye(size(A,2));
    end
    DeltaM = 1e-5 * randn(size(B)); % Consistent perturbation
    
    % Run methods and store final error
    [~, err_ab, ~, ~] = ABgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
    [~, err_ba, ~, ~] = BAgmres_nonhybrid_bounds(A, B, b, x_true, tol, maxit, DeltaM);
    [~, err_hab, ~, ~] = ABgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
    [~, err_hba, ~, ~] = BAgmres_hybrid_bounds(A, B, b, x_true, tol, maxit, lambda, DeltaM);
    
    final_errors(i, :) = [err_ab(end), err_ba(end), err_hab(end), err_hba(end)];
end
fprintf('All simulations complete.\n');

%% 3) Plot the results as a bar chart
figure('Name', 'Performance Across Test Problems', 'Position', [100 100 800 500]);

b_chart = bar(final_errors, 'grouped');
set(gca, 'YScale', 'log'); % Use log scale for error
grid on;

% Add labels and title
title('Final Relative Error Across Different Test Problems');
ylabel('Final Relative Error Norm (log scale)');
set(gca, 'XTickLabel', problem_list); % Label x-axis with problem names
xtickangle(30);

legend('non-hybrid AB', 'non-hybrid BA', 'hybrid AB', 'hybrid BA', 'Location', 'NorthEast');

% Add text labels on top of bars
for i = 1:length(b_chart)
    xtips = b_chart(i).XEndPoints;
    ytips = b_chart(i).YEndPoints;
    labels = string(arrayfun(@(x) sprintf('%.2e', x), ytips, 'UniformOutput', false));
    text(xtips, ytips, labels, 'HorizontalAlignment','center',...
        'VerticalAlignment','bottom', 'FontSize', 8);
end

ylim([min(final_errors(:))*0.1, max(final_errors(:))*5]);

end

% --- You need this helper function in the same file or on your MATLAB path ---
function [A, b, x_true] = generate_test_problem(name, n)
% This function remains unchanged.
    switch lower(name)
        case 'shaw'
            [A, b, x_true] = shaw(n);
        case 'heat'
            [A, b, x_true] = heat(n);
        case 'deriv2'
            [A, b, x_true] = deriv2(n);
        otherwise
            error('Unknown problem name. Use shaw, heat, or deriv2.');
    end
end