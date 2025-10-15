function [A, b_exact, x_true] = generate_test_problem(name, n)
    switch lower(name)
        case 'shaw'
            [A, b_exact, x_true] = shaw(n);
        case 'heat'
            [A, b_exact, x_true] = heat(n);
        case 'deriv2'
            [A, b_exact, x_true] = deriv2(n);
        otherwise
            error('Unknown problem name. Use shaw, heat, or deriv2.');
    end
end