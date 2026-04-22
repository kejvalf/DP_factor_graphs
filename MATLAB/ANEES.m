function [ANEES]= ANEES(estimated_states,true_states_to_compare,estimated_variances,model)
    N = size(estimated_states,2);
    nees_values = zeros(N, 1);
    for i = 1:N
        P_i = estimated_variances{i};
        error_i = true_states_to_compare(:, i) - estimated_states(:, i);
        nees_values(i) = error_i' * (P_i \ error_i);
    end
    ANEES = mean(nees_values);
    alpha = 0.05; % 95% konfidenční interval
    lower_bound = chi2inv(alpha/2, N * model.nx) / N;
    upper_bound = chi2inv(1 - alpha/2, N * model.nx) / N;

    fprintf('\nMetrika konzistence ANEES:\n');
    fprintf('  Hodnota ANEES = %.4f\n', ANEES);
    fprintf('  Očekávaná hodnota (dimenze stavu nx) = %d\n', model.nx);
    fprintf('  95%% konfidenční interval: [%.4f, %.4f]\n', lower_bound, upper_bound);

    if ANEES >= lower_bound && ANEES <= upper_bound
        fprintf('  >> VÝSLEDEK: Estimátor je konzistentní.\n');
    elseif ANEES < lower_bound
       fprintf('  >> VÝSLEDEK: Estimátor je pesimistický (nadhodnocuje svou nejistotu).\n');
    else
       fprintf('  >> VÝSLEDEK: Estimátor je optimistický (podhodnocuje svou nejistotu).\n');
    end
end