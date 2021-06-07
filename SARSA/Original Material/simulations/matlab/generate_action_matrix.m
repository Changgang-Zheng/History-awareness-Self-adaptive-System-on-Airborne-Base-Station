ccc
dbstop if error
min = 0;
max = 10;
step = 1;
n_drones = 2;
% coordinates for 1 drone
[a, b] = meshgrid(min:step:max);
possible_pos = [a(:) b(:)];

% coordinates for 1 drone
idx = nchoosek(1:length(possible_pos), 2);
idx = [idx; circshift(idx, 1, 2)];

states = [possible_pos(idx(:, 1), :) possible_pos(idx(:, 2), :)];

possible_actions = [1 0; -1 0; 0 1; 0 -1; 0 0];

action_matrix = NaN(length(states), length(possible_actions), n_drones);
tic
for state=1:length(states) 
    for drone=1:n_drones
        for action=1:length(possible_actions)
            new_pos = states(state,2 * drone - 1:2 * drone) + possible_actions(action, :);
            new_pos(new_pos > max) = max;
            new_pos(new_pos < min) = min;
            still_drone = states(state, -2*drone + 5: -2*drone+6);
            if isequal(new_pos,still_drone)
                action_matrix(state, action, drone) = state;
            else
                find_str = [new_pos still_drone];
                new_state = find(ismember(states, find_str, 'rows') == 1);                
                action_matrix(state, action, drone) = new_state;
            end
        end
    end
    disp(['completed ' num2str(state) ' out of ' num2str(length(states))]);
end
save states.mat states action_matrix
toc
