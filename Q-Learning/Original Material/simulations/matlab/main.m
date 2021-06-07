clear all
close all
clc
random_seed = 3;
rng(random_seed,'twister');

% grid limits
min_x = 0;
max_x = 100;
min_y = 0;
max_y = 100;
% step length for the drones in x and y
step = 10;

% possible x and y coordinates for the drones
possible_x = min_x:step:max_x;
possible_y = min_y:step:max_y;

% number of users
num_users = 20;

% standard deviation for the users, normally distributed around the cluster center
std_deviation = 20;

% how far from the edges can the cluster center be positioned
cluster_radius = 30;
% cluster center
cluster_center = [cluster_radius/step - 1 cluster_radius/step - 1] + randi(max_x / step - 2 * cluster_radius/step + 1, 1 , 2);

% Distributes the users around the cluster center according to a Gaussian distribution
user_pos = repmat(possible_x(cluster_center), num_users, 1) + randn([num_users 2]).* std_deviation;

% directivity of the antennae
directivity_angle = pi/3;

%Learning Rate
alpha = 0.3;

%Discount factor
gamma = 0.9;

% bandwidth
BW = 200e3;

% Noise spectral power density
N0 = 10^(-20.4);

% carrier frequency
fc = 2.4e9;

% transmit power (dB)
Pt = 0;

% connectivity threshold  (dB)
thresh = 40;

% Loads the possible states. To generate the possible states, run generate_action_matrix
load states.mat

% Initializes the Q matrix, the number of episodes and the number of iterations
Q = zeros(size(action_matrix));
episodes = 100;
iterations = 20e3;

% Each drone has a fixed height of 30 meters.
drone_z = 30;

% determines the number of drones
num_drones = size(action_matrix, 3);

% vector containing the possible actions: moving one step in either direction or staying still.
possible_actions = [1 0; -1 0; 0 1; 0 -1; 0 0];

% Initializes metric vectors
average_r = NaN(1, episodes);
max_r = zeros(1, episodes);

% How many slices per JSON file to incude
slices_per_file = 100000;
slices = cell(1, episodes * iterations * num_drones/ slices_per_file);

% saves everything to the JSON file
initial_parameters.random_seed = random_seed;
initial_parameters.num_drones = 2;
initial_parameters.num_users = num_users;
initial_parameters.user_positions = user_pos;
initial_parameters.carrier_frequency = fc;
initial_parameters.transmit_power = Pt;
initial_parameters.sinr_threshold = thresh;
initial_parameters.drone_user_capacity = drone_cap;
initial_parameters.step = step;
initial_parameters.x_min = min_x;
initial_parameters.x_max = max_x;
initial_parameters.y_min = min_y;
initial_parameters.y_max = max_y;
initial_parameters.possible_actions = possible_actions;
initial_parameters.learning_rate = alpha;
initial_parameters.total_episodes = episodes;
initial_parameters.iterations_per_episode = iterations;
initial_parameters.discount_factor = gamma;
json.initial_parameters = initial_parameters;

% Initializes conunters for generating the JSON files.
cont_slice = 1;
cont_files = 1;
for episode=1:episodes
    state = randi([1 length(states)]);
    avg_reward = 0;
    for iteration=1:iterations
        for drone=1:num_drones
            action = choose_action(action_matrix, epsilon, state, drone, Q);
            new_state = action_matrix(state, action, drone);
            action_new_state = choose_action(action_matrix, epsilon, new_state, drone, Q);
            drone_pos = [states(new_state, 1:2)*step drone_z; states(new_state, 3:4)*step drone_z];
            [reward, alloc, SINR] = calculate_reward(user_pos, drone_pos, fc, directivity_angle, N0, BW, Pt, thresh);
            Q(state, action, drone) = Q(state, action, drone) + alpha * (reward + gamma * Q(new_state, action_new_state, drone) - Q(state, action, drone));
            state = new_state;
            avg_reward = avg_reward + reward;
            if reward > max_r(episode)
                max_r(episode) = reward;
            end
            slice.episode = episode;
            slice.iteration = iteration;
            slice.Q_matrix = Q(state, :, drone);
            slice.reward = reward;
            slice.SINR = SINR;
            slice.action = action;
            slice.state = states(state, :);
            slice.drone = drone;
            slices{cont_slice} = slice;
            cont_slice = cont_slice + 1;
        end
    end
    if rem(episode, slices_per_file) == 0
        json.initial_parameters.episodes = [1 + episode - slices_per_file episode];
        json.timeslices = slices;
        file = fopen(['simple_scenario' num2str(cont_files) '.json'], 'w');
        fprintf(file, jsonencode(json));
        fclose(file);
        cont_files = cont_files +1;
        cont_slice = 1;
    end
    average_r(episode) = avg_reward / iterations / num_drones;
    disp(['episode ' num2str(episode) ' avg reward: ' num2str(average_r(episode)) ' max reward: ' num2str(max_r(episode))])
    epsilon = 1/exp(episode/20)
end

max_r_exhaustive = 0;
for state=1:length(states)
        drone_pos = [states(state, 1:2)*step drone_z; states(state, 3:4)*step drone_z];
        [reward, alloc] = calculate_reward(user_pos, drone_pos, fc, directivity_angle, N0, BW, Pt, thresh);
        if reward > max_r_exhaustive
            max_r_exhaustive = reward;
        end
end

figure
plot(1:episodes, average_r, '-^', 'displayname', 'Avg reward', 'linewidth', 2);
hold all
plot(1:episodes, max_r, '-x','displayname', 'Max reward', 'linewidth', 2);
plot(1:episodes, max_r_exhaustive*ones(1, episodes), '*', 'displayname', 'Exhaustive Search', 'linewidth', 2)
xlabel("episode");
ylabel("reward");
legend('-dynamiclegend')
grid on
ylim([-inf 18])


function action = choose_action(action_matrix, epsilon, state, drone, Q)
if rand(1) < epsilon
    action = randi([1 size(action_matrix, 2)]);
else
    if sum(nnz(Q(state, :, drone))) < 1
        action = 1 + randi(size(action_matrix(2)));
    else
        [~, action] = max(Q(state, :, drone));
    end
end
end

function [reward, alloc, SINR] = calculate_reward(user_pos, drone_pos, fc, directivity_angle, N0, BW, Pt, thresh)
[alloc, SINR] = allocate_users([user_pos zeros(1, length(user_pos))'], drone_pos, fc, directivity_angle, N0, BW, Pt, thresh);
reward = nnz(alloc);
end

function [alloc, SINR] = allocate_users(user_pos, drone_pos, fc, directivity_angle, N0, BW, Pt, thresh)
num_drones = size(drone_pos, 1);
num_users = size(user_pos, 1);
alloc = zeros(num_users, 1);
PL = path_loss(fc,drone_pos, user_pos, directivity_angle);
RSRP = Pt - PL;
SINR = compute_SINR(RSRP, num_drones, N0, BW);
for drone=1:num_drones
    alloc(SINR(:, drone) > thresh) = drone;
end
end

function ret = plot_scenario(user_pos, drone_pos, directivity_angle, alloc)
num_drones = size(drone_pos, 1);
figure('units','normalized','outerposition',[0 0 1 1]);
scatter(user_pos(alloc == 0,1),user_pos(alloc == 0,2), 75, 'b', 'markerfacecolor', 'b');
hold all
scatter(user_pos(alloc > 0,1),user_pos(alloc > 0,2), 75, 'r', 'markerfacecolor', 'r');
for drone=1:num_drones
    scatter(drone_pos(drone, 1),drone_pos(drone, 2), 150, 'dr', 'markerfacecolor', 'r');
    radius = drone_pos(drone,3)'*tan(directivity_angle/2);
    ang=0:0.01:2*pi;
    xp=radius*cos(ang);
    yp=radius*sin(ang);
    plot(drone_pos(drone,1)+xp,drone_pos(drone,2)+yp,'color','r','linewidth',2);
end
grid on
xlim([0 100])
ylim([0 100])
xlabel('$x$', 'interpreter', 'latex', 'fontsize', 22)
ylabel('$y$', 'interpreter', 'latex', 'fontsize', 22)
end
