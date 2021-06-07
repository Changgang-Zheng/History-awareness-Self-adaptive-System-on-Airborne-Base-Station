function pl = path_loss(fc, drone_pos, user_pos, directivity_angle)
numDrones = size(drone_pos,1);
numUsers = size(user_pos,1);

c = 3e8;
% distance in m
d = distance(repmat(user_pos(:,1),1,numDrones),repmat(user_pos(:,2),1,numDrones),repmat(user_pos(:,3),1,numDrones),repmat(drone_pos(:,1)',numUsers,1),repmat(drone_pos(:,2)',numUsers,1),repmat(drone_pos(:,3)',numUsers,1));

% free space path loss dB
pl = 20*log10(4*pi*fc.*d/c);

%Computes if a user is inside the angle of drone antenna.
radius = distance(repmat(user_pos(:,1),1,numDrones),repmat(user_pos(:,2),1,numDrones),0,repmat(drone_pos(:,1)',numUsers,1),repmat(drone_pos(:,2)',numUsers,1),0);
angleThresh = drone_pos(:,3)'*tan(directivity_angle/2);
pl(radius>angleThresh) = inf;
