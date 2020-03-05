function SINR = compute_SINR(RSRP, num_drones, N0, BW)
% Thermal noise power W
N = BW*N0;

RSRP = 10.^(.1*RSRP);
interference = repmat(RSRP,1,1,num_drones);

for i=1:num_drones
    interference(:,i,i) = 0;
end

interference = sum(interference,2);

SINR = 10*log10(RSRP) - 10*log10(N+permute(interference,[1 3 2]));

end
