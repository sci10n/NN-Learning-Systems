GWXSIZE = 10;
GWYSIZE = 15;

actions = 4;
% \alpha = learning rate (good = 0.1-0.5)
learning = 0.3;
% \gamma = discount factor (good = 0.9)
discount = 0.9;
% \epsilon = expolration factor (good 1 in start and 0 in end)
exploration = 0.1;
done = 0;
Q = zeros(GWXSIZE,GWYSIZE, actions);
Q(1,:,2) = -Inf;
Q(GWXSIZE,:,1) = -Inf;
Q(:,1,4) = -Inf;
Q(:,GWYSIZE,3) = -Inf;
numEpisodes = 1000;
maxEpisodes = numEpisodes;

while numEpisodes > 0
    gwinit(4);
    start_state = gwstate();
    current_state = start_state;
    
    %exploration = numEpisodes/maxEpisodes;
    numEpisodes = numEpisodes - 1;
    while current_state.isterminal == 0
        
        if exploration > rand()
            % take action a_j and observe reward r and next state s_k+1
            action  = sample([1 2 3 4], [0.25 0.25 0.25 0.25]);
        else
            [~,I] = max(Q,[],3);
            action = I(current_state.pos(1),current_state.pos(2));
        end
        
        result_state = gwaction(action);
        
        % Deal with bad actions
        while result_state.isvalid == 0
            action  = sample([1 2 3 4], [0.25 0.25 0.25 0.25]);
            result_state = gwaction(action);
        end
        
        if ~result_state.isterminal
            % update estimated q-function Q(s_k,a_j) =
            r = result_state.feedback;
            update = r + discount * max(Q(result_state.pos(1), result_state.pos(2),:));
            Q(current_state.pos(1), current_state.pos(2), action) = (1 - learning) * Q(current_state.pos(1), current_state.pos(2), action) + learning * update;
            current_state = result_state;
        else
            Q(current_state.pos(1),current_state.pos(2), action) = 0;
            current_state = result_state;
            break;
        end
        
    end
    
end

gwdraw();
            [~,I] = max(Q,[],3);
            for x = 1:GWXSIZE
                for y = 1:GWYSIZE
                    bestaction = I(x,y);
                    gwplotarrow([x,y],bestaction);
                end
            end

figure(3); clf;
for i = 1:4
    subplot(2,2,i);imagesc(Q(:,:,i)); colorbar;
end
