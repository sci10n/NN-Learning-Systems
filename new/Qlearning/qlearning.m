clear all
% \alpha = learning rate (good = 0.1-0.5)
% \gamma = discount factor (good = 0.9)
% \epsilon = expolration factor (good 1 in start and 0 in end)
GWXSIZE = 10;
GWYSIZE = 15;

gwinit(1);
actions = 4;
learning = 0.3;
discount = 0.9;
exploration = 1;
done = 0;
Q = zeros(GWXSIZE,GWYSIZE, actions);
numEpisodes = 10000;
maxEpisodes = numEpisodes;
while numEpisodes > 0
    exploration = numEpisodes/maxEpisodes;
    start_state = gwstate();
    start_state.pos = [ceil(rand() * GWXSIZE),ceil(rand() * GWYSIZE)];
    current_state = start_state;
    % repeat for each step k in the episode
    while done == 0
        % take action a_j and observe reward r and next state s_k+1
        action  = sample([1 2 3 4], [0.25 0.25 0.25 0.25]);
        result_state = gwaction(action);
        
        % Deal with bad actions
        while result_state.isvalid == 0
            action  = sample([1 2 3 4], [0.25 0.25 0.25 0.25]);
            result_state = gwaction(action);
        end
        
        done = result_state.isterminal;

        % update estimated q-function Q(s_k,a_j) =
        maximum_value = max(max(Q(result_state.pos(1), result_state.pos(2),:)));
        Q(current_state.pos(1),current_state.pos(2), action) = (1-learning) * Q(current_state.pos(1),current_state.pos(1), action) + learning * (exploration + discount * maximum_value);
        current_state = result_state;
    end
    numEpisodes = numEpisodes - 1;
end
    gwdraw();
    [~,I] = max(Q,[],3);
    for x = 1:GWXSIZE
        for y = 1:GWYSIZE
            % Pick best action from current state
            
            bestaction = I(x,y);
            gwplotarrow([x,y],bestaction);
        end
    end
    

